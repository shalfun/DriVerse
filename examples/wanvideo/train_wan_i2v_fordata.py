import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
from torch.utils.data import default_collate
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import transforms


def custom_collate(batch):
    # print("batch size = ", len(batch))
    # for item in batch:
    #     if item['video'] is None:
    #         print(item)
    #         dist.barrier()
    #     print(dist.get_rank(), item['text'], item['path'])
    #     print(dist.get_rank(), item['video'].shape)
    batch = [item for item in batch if item is not None]  # 过滤 None 值
    return default_collate(batch)  # 使用默认的 collate 函数

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        # if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
        #     reader.close()
        #     return None
        
        num_frames_actual = min(num_frames, reader.count_frames() - start_frame_id)

        frames = []
        for frame_id in range(num_frames_actual):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)

        if num_frames_actual < num_frames:
            for frame_id in range(num_frames - num_frames_actual):
                frame = reader.get_data(num_frames_actual-1)
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame)
                frame = frame_process(frame)
                frames.append(frame)
            
        # print("Len frames = ", len(frames))

        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            video = self.load_image(path)
            dwpose = ""
        else:
            video = self.load_video(path)
            dwpose_path = path.replace('/videos/', '/dwpose/')
            if os.path.exists(dwpose_path):
                dwpose = self.load_video(path.replace('/videos/', '/dwpose/'))
            else:
                dwpose = ""
        
        data = {"text": text, "video": video, "dwpose": dwpose, "path": path}
        if video is None:
            print(data)
            exit(0)
        return data
    

    def __len__(self):
        return len(self.path)



def encode_image(pipe, image, image_last, num_frames, height, width):
    image = pipe.preprocess_image(image.resize((width, height))).to(pipe.device)
    image_last = pipe.preprocess_image(image_last.resize((width, height))).to(pipe.device)
    
    clip_context = pipe.image_encoder.encode_image([image])   ### clip to extract feature
    msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device) # 1xNxHxW
    msk[:, 1:] = 0 
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1) # 1x(N+3)xHxW
    msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8) # 1x(N+3)//4x4xHxW
    msk = msk.transpose(1, 2)[0] # 4x(N+3)//4xHxW
    msk[:, -1, :, :] = 1

    vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-2, height, width).to(image.device), image_last.transpose(0, 1)], dim=1)
    y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device)[0]
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
    y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
    return {"clip_feature": clip_context, "y": y}


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path, vae_path])
        model_manager.load_models(
            ["/root/paddlejob/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
            torch_dtype=torch.float32, # Image Encoder is loaded with float32
        )
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        
        self.to_pil = transforms.ToPILImage()
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, dwpose, path = batch["text"][0], batch["video"], batch["dwpose"], batch["path"][0]
        video = video.to(torch.bfloat16)
        if type(dwpose) is not list:
            dwpose = dwpose.to(torch.bfloat16)
        
        # print(batch_idx, "Text: ", text is None, "; Path: ", path is None, "; Video: ", video is None)
        self.pipe.device = self.device
        if video is not None:
            prompt_emb = self.pipe.encode_prompt(text)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            if type(dwpose) is not list:
                latents_dwpose = self.pipe.encode_video(dwpose, **self.tiler_kwargs)[0]
            else:
                latents_dwpose = ""
            
            # latents_first, latents_last = [], []
            # for i in range (4):
            #     latents_image = self.pipe.encode_video(video[:, :, 0:i+1], **self.tiler_kwargs)[0] #1-4, same length
            #     latents_first.append(latents_image)
            #     latents_image = self.pipe.encode_video(video[:, :, -2-i:-1], **self.tiler_kwargs)[0] #1-4, same length
            #     latents_last.append(latents_image)

            _, _, num_frames, height, width = video.shape
            pil_image = self.to_pil((video[0, :, 0].to(torch.float32) + 1) / 2)
            pil_image_last = self.to_pil((video[0, :, -1].to(torch.float32) + 1) / 2)
            image_feature = encode_image(self.pipe, pil_image, pil_image_last, num_frames, height, width)
            # data = {"latents": latents, "latents_dwpose": latents_dwpose, "latents_first": latents_first, "latents_last": latents_last, "prompt_emb": prompt_emb}
            data = {"latents": latents, "prompt_emb": prompt_emb}
            data = data | image_feature

            # torch.save(data, path.replace('/videos/', '/wanx_tensors/') + ".tensors.pth")
            torch.save(data, path + ".tensors.pth")


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        print(metadata_path)
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i.replace('/videos/', '/wanx_tensors/') + ".tensors.pth" for i in self.path if os.path.exists(i.replace('/videos/', '/wanx_tensors/') + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(self, dit_path, learning_rate=1e-5, lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", use_gradient_checkpointing=True, total_iters=None):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
            self.pipe.denoising_model().freeze_for_controlnet()
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
        assert total_iters is not None, "total_iters must be specified."
        self.total_steps = total_iters
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming"):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        latents_dwpose = batch["latents_dwpose"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = [prompt_emb["context"][0][0].to(self.device)]
        
        # Loss
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            noise_pred = self.pipe.denoising_model()(
                noisy_latents, tracking_maps=latents_dwpose, timestep=timestep, **prompt_emb, **extra_input,
                use_gradient_checkpointing=self.use_gradient_checkpointing
            )
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
            loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss



    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        
        # Define the optimizer
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)

        # 设置热身和总步骤
        warmup_steps = 100  # 根据需要调整

        # 定义学习率调度器
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # 线性热身
            else:
                input_value = torch.tensor((step - warmup_steps) / (self.total_steps - warmup_steps) * 3.141592653589793)
                return 0.5 * (1 + torch.cos(input_value))
            
        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 在每个训练步骤后更新
            },
        }
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=15,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for each iteration.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        collate_fn=custom_collate,
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        total_iters=args.steps_per_epoch * args.max_epochs
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
