from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
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
from datetime import datetime



class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, frame_path_first, frame_path_last, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):
        # metadata = pd.read_csv(metadata_path)
        self.frame_path_first = frame_path_first
        self.frame_path_last = frame_path_last
        
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
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):

        assert self.is_image(self.frame_path_first) and self.is_image(self.frame_path_last)

        frame_first = self.load_image(self.frame_path_first)
        frame_last = self.load_image(self.frame_path_last)

        return {
            "text": "",
            "frame_first": frame_first,
            "frame_last": frame_last,
        }
    

    def __len__(self):
        return 1


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


import argparse

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="处理开始、结束和输出参数")

    # 添加参数
    parser.add_argument('--start', type=str, required=True, help='开始字符串')
    parser.add_argument('--end', type=str, required=True, help='结束字符串')
    parser.add_argument('--output', type=str, required=True, help='输出文件名')

    # 解析参数
    args = parser.parse_args()

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldername = os.path.join('data', f"{current_time}" + f"-{args.output}")
    os.makedirs(foldername, exist_ok=True)


    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    # model_manager.load_models(["/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00001-of-00007.safetensors,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00002-of-00007.safetensors,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00003-of-00007.safetensors,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00004-of-00007.safetensors,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00005-of-00007.safetensors,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00006-of-00007.safetensors,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output/model-00007-of-00007.safetensors".split(",")])
    # model_manager.load_models(["/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/root/paddlejob/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors".split(",")])
    model_manager.load_models(["/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00001-of-00007.bin,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00002-of-00007.bin,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00003-of-00007.bin,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00004-of-00007.bin,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00005-of-00007.bin,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00006-of-00007.bin,/root/paddlejob/Wan14B/models/lightning_logs/version_41/checkpoints/output1/pytorch_model-00007-of-00007.bin".split(',')])
    model_manager.load_models(
                ["/root/paddlejob/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            )
    model_manager.load_models([
        # "/root/chenming/Wan2.1/DiffSynth-Studio/models/lightning_logs/version_23/checkpoints/epoch=9-step=1570.ckpt",
        # "/root/chenming/Wan2.1/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        # "/root/chenming/Wan2.1/DiffSynth-Studio/test.ckpt",
        "/root/paddlejob/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "/root/paddlejob/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)


    dataset = TextVideoDataset(frame_path_first=args.start,
                            frame_path_last=args.end,
                            height=1280,
                            width=720)

    batch = dataset[0]
    text, frame_first, frame_last = batch["text"], batch["frame_first"], batch["frame_last"]
    to_pil = transforms.ToPILImage()

    height, width = 1280, 720
    num_frames = 81
    pil_image = to_pil((frame_first[:, 0].to(torch.float32) + 1) / 2)
    pil_image.save(os.path.join(foldername, 'start.jpg'))

    pil_image_last = to_pil((frame_last[:, 0].to(torch.float32) + 1) / 2)
    pil_image_last.save(os.path.join(foldername, 'end.jpg'))

    pipe.vae = pipe.vae.to(pipe.device)
    image_feature = encode_image(pipe, pil_image, pil_image_last, num_frames, height, width)

    prompt = "一个女生的脸部特写，以匀速进行转头"
    prompt = "一个年轻女孩的脸部特写，她的表情温柔而自信。她缓慢而优雅地转头，眼神明亮，似乎在思考或观察周围的事物。光线洒在脸上，勾勒出柔和的光影，突出她的五官和微笑。整个画面传达出一种宁静与思索的氛围。"
    # prompt = "一个女生的脸部特写，相机以匀速从正脸转到侧脸"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=pil_image,
        image_emb_external=image_feature,
        num_inference_steps=40,
        seed=0, tiled=True,
        height=1280,
        width=720,
    )

    save_video(video, os.path.join(foldername, 'video.mp4'), fps=30, quality=5)


main()