import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    # "models/lightning_logs/version_1/checkpoints/epoch=0-step=500.ckpt",
    "lightning_logs/version_0/checkpoints/epoch\=5-step\=300.ckpt.consolidated",
    "/root/paddlejob/workspace/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/paddlejob/workspace/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
])

pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

video = pipe(
    prompt="...",
    negative_prompt="...",
    num_inference_steps=50,
    seed=0, tiled=True
)
save_video(video, "video.mp4", fps=30, quality=5)


# model = torch.load('lightning_logs/version_0/checkpoints/epoch\=5-step\=300.ckpt.consolidated')
# model.update(model['state_dict'])
# model.pop('state_dict')
# torch.save(model, 'xxx')