# # ✅ 优化 CUDA 显存分配，避免碎片化
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # ✅ 避免 NCCL 卡死问题（特别是在 container 里多卡训练）
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_P2P_DISABLE=0
# export NCCL_ASYNC_ERROR_HANDLING=1

# # ✅ 通常推荐在单节点多卡时设置这两个
# export NCCL_SOCKET_IFNAME=^lo,docker0   # 排除 loopback 网络接口
# export NCCL_LAUNCH_MODE=GROUP           # 分布式同步启动

# # ✅ Torch Distributed Debug（遇到 hanging/死锁时开启）
# # export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 只在排查时开启即可

# # ✅ 更快的 cuBLAS + Tensor Core 支持（部分模型对大矩阵有提升）
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# # ✅ 控制 tokenizer 或 data loader 的随机性
# export TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" MASTER_ADDR="0.0.0.0" MASTER_PORT='8044' WORD_SIZE=8 NODE_RANK=0 python \
  examples/wanvideo/train_wan_t2v_vae_control_wm.py \
  --task train \
  --train_architecture full  \
  --dataset_path /root/bos_folder/MIMO_WAN/MIMO_WAN/  \
  --text_encoder_path "/root/paddlejob/workspace/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/root/paddlejob/workspace/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
  --clip_path "/root/paddlejob/workspace/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --output_path ckpts/fs0409/  \
  --dit_path "hybrid_control_dit.ckpt" \
  --steps_per_epoch 200   \
  --max_epochs 40  \
  --learning_rate 1e-5  \
  --accumulate_grad_batches 3  \
  --use_gradient_checkpointing \
  --height 480 \
  --width 832 \
  --num_nodes 1 \
  --use_gradient_checkpointing_offload \
  --training_strategy "deepspeed_stage_3" \
  --dataloader_num_workers 8 \
  --num_clips 797 \
  # 797 263

  # --dit_path "hybrid_control_dit.ckpt" \
  # --dit_path "lightning_logs/version_0/checkpoints/epoch=5-step=300.ckpt.consolidated" \

 # --dit_path "model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,model/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors" \
