#!/bin/bash
# ============================================
#  LeRobot SmolVLA 离线训练脚本
#  功能：设置离线环境变量并启动训练
# ============================================

# ----- 离线训练必需的环境变量 -----
# 禁止 Hugging Face Hub 联网（模型、数据集均从本地加载）
export HF_HUB_OFFLINE=1
# 禁止 Transformers 库联网下载模型配置
export TRANSFORMERS_OFFLINE=1
# 禁止 Datasets 库联网（如果 lerobot 内部用到）
export HF_DATASETS_OFFLINE=1

# 可选：指定缓存目录（便于管理，默认在 ~/.cache/huggingface）
# export HF_HOME=/path/to/your/cache

# 可选：如果使用代理，可以取消注释下面两行
# export http_proxy=http://your-proxy:port
# export https_proxy=http://your-proxy:port

# ----- 打印当前设置（方便调试） -----
echo "🔧 离线模式已启用："
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "开始训练..."

# ----- 执行训练命令 -----
lerobot-train \
  --dataset.repo_id=local/svla_so101 \
  --dataset.root=hf_cache/lerobot/my_dataset/so101_train_B_20260907_173056 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.random_order=true \
  --policy.type=smolvla \
  --policy.load_vlm_weights=false \
  --policy.vlm_model_name=/tmp/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.pretrained_path=outputs/train/smolvla_so101_bed/checkpoints/013000/pretrained_model \
  --policy.push_to_hub=false \
  --policy.use_peft=false \
  --output_dir=outputs/train/smolvla_so101_bedb \
  --job_name=smolvla_so101_3060 \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=11250 \
  --batch_size=32 \
  --num_workers=4

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "✅ 训练完成！"
else
    echo "❌ 训练出错，请检查日志。"
    exit 1
fi