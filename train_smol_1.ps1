lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test --policy.push_to_hub=false \
  --policy.type=smolvla --policy.device=cuda \
  --output_dir=outputs/train/smolvla_test \
  --job_name=smolvla_test \
  --batch_size=64 --steps=20000 \
  --wandb.enable=false