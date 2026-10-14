@echo off

lerobot-train ^
  --dataset.repo_id=local/my_smolvla ^
  --dataset.root=E:/WorkSpace/Learn/VLA/Program/lerobot/hf_cache/lerobot/my_dataset/so101_train_20260831_110047 ^
  --policy.type=smolvla ^
  --policy.pretrained_path=E:/WorkSpace/Learn/VLA/Program/lerobot/models/smolvla_base ^
  --policy.push_to_hub=false ^
  --output_dir=outputs/train/my_smolvla ^
  --job_name=my_smolvla_local ^
  --policy.device=cuda ^
  --wandb.enable=false ^
  --steps=1000 ^
  --batch_size=4