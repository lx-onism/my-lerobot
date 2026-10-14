# train.ps1
lerobot-train `
  --dataset.repo_id=lerobot/pusht `
  --policy.type=smolvla `
  --policy.push_to_hub=false `
  --output_dir=outputs/train/my_smolvla `
  --policy.device=cuda