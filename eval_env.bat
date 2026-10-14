lerobot-eval ^
  --policy.path=./outputs/train/smolvla_so101_bed/checkpoints/013000/pretrained_model ^
  --env.type=libero ^
  --env.task=libero_object ^
  --eval.n_episodes=30 ^
  --eval.batch_size=10