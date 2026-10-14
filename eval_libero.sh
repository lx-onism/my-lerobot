lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --policy.n_action_steps=10 \
  --env.fps=20