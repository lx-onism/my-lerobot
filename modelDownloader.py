import os
from huggingface_hub import snapshot_download

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 模型 ID 和本地路径
repo_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
local_dir = "./models/SmolVLM2-500M-Video-Instruct"  # 相对于当前目录

# 开始下载，增加超时时间和重试
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    max_workers=4                 # 限制并发，避免网络拥塞
    #token=False,                   # 不需要登录
    #ignore_patterns=["*.safetensors"],  # 可选：排除某些文件（如果只想下载部分）
)
print("下载完成！")