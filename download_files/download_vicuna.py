import os
import time
from huggingface_hub import snapshot_download

# 强行设置镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "lmsys/vicuna-7b-v1.5"
local_dir = "utils/weights/vicuna-7b-v1.5"

print(f"开始下载 {model_id}...")

while True:
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            # 1.4.0版本中，不需要 resume_download 参数，它默认就是断点续传的
            # 减少并发连接数，防止网络波动导致 peer closed connection
            max_workers=1, 
            token=None # 如果是公开模型不需要token
        )
        print("\n🎉 下载圆满完成！")
        break
    except Exception as e:
        print(f"\n错误: {e}")
        print("网络波动，10秒后自动尝试恢复下载...")
        time.sleep(10)