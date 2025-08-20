from huggingface_hub import snapshot_download

# 设置模型名称
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 下载模型到本地目录
local_dir = "./models/DeepSeek-R1-Distill-Qwen-1.5B"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    revision="main",  # 模型版本（默认 main）
    force_download=True  # 强制重新下载
)

print(f"模型已下载到：{local_dir}")