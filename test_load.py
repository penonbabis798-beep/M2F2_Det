from llava.model.builder import load_deepfake_model

model_path = "/data/tangchengwen/Deepfake视频检测/M2F2_Det/checkpoints/deepfake-M2F2-Det-Final"
model_name = "llava-v1.5-7b"

try:
    tokenizer, model, image_processor, context_len = load_deepfake_model(
        model_path, None, model_name, device_map='cpu'
    )
    print("加载成功！")
except Exception as e:
    print("加载失败：", str(e))