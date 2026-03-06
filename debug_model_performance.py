import torch
import os
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def final_repair_and_test():
    # --- 路径配置 ---
    model_path = "/data/tangchengwen/Deepfake视频检测/M2F2_Det/checkpoints/deepfake-M2F2-Det-Final"
    image_path = "./utils/DDVQA_images/c40/images_all/0_006_002.jpg"
    
    # 这里填入你刚才 find 到的 CLIP 绝对路径（去掉最后的文件名）
    clip_local_path = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/"

    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 {image_path}")
        return

    # 1. 加载基础架构 (开启离线模式)
    print("🏗️  1. 正在加载基础架构 (离线模式)...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
        local_files_only=True  # 强制不连接 Hugging Face
    ).cuda()

    # 2. 检查 Deepfake 模块
    print("🛠️  2. 检查 Deepfake 模块状态...")
    if hasattr(model.get_model(), 'initialize_deepfake_modules'):
        try:
            model.get_model().initialize_deepfake_modules(model.config)
            model.cuda()
            print("✅ 手动激活 Deepfake 模块成功")
        except Exception as e:
            print(f"⚠️ 激活函数调用失败 (可能已自动激活): {e}")

    # 3. 挂载专用权重
    print("🔗 3. 挂载专用权重...")
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        extra_weights = torch.load(non_lora_path, map_location="cpu")
        cleaned_weights = {}
        for k, v in extra_weights.items():
            k = k.replace("base_model.model.", "")
            cleaned_weights[k] = v
        model.load_state_dict(cleaned_weights, strict=False)
        print(f"✅ non_lora_trainables.bin 挂载成功")

    # 4. 同步视觉塔并加载处理器
    print("⚙️  4. 准备图像处理器 (本地路径)...")
    vision_tower = model.get_model().get_vision_tower()
    
    # 强制让视觉塔从本地路径加载，不走网络
    if not vision_tower.is_loaded:
        # 修改 vision_tower 的路径指向本地
        vision_tower.vision_tower_name = clip_local_path
        vision_tower.load_model()
    
    vision_tower.to(device='cuda', dtype=torch.float16)
    
    # 显式从本地路径加载处理器
    image_processor = CLIPImageProcessor.from_pretrained(clip_local_path, local_files_only=True)

    # 5. 处理图像文件
    print(f"🖼️  5. 正在处理图片: {image_path}")
    raw_image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.half().cuda()

    # 6. 推理测试
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    
    prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\nPlease analyze this image carefully and determine if it is a deepfake. If it is, explain why. ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    print("\n🚀 6. 开始最终推理验证...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            top_p=1.0
        )
    
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n✨ 模型判定输出：\n{result}")

if __name__ == "__main__":
    # 运行前设置环境变量，双重保险
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    final_repair_and_test()