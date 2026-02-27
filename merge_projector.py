import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake

# --- 配置路径 ---
model_base = "./utils/weights/vicuna-7b-v1.5"
projector_path = "./checkpoints/stage_2/mm_projector.bin"
save_path = "./checkpoints/llava-v1.5-7b-deepfake-stage-2-merged"

print(f"🔄 正在加载基座模型: {model_base}")
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
model = LlavaLlamaForCausalLMDeepfake.from_pretrained(
    model_base, 
    low_cpu_mem_usage=False,  
    torch_dtype=torch.float16
)

# [关键修复] 手动修正 Generation Config，避免保存时报错
if model.generation_config is not None:
    print("🛠️  正在修正 Generation Config...")
    # 强制让 do_sample 为 True，因为有 temperature/top_p
    model.generation_config.do_sample = True 
    # 或者如果不想要随机采样，就把 temperature 设为 None
    # model.generation_config.temperature = None
    # model.generation_config.top_p = None

print(f"🔍 正在加载 Stage 2 权重: {projector_path}")
if not os.path.exists(projector_path):
    alt_path = "./checkpoints/stage_2/pytorch_model.bin"
    if os.path.exists(alt_path):
        print(f"⚠️ 未找到 mm_projector.bin，自动切换为: {alt_path}")
        projector_path = alt_path
    else:
        raise FileNotFoundError(f"❌ 找不到权重文件，请检查: {projector_path}")

state_dict = torch.load(projector_path, map_location="cpu")

# --- 智能键名修正 ---
new_state_dict = {}
print("🛠️  开始处理键名...")
for k, v in state_dict.items():
    clean_k = k.replace("module.", "")
    if "deepfake_projector" in clean_k:
        if not clean_k.startswith("model."):
             new_k = "model." + clean_k
        else:
             new_k = clean_k
        new_state_dict[new_k] = v
        print(f"  [保留] {k} -> {new_k} | Shape: {v.shape}")
    elif "mm_projector" in clean_k:
        if v.shape[1] == 2 or v.shape[0] == 2: 
             new_k = clean_k.replace("mm_projector", "deepfake_projector")
             if not new_k.startswith("model."):
                 new_k = "model." + new_k
             new_state_dict[new_k] = v
             print(f"  [修正重命名] {k} -> {new_k} | Shape: {v.shape}")

if len(new_state_dict) == 0:
    print("⚠️ 警告：没有提取到 deepfake_projector 权重！")
else:
    print(f"📦 准备合并 {len(new_state_dict)} 个张量...")
    m, u = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ 权重合并完毕。")

print(f"💾 正在保存合并后的模型到: {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("🎉 合并完成！")