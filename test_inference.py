import torch
# 严格引用官方定义的常量
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEEPFAKE_TOKEN_INDEX
try:
    from llava.constants import DEFAULT_DEEPFAKE_TOKEN
except ImportError:
    DEFAULT_DEEPFAKE_TOKEN = "<deepfake>"

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from PIL import Image
import os

# ================= 配置区域 =================
model_path = "./checkpoints/deepfake-M2F2-Det-Final"
model_base = None 
image_file = "./utils/DDVQA_images/c40/images_all/0_005_010.jpg" 
deepfake_encoder_path = "./checkpoints/stage_1/stage1_best.pth" 
# ===========================================

def load_deepfake_weights_robust(model, weight_path):
    print(f"🔧 正在挂载 Deepfake 显微镜权重...")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'): new_key = k.replace('features.', 'deepfake_encoder.')
        elif k.startswith('deepfake_encoder.'): new_key = k
        elif 'classifier' in k: continue
        else: new_key = 'deepfake_encoder.' + k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Deepfake 显微镜权重已挂载")

def load_projector_weights_robust(model, model_path):
    projector_path = os.path.join(model_path, "non_lora_trainables.bin")
    if not os.path.exists(projector_path): return
    print(f"🔗 正在深度匹配 Projector 权重...")
    state_dict = torch.load(projector_path, map_location='cpu')
    model_keys = model.state_dict().keys()
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("base_model.model.", "")
        if clean_key in model_keys: new_state_dict[clean_key] = v
        elif ("model." + clean_key) in model_keys: new_state_dict["model." + clean_key] = v
        else:
            if "mm_projector" in clean_key or "deepfake_projector" in clean_key:
                for mk in model_keys:
                    if clean_key in mk:
                        new_state_dict[mk] = v
                        break
    model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Projector 已激活 (匹配层数: {len(new_state_dict)})")

def encode_prompt_special(prompt, tokenizer):
    """
    【绝对关键】：手动构建 input_ids，确保特殊占位符不会被切词器破坏
    """
    input_ids = [tokenizer.bos_token_id]
    # 使用占位符分割
    parts = prompt.split(DEFAULT_IMAGE_TOKEN)
    for i, part in enumerate(parts):
        sub_parts = part.split(DEFAULT_DEEPFAKE_TOKEN)
        for j, sub_part in enumerate(sub_parts):
            if sub_part:
                # 正常的文本转为 token
                input_ids.extend(tokenizer(sub_part, add_special_tokens=False).input_ids)
            if j < len(sub_parts) - 1:
                input_ids.append(DEEPFAKE_TOKEN_INDEX) # 插入 -201
        if i < len(parts) - 1:
            input_ids.append(IMAGE_TOKEN_INDEX) # 插入 -200
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()

# --- 主程序 ---
try:
    disable_torch_init()
    print(f"⏳ 正在初始化 M2F2-Det 系统...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, get_model_name_from_path(model_path))

    load_deepfake_weights_robust(model, deepfake_encoder_path)
    load_projector_weights_robust(model, model_path)

    # 统一转换 BFloat16 并推流
    model = model.to(device='cuda:0', dtype=torch.bfloat16)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded: vision_tower.load_model()
    vision_tower.to(device='cuda:0', dtype=torch.bfloat16)

    # 构建 Prompt
    query = "Is this video real or fake? Please provide a detailed analysis."
    # M2F2 训练时的标准格式：图像先，深度特征后
    prompt_raw = f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_DEEPFAKE_TOKEN}\n{query}"
    
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], prompt_raw)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = encode_prompt_special(prompt, tokenizer)
    
    print(f"🔢 输入序列检查: {input_ids[0].tolist()[:20]}...")
    print(f"🔍 是否包含图像占位符 (-200): {IMAGE_TOKEN_INDEX in input_ids}")
    print(f"🔍 是否包含伪造占位符 (-201): {DEEPFAKE_TOKEN_INDEX in input_ids}")

    print(f"🖼️  分析图像: {image_file}")
    image = Image.open(image_file).convert('RGB')
    
    print("🚀 推理中 (强制开口模式)...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[image], 
            deepfake_inputs=[image], 
            do_sample=True,     # 必须开启采样
            temperature=0.7,    # 增加随机性防止闭嘴
            top_p=0.95,
            max_new_tokens=512,
            min_new_tokens=1,   # 强行要求至少出一个词
            use_cache=True
        )

    # 提取生成内容
    output_tokens = output_ids[0, input_ids.shape[1]:]
    outputs = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    print("\n" + "✨" * 30)
    print("🔮 M2F2-Det 判定结果：")
    if not outputs:
        print("【警告】：模型仍然没有生成文本。")
        print("这通常意味着注入的视觉特征在注意力机制中被屏蔽了。")
        print(f"生成的原始 Token IDs: {output_tokens.tolist()}")
    else:
        print(outputs)
    print("✨" * 30 + "\n")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ 运行失败: {e}")