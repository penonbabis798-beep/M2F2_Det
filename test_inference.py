import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from io import BytesIO

# --- 设置 ---
model_path = "./checkpoints/M2F2-Det-Final"
model_base = None # 已合并，不需要指定 base
image_file = "./utils/DDVQA_images/c40/images_all/0_005_010.jpg" # 替换成你的一张真实存在的图片路径
deepfake_encoder_path = "./checkpoints/stage_1/stage1_best.pth" # 别忘了显微镜！

# 提示词 (必须对应 Stage 3 训练时的格式)
query = "Is this video real or fake?"

# --- 加载模型 ---
disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

# 加载 Deepfake Encoder (显微镜)
print("加载 Deepfake Encoder...")
model.load_deepfake_encoder(deepfake_encoder_path)
model.to(device="cuda", dtype=torch.float16)

# --- 处理输入 ---
# 1. 文本
qs = DEFAULT_IMAGE_TOKEN + "\n" + query
conv = conv_templates["v1"].copy() # 或者是 'vicuna_v1'，看你训练用的 template
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

# 2. 图片 (通用视觉)
# 这里你需要一张测试图片
try:
    image = Image.open(image_file).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    # 3. 视频帧 (Deepfake Encoder 需要的输入)
    # M2F2-Det 的逻辑是把图片也当成单帧视频处理，需要 transform
    # 这里直接复用 image_tensor 或者使用 deepfake_encoder 专用的 transform
    # 为了简单演示，我们假设 deepfake_encoder 的输入处理逻辑封装在 model forward 里
    # 或者你需要手动调用 processor (参考 train_deepfake.py 里的 __getitem__)
    
    # 构造输入 tensor
    # 注意：根据 M2F2 代码，images 参数可能需要包含 [clip_image, deepfake_image]
    # 这里简化处理，直接传入 tensor list
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print("\n🔮 模型判定结果:\n", outputs)

except Exception as e:
    print(f"❌ 运行出错 (可能是图片路径不对): {e}")
    print("请确保 image_file 变量指向了一张存在的图片。")