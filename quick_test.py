import os
import warnings
import logging
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from PIL import Image

# 过滤烦人的警告
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ================= 配置 =================
MODEL_PATH = "checkpoints/llava-v1.5-7b-M2F2-Det"
MODEL_NAME = "llava-v1.5-7b-M2F2-Det"
IMAGE_FILE = "utils/DDVQA_images/test_image.png"
PROMPT_TEXT = "Determine the authenticity of this image and explain why."
# =======================================

def run_inference():
    print(f"\n{'='*20} M2F2-Det Final Fix (Direct Load) {'='*20}")
    print(f"[Info] Environment: PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")
    
    disable_torch_init()

    # 路径检查
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    if not os.path.exists(IMAGE_FILE):
        print("[Info] Downloading sample image...")
        os.makedirs(os.path.dirname(IMAGE_FILE), exist_ok=True)
        os.system(f"wget https://github.com/CHELSEA234/M2F2_Det/raw/main/asset/teaser.png -O {IMAGE_FILE}")

    print(f"[Info] Loading Model directly to VRAM (No 'auto' map)...")
    
    try:
        # 关键修改：device_map=None
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=MODEL_PATH,
            model_base=None,
            model_name=MODEL_NAME,
            load_8bit=False,
            load_4bit=False,
            device_map=None,      # <--- 核心修复点
            device="cuda",        # 指定设备
            use_flash_attn=False
        )
        # 确保模型真的在 GPU 上
        model.to("cuda")
        print("[Info] Model successfully loaded to RTX 5090!")
    except Exception as e:
        print(f"[Error] Loading Failed: {e}")
        return

    print(f"[Info] Processing Image...")
    image = Image.open(IMAGE_FILE).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # 构造 Prompt
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    inp = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT_TEXT
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(f"[Info] Running Inference...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
    print("\n" + "#"*60)
    print("DETECTION RESULT:")
    print("-" * 20)
    print(outputs)
    print("#"*60 + "\n")

if __name__ == "__main__":
    run_inference()