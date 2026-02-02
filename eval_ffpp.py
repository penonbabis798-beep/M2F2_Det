import os
import cv2
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
from transformers import logging as hf_logging

# 引入 M2F2-Det 模块
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# ================= 核心配置区域 =================
# 1. 模型路径
MODEL_PATH = "checkpoints/llava-v1.5-7b-M2F2-Det"
MODEL_NAME = "llava-v1.5-7b-M2F2-Det"

# 2. 数据集根目录 (根据你的服务器路径自动拼接)
BASE_DIR = "/data/tangchengwen/Deepfake视频检测/Dataset/FFPP"
COMPRESSION = "c23"  # 这里可以改 c40

# 3. 快速测试模式 (设为 None 则跑全量数据集)
# 建议先跑 20 个视频验证流程，没问题了再设为 None 跑全量
MAX_TEST_VIDEOS = 20 

# 4. 每个视频抽多少帧 (标准做法是均匀抽 10 帧，为了快先抽 1 帧)
FRAMES_PER_VIDEO = 1
# ===============================================

# 静音设置
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

def init_model():
    print(f"[Init] Loading M2F2-Det on RTX 5090...")
    disable_torch_init()
    # 针对 5090 的终极修复加载方式
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=MODEL_NAME,
        load_8bit=False,
        load_4bit=False,
        device_map=None,      # 必须为 None
        device="cuda",
        use_flash_attn=False  # 必须关闭
    )
    model.to("cuda")
    print("[Init] Model Loaded Successfully!")
    return tokenizer, model, image_processor

def get_video_paths(root_dir, compression):
    """自动扫描 FF++ 数据集结构"""
    real_dir = os.path.join(root_dir, compression, "original_sequences", "youtube", compression, "videos")
    
    # 伪造方法列表
    fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    fake_paths_dict = {}

    # 扫描真实视频
    real_videos = []
    if os.path.exists(real_dir):
        real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".mp4")]
    
    # 扫描伪造视频
    for method in fake_methods:
        method_dir = os.path.join(root_dir, compression, "manipulated_sequences", method, compression, "videos")
        if os.path.exists(method_dir):
            videos = [os.path.join(method_dir, f) for f in os.listdir(method_dir) if f.endswith(".mp4")]
            fake_paths_dict[method] = videos
            print(f"[Data] Found {len(videos)} videos for {method}")
        else:
            print(f"[Data] Warning: Directory not found for {method}: {method_dir}")

    return real_videos, fake_paths_dict

def extract_face_from_video(video_path, mtcnn, num_frames=1):
    """读取视频 -> 均匀抽帧 -> MTCNN 扣脸"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return []
    
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    faces = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        # MTCNN 检测
        # save_path=None 表示不保存文件，直接返回 tensor
        face_tensor = mtcnn(img_pil) 
        
        if face_tensor is not None:
            # Tensor (C,H,W) -> Numpy -> PIL Image
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
            # 反归一化 (MTCNN 输出是 -1~1)
            face_np = (face_np * 128 + 127.5).clip(0, 255).astype(np.uint8)
            faces.append(Image.fromarray(face_np))
            
    cap.release()
    return faces

def run_inference(model_pkg, face_image):
    tokenizer, model, image_processor = model_pkg
    
    image_tensor = image_processor.preprocess(face_image, return_tensors='pt')['pixel_values'].half().cuda()
    
    # 论文标准 Prompt
    prompt = "Determine the authenticity of this image."
    
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=64
        )
    
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().lower()
    return output

def main():
    # 1. 初始化
    mtcnn = MTCNN(select_largest=True, device="cuda")
    tokenizer, model, processor = init_model()
    model_pkg = (tokenizer, model, processor)
    
    # 2. 获取数据
    print(f"\n[Task] Scanning FF++ ({COMPRESSION})...")
    real_videos, fake_dict = get_video_paths(BASE_DIR, COMPRESSION)
    
    if not real_videos and not fake_dict:
        print("[Error] No videos found! Check your path.")
        return

    # 3. 准备评测列表
    # 结构: [(path, label_type)] -> label_type: 'real' or 'fake'
    test_list = []
    
    # 添加真实视频
    if MAX_TEST_VIDEOS:
        real_videos = real_videos[:MAX_TEST_VIDEOS]
    for p in real_videos:
        test_list.append((p, "real", "Real"))

    # 添加伪造视频
    for method, paths in fake_dict.items():
        if MAX_TEST_VIDEOS:
            paths = paths[:MAX_TEST_VIDEOS]
        for p in paths:
            test_list.append((p, "fake", method))

    print(f"[Task] Total videos to evaluate: {len(test_list)}")
    
    # 4. 循环评测
    correct = 0
    total = 0
    results_by_method = {k: {"total":0, "correct":0} for k in ["Real"] + list(fake_dict.keys())}
    
    pbar = tqdm(test_list, desc="Evaluating")
    for video_path, label, method_name in pbar:
        try:
            faces = extract_face_from_video(video_path, mtcnn, FRAMES_PER_VIDEO)
            if not faces:
                continue # 没检测到人脸，跳过
            
            # 这里简化逻辑：只要有一帧被判为假，就认为是假视频 (视频级聚合)
            video_pred_is_fake = False
            for face in faces:
                out_text = run_inference(model_pkg, face)
                # 简单的关键词匹配
                if "fake" in out_text or "manipulated" in out_text or "not authentic" in out_text or "artificial" in out_text:
                    video_pred_is_fake = True
                    break # 只要有一帧是假，整个视频判为假
            
            # 统计
            is_correct = False
            if label == "real" and not video_pred_is_fake:
                is_correct = True
            elif label == "fake" and video_pred_is_fake:
                is_correct = True
            
            if is_correct:
                correct += 1
                results_by_method[method_name]["correct"] += 1
            
            total += 1
            results_by_method[method_name]["total"] += 1
            
            # 实时更新进度条上的准确率
            pbar.set_postfix({"Acc": f"{correct/total:.2%}"})
            
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            continue

    # 5. 输出最终报表
    print("\n" + "="*50)
    print(f"FINAL RESULTS (FF++ {COMPRESSION})")
    print("="*50)
    print(f"{'Method':<20} | {'Acc':<10} | {'Count'}")
    print("-" * 45)
    
    for method, res in results_by_method.items():
        if res['total'] > 0:
            acc = res['correct'] / res['total']
            print(f"{method:<20} | {acc:.2%}    | {res['total']}")
            
    print("-" * 45)
    print(f"{'OVERALL':<20} | {correct/total:.2%}    | {total}")
    print("="*50)

if __name__ == "__main__":
    main()