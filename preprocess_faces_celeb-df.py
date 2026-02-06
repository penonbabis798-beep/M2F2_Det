import os
import cv2
import glob
import torch
import math
import multiprocessing as mp
from facenet_pytorch import MTCNN
from PIL import Image  # 关键修复：导入 PIL
from tqdm import tqdm
import time

# ================= 配置区域 =================
# 数据根目录 (Celeb-DF v2)
# 请确保你的解压路径是这个，或者根据实际情况修改
SOURCE_ROOT = "/data/tangchengwen/Deepfake视频检测/Dataset/Celeb-DF-v2"

# 输出路径 (保存人脸裁剪图)
DEST_ROOT = "/data/tangchengwen/Deepfake视频检测/Dataset/Celeb-DF-v2-Faces"

# 采样间隔：每10帧存一张
# Celeb-DF 视频通常较短，10帧一张是合理的，既能保证多样性又不会导致数据量爆炸
FRAME_INTERVAL = 10 

# 使用的显卡ID列表
# 你的服务器有4张卡，全部利用起来
GPU_IDS = [0, 1, 2, 3]
# ===========================================

def process_video_chunk(gpu_id, video_files, queue):
    """
    工作进程：只负责处理，不打印进度条，处理完一个视频向队列发送信号
    """
    # 显存分配策略：避免一下子占满
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # 初始化 MTCNN
        # image_size=224 是通常用于 EfficientNet/ViT 的输入大小
        mtcnn = MTCNN(
            image_size=224, 
            margin=0, 
            keep_all=False, 
            select_largest=True, 
            device=device
        )
    except Exception as e:
        queue.put(('error', f"GPU {gpu_id} Init Error: {str(e)}"))
        return

    for video_path in video_files:
        try:
            # 构建输出目录
            # os.path.relpath 会保留 'Celeb-real/id0_0000' 这样的结构
            rel_path = os.path.relpath(video_path, SOURCE_ROOT)
            video_name_no_ext = os.path.splitext(rel_path)[0]
            output_dir = os.path.join(DEST_ROOT, video_name_no_ext)
            
            # 断点续传检查：如果文件夹存在且非空，跳过
            if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
                queue.put(('success', 1)) # 视为已完成
                continue
                
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % FRAME_INTERVAL == 0:
                    # BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # === 关键修复 ===
                    # 将 numpy 数组转换为 PIL Image
                    # MTCNN save_path 功能需要 PIL 格式才能正常工作
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    save_path = os.path.join(output_dir, f"{frame_count:04d}.png")
                    
                    # 检测并保存
                    try:
                        # save_path 参数会让 mtcnn 直接保存裁剪好的图片
                        mtcnn(frame_pil, save_path=save_path)
                        saved_count += 1
                    except Exception:
                        pass # 偶尔某些帧检测不到脸是正常的，跳过
                
                frame_count += 1
            cap.release()
            
            # 处理完一个视频，发送信号
            queue.put(('success', 1))
            
        except Exception as e:
            # 发送错误信息
            queue.put(('error', f"Error in {os.path.basename(video_path)}: {str(e)}"))

def main():
    # 设置启动方式 (必须使用 spawn 以兼容 CUDA 多进程)
    mp.set_start_method('spawn', force=True)
    
    print(f"正在搜索视频文件: {SOURCE_ROOT} ...")
    # 兼容 mp4 和 avi (Celeb-DF 主要是 mp4)
    all_videos = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi')):
                all_videos.append(os.path.join(root, file))
                
    total_videos = len(all_videos)
    print(f"共找到 {total_videos} 个视频文件。")
    
    if total_videos == 0:
        print("未找到视频，请检查 SOURCE_ROOT 路径是否正确。")
        return

    # 分配任务
    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(total_videos / num_gpus)
    video_chunks = [all_videos[i:i + chunk_size] for i in range(0, total_videos, chunk_size)]
    
    # 创建通信队列
    manager = mp.Manager()
    queue = manager.Queue()
    
    print(f"启动 {num_gpus} 个进程并行处理 (Celeb-DF v2)...")
    
    processes = []
    for i, gpu_id in enumerate(GPU_IDS):
        if i < len(video_chunks):
            p = mp.Process(target=process_video_chunk, args=(gpu_id, video_chunks[i], queue))
            p.start()
            processes.append(p)
            
    # 主进程负责显示进度条
    pbar = tqdm(total=total_videos, desc="Preprocessing Celeb-DF", unit="vid")
    processed_count = 0
    
    while processed_count < total_videos:
        # 检查子进程是否都还活着
        any_alive = any(p.is_alive() for p in processes)
        if not any_alive and queue.empty():
            break
            
        try:
            # 从队列获取消息，设置超时避免死锁
            msg_type, msg_content = queue.get(timeout=1)
            
            if msg_type == 'success':
                processed_count += 1
                pbar.update(1)
            elif msg_type == 'error':
                # 在进度条下方打印错误，不破坏进度条
                pbar.write(f"[WARNING] {msg_content}")
                # 错误也算处理完了（虽然是失败）
                processed_count += 1
                pbar.update(1)
                
        except:
            # 队列空的时候 pass
            pass
            
    pbar.close()
    
    for p in processes:
        p.join()
        
    print(f"所有处理完成！输出目录: {DEST_ROOT}")

if __name__ == "__main__":
    main()