import torch
import os
import glob
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 视频级 Celeb-DF Dataset (每次返回 8 帧) ---
class CelebDFVideoDataset(Dataset):
    def __init__(self, root_dir, transform_cfg, num_frames=8):
        self.root_dir = root_dir
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.num_frames = num_frames
        self.samples = [] # 存储的是视频文件夹路径
        
        print(f"正在扫描数据目录: {root_dir}")
        
        # 1. 真脸 (Celeb-real)
        real_root = os.path.join(root_dir, "Celeb-real")
        real_count = 0
        if os.path.exists(real_root):
            video_folders = [os.path.join(real_root, d) for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))]
            for folder in video_folders:
                # 只有包含图片的文件夹才加入
                if len(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))) > 0:
                    self.samples.append((folder, 0)) # Label 0
                    real_count += 1
        
        # 2. 假脸 (Celeb-synthesis)
        fake_root = os.path.join(root_dir, "Celeb-synthesis")
        fake_count = 0
        if os.path.exists(fake_root):
            video_folders = [os.path.join(fake_root, d) for d in os.listdir(fake_root) if os.path.isdir(os.path.join(fake_root, d))]
            for folder in video_folders:
                if len(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))) > 0:
                    self.samples.append((folder, 1)) # Label 1
                    fake_count += 1

        print(f"\n加载完成! 真脸视频数: {real_count}, 假脸视频数: {fake_count}")
        if len(self.samples) == 0:
            raise RuntimeError("未找到有效数据。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        folder_path, label = self.samples[index]
        imgs = sorted(glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg")))
        frame_count = len(imgs)
        
        # 均匀采样 8 帧
        if frame_count >= self.num_frames:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            indices = np.pad(np.arange(frame_count), (0, self.num_frames - frame_count), mode='edge')
            
        frames = []
        for idx in indices:
            try:
                img = Image.open(imgs[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                frames.append(torch.zeros(3, 224, 224))
                
        video_tensor = torch.stack(frames, dim=0) # [8, 3, 224, 224]
        return video_tensor, label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "/data/tangchengwen/Deepfake视频检测/Dataset/Celeb-DF-v2-Faces"
    CLIP_PATH = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # 强制指向视频级训练的最佳模型
    ckpt_path = 'checkpoints/stage_1_video_level/current_model_35.pth' 
    
    print(f"--- 启动 Celeb-DF v2 时空特征验证 ---")
    model = M2F2Det(
        clip_text_encoder_name=CLIP_PATH,
        clip_vision_encoder_name=CLIP_PATH,
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    ).to(device)

    print(f"正在加载最佳权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model']
    new_state_dict = { (k.replace('module.', '')): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = CelebDFVideoDataset(DATA_ROOT, get_val_cfg(), num_frames=8)
    # 因为输入变大，调小 batch_size 防止 OOM
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    
    y_true, y_scores = [], []
    print("\n开始推理...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing Celeb-DF"):
            imgs = imgs.to(device)
            out = model(imgs, return_dict=True)
            probs = torch.softmax(out['pred'], dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    y_pred = (np.array(y_scores) >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"Celeb-DF v2 验证结果 (Video-Level):")
    print(f"AUC: {auc:.4f}  |  AP: {ap:.4f}  |  ACC: {acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_verify()