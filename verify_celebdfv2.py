import torch
import os
import glob
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 1. 强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入模型组件
from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 专用于 Celeb-DF 的直读 Dataset ---
class CelebDFScanDataset(Dataset):
    def __init__(self, root_dir, transform_cfg, max_frames_per_video=10):
        self.root_dir = root_dir
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.samples = []
        
        print(f"正在扫描数据目录: {root_dir}")
        
        # 1. 扫描真脸 (Celeb-real)
        real_root = os.path.join(root_dir, "Celeb-real")
        real_count = 0
        if os.path.exists(real_root):
            # 获取所有视频文件夹
            video_folders = [os.path.join(real_root, d) for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))]
            print(f"发现 {len(video_folders)} 个真脸视频文件夹，正在索引图片...")
            
            for folder in tqdm(video_folders, desc="索引真脸"):
                imgs = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
                # 采样: 如果图片太多，只取均匀分布的几张
                if len(imgs) > max_frames_per_video:
                    step = len(imgs) // max_frames_per_video
                    imgs = imgs[::step][:max_frames_per_video]
                
                for img_path in imgs:
                    self.samples.append((img_path, 0)) # Label 0: Real
                    real_count += 1
        else:
            print(f"警告: 未找到 {real_root}")

        # 2. 扫描假脸 (Celeb-synthesis)
        fake_root = os.path.join(root_dir, "Celeb-synthesis")
        fake_count = 0
        if os.path.exists(fake_root):
            video_folders = [os.path.join(fake_root, d) for d in os.listdir(fake_root) if os.path.isdir(os.path.join(fake_root, d))]
            print(f"发现 {len(video_folders)} 个假脸视频文件夹，正在索引图片...")
            
            for folder in tqdm(video_folders, desc="索引假脸"):
                imgs = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
                if len(imgs) > max_frames_per_video:
                    step = len(imgs) // max_frames_per_video
                    imgs = imgs[::step][:max_frames_per_video]
                
                for img_path in imgs:
                    self.samples.append((img_path, 1)) # Label 1: Fake
                    fake_count += 1
        else:
            print(f"警告: 未找到 {fake_root}")

        print(f"\n加载完成!")
        print(f" - 真脸图片数: {real_count}")
        print(f" - 假脸图片数: {fake_count}")
        print(f" - 总计样本数: {len(self.samples)}")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"目录 {root_dir} 下未找到有效数据，请检查路径结构。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"读取错误 {path}: {e}")
            return torch.zeros(3, 224, 224), label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 配置路径 ---
    # 图片根目录 (预处理脚本的输出目录)
    DATA_ROOT = "/data/tangchengwen/Deepfake视频检测/Dataset/Celeb-DF-v2-Faces"
    
    # CLIP 路径 (自动沿用)
    CLIP_PATH = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # 权重路径 (自动寻找最佳或最新)
    ckpt_path = 'checkpoints/stage_1/best_model.pth' 
    if not os.path.exists(ckpt_path):
        files = glob.glob("checkpoints/stage_1/current_model_*.pth")
        if files:
            ckpt_path = max(files, key=os.path.getctime)
    
    print(f"--- 启动 Celeb-DF v2 跨库验证 ---")
    print(f"数据路径: {DATA_ROOT}")
    print(f"模型权重: {ckpt_path}")

    if not os.path.exists(DATA_ROOT):
        print(f"错误: 数据目录不存在 {DATA_ROOT}")
        return

    # --- 2. 初始化模型 ---
    model = M2F2Det(
        clip_text_encoder_name=CLIP_PATH,
        clip_vision_encoder_name=CLIP_PATH,
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    ).to(device)

    # --- 3. 加载权重 ---
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model']
    new_state_dict = { (k.replace('module.', '')): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    
    # 加载 Vision Tower (关键步骤)
    vt_path = './utils/weights/vision_tower.pth'
    if os.path.exists(vt_path):
        vt_weights = torch.load(vt_path, map_location='cpu')
        vt_dict = {k.replace("vision_tower.", ""): v for k, v in vt_weights.items()}
        target_sd = model.clip_vision_encoder.model.state_dict()
        filtered = {k: v for k, v in vt_dict.items() if k in target_sd}
        model.clip_vision_encoder.model.load_state_dict(filtered, strict=False)

    model.eval()

    # --- 4. 准备数据 ---
    # max_frames_per_video=10 表示每个视频只取10张图，加快验证速度
    dataset = CelebDFScanDataset(DATA_ROOT, get_val_cfg(), max_frames_per_video=10)
    
    # 稍微调大 batch_size 加速，因为不训练
    loader = DataLoader(dataset, batch_size=160, shuffle=False, num_workers=8)
    
    y_true = []
    y_scores = []
    
    # --- 5. 推理 ---
    print("\n开始推理...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            out = model(imgs, return_dict=True)
            probs = torch.softmax(out['pred'], dim=1)[:, 1]
            
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            
    # --- 6. 计算结果 ---
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    print("\n" + "="*50)
    print(f"Celeb-DF v2 验证结果:")
    print(f"样本总数: {len(y_true)}")
    print(f"AUC:      {auc:.4f}")
    print(f"AP:       {ap:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_verify()