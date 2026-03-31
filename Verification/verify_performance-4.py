import torch
import os
import json
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 1. 强制离线模式 & 显卡设置
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入模型组件 (确保你已经按照上一步修改了 model.py 增加了频域分支)
from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 视频级评估 Dataset (针对单帧模型)：每个视频采样 15 帧 ---
class VideoLevelSingleFrameDataset(Dataset):
    def __init__(self, data_root, transform_cfg, split_fn, target_method=None, num_frames=15):
        self.data_root = data_root
        self.num_frames = num_frames
        
        with open(split_fn, 'r') as f:
            self.folder_list = json.load(f)
        
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.samples = []
        
        methods = [target_method] if target_method else ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
        
        for pair in self.folder_list:
            id1, id2 = pair[0], pair[1]
            
            # 1. 真脸视频
            real_path = os.path.join(self.data_root, "original_sequences/youtube/c23/videos", id1)
            if os.path.exists(real_path) and len(glob.glob(os.path.join(real_path, "*.png"))) > 0:
                self.samples.append((real_path, 0)) # Label 0
            
            # 2. 假脸视频
            fake_folder_name = f"{id1}_{id2}"
            for m in methods:
                fake_path = os.path.join(self.data_root, "manipulated_sequences", m, "c23/videos", fake_folder_name)
                if os.path.exists(fake_path) and len(glob.glob(os.path.join(fake_path, "*.png"))) > 0:
                    self.samples.append((fake_path, 1)) # Label 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        imgs_path = sorted(glob.glob(os.path.join(video_path, "*.png")))
        frame_count = len(imgs_path)
        
        # 均匀采样 num_frames 帧
        if frame_count >= self.num_frames:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            indices = np.pad(np.arange(frame_count), (0, self.num_frames - frame_count), mode='edge')
            
        frames = []
        for idx in indices:
            try:
                img = Image.open(imgs_path[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                frames.append(torch.zeros(3, 224, 224))
                
        video_tensor = torch.stack(frames, dim=0) # 输出维度: [15, 3, 224, 224]
        return video_tensor, label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动定位 CLIP 路径
    local_clip_path = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

    print(f"--- 初始化空频双流感知大模型 (Spatial-Frequency Dual-Stream) ---")
    model = M2F2Det(
        clip_text_encoder_name=local_clip_path,
        clip_vision_encoder_name=local_clip_path,
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    ).to(device)

    # --- 自动寻找最新/最好的权重 ---
    # 假设你按照上一步的命名为 stage_1_freq_spatial_dual
    ckpt_dir = 'checkpoints/stage_1_spatial'
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    
    if not os.path.exists(ckpt_path):
        list_of_files = glob.glob(f'{ckpt_dir}/current_model_*.pth')
        if list_of_files:
            ckpt_path = max(list_of_files, key=os.path.getctime) # 找最新的
        else:
            print(f"错误：找不到模型权重文件！请检查 {ckpt_dir} 目录是否存在。")
            return

    print(f"正在加载最优权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 权重加载逻辑
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else (checkpoint['model'] if 'model' in checkpoint else checkpoint)
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    
    # Vision Tower 匹配
    vt_path = './utils/weights/vision_tower.pth'
    if os.path.exists(vt_path):
        vt_weights = torch.load(vt_path, map_location='cpu')
        vt_dict = {k.replace("vision_tower.", ""): v for k, v in vt_weights.items()}
        target_sd = model.clip_vision_encoder.model.state_dict()
        filtered = {k: v for k, v in vt_dict.items() if k in target_sd or f"vision_model.{k}" in target_sd}
        model.clip_vision_encoder.model.load_state_dict(filtered, strict=False)

    model.eval()

    # 验证逻辑
    data_root = "/data/tangchengwen/Deepfake视频检测/Dataset/FFPP_Faces"
    val_split = './utils/FFPP_split/test.json'
    methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    print("\n" + "="*85)
    print(f"{'Method':<20} | {'AUC':<10} | {'AP':<10} | {'Best ACC':<10} | {'Opt Thresh':<10}")
    print("-" * 85)
    
    # ================= 1. 分类别独立测试 =================
    for method in methods:
        dataset = VideoLevelSingleFrameDataset(data_root, get_val_cfg(), val_split, target_method=method, num_frames=15)
        if len(dataset) == 0: continue
            
        # batch_size=4 表示一次处理 4 个视频 (60 张图片)，单卡 32G 显存完全足够
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs_batch, labels in tqdm(loader, desc=f"Testing {method}", leave=False):
                # imgs_batch: [B, 15, 3, 224, 224]
                B, N, C, H, W = imgs_batch.shape
                # 摊平为单帧送入网络
                flat_imgs = imgs_batch.view(-1, C, H, W).to(device)
                
                out = model(flat_imgs, return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1] # [B * 15]
                
                # [核心] 把 15 帧的概率求平均，得到极其稳定的视频级概率
                video_probs = probs.view(B, N).mean(dim=1)
                
                y_true.extend(labels.numpy())
                y_scores.extend(video_probs.cpu().numpy())
        
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            best_acc = 0
            opt_thresh = 0.5
            for thresh in np.arange(0.1, 0.9, 0.01):
                acc = accuracy_score(y_true, (np.array(y_scores) >= thresh).astype(int))
                if acc > best_acc:
                    best_acc = acc
                    opt_thresh = thresh
            
            print(f"{method:<20} | {auc:.4f}     | {ap:.4f}     | {best_acc:.4f}     | {opt_thresh:.2f}")
        else:
            print(f"{method:<20} | 无法计算 (数据单一)")

    # ================= 2. 混合全量测试 (还原 1:5 的真实分布) =================
    print("\n" + "="*85)
    print(f"👉 评估全部混合伪造数据集 (All Methods Mixed)")
    print("-" * 85)
    
    all_dataset = VideoLevelSingleFrameDataset(data_root, get_val_cfg(), val_split, target_method=None, num_frames=15)
    if len(all_dataset) > 0:
        loader = DataLoader(all_dataset, batch_size=4, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs_batch, labels in tqdm(loader, desc="Testing ALL MIXED", leave=False):
                B, N, C, H, W = imgs_batch.shape
                flat_imgs = imgs_batch.view(-1, C, H, W).to(device)
                out = model(flat_imgs, return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                video_probs = probs.view(B, N).mean(dim=1)
                
                y_true.extend(labels.numpy())
                y_scores.extend(video_probs.cpu().numpy())
        
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            acc_05 = accuracy_score(y_true, (np.array(y_scores) >= 0.5).astype(int))
            
            best_acc = 0
            opt_thresh = 0.5
            for thresh in np.arange(0.1, 0.9, 0.01):
                acc = accuracy_score(y_true, (np.array(y_scores) >= thresh).astype(int))
                if acc > best_acc:
                    best_acc = acc
                    opt_thresh = thresh
            
            print(f"{'ALL MIXED':<20} | {auc:.4f}     | {ap:.4f}     | {best_acc:.4f}     | {opt_thresh:.2f}")
            print(f"🌟 [重要] 默认阈值 (0.5) 下的整体 ACC: {acc_05:.4f}")
            print("="*85)

if __name__ == "__main__":
    run_verify()