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

# 导入模型组件
from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 核心修改：升级为“视频级 (8帧)”的验证数据集 ---
class VideoFolderJSONDataset(Dataset):
    def __init__(self, data_root, transform_cfg, split_fn, target_method=None, num_frames=8):
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
        
        # 均匀采样 8 帧
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
                
        video_tensor = torch.stack(frames, dim=0) # 输出维度: [8, 3, 224, 224]
        return video_tensor, label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动定位 CLIP 路径
    local_clip_path = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

    print(f"--- 初始化视频时空模型 ---")
    model = M2F2Det(
        clip_text_encoder_name=local_clip_path,
        clip_vision_encoder_name=local_clip_path,
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    ).to(device)

    # --- 核心修改：精准指向你训练出来的最好的模型 (Epoch 35) ---
    ckpt_path = 'checkpoints/stage_1_video_level/current_model_35.pth' 
    
    if not os.path.exists(ckpt_path):
        print(f"错误：找不到模型权重文件 {ckpt_path}！请检查路径。")
        return

    print(f"正在加载最佳权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
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
    
    print("\n" + "="*70)
    print(f"{'Method':<20} | {'AUC':<10} | {'AP':<10} | {'ACC':<10}")
    print("-" * 70)
    
    for method in methods:
        dataset = VideoFolderJSONDataset(data_root, get_val_cfg(), val_split, target_method=method, num_frames=8)
        if len(dataset) == 0: continue
            
        # 注意：因为输入变成了 8 帧，为了防止单卡 OOM，将 batch_size 调低到 16 或 8
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Testing {method}", leave=False):
                # imgs: [B, 8, 3, 224, 224]
                out = model(imgs.to(device), return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                y_true.extend(labels.numpy())
                y_scores.extend(probs.cpu().numpy())
        
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            y_pred = (np.array(y_scores) >= 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred)
            
            print(f"{method:<20} | {auc:.4f}     | {ap:.4f}     | {acc:.4f}")
        else:
            print(f"{method:<20} | 无法计算 (数据单一)")

if __name__ == "__main__":
    run_verify()