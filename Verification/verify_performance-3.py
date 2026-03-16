import torch
import os
import json
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 终极视频级测试：多片段融合采样 (Multi-Clip) ---
class VideoMultiClipDataset(Dataset):
    def __init__(self, data_root, transform_cfg, split_fn, target_method=None, num_frames=8, num_clips=3):
        self.data_root = data_root
        self.num_frames = num_frames
        self.num_clips = num_clips
        
        with open(split_fn, 'r') as f:
            self.folder_list = json.load(f)
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.samples = []
        methods = [target_method] if target_method else ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
        
        for pair in self.folder_list:
            id1, id2 = pair[0], pair[1]
            real_path = os.path.join(self.data_root, "original_sequences/youtube/c23/videos", id1)
            if os.path.exists(real_path) and len(glob.glob(os.path.join(real_path, "*.png"))) > 0:
                self.samples.append((real_path, 0))
            
            fake_folder_name = f"{id1}_{id2}"
            for m in methods:
                fake_path = os.path.join(self.data_root, "manipulated_sequences", m, "c23/videos", fake_folder_name)
                if os.path.exists(fake_path) and len(glob.glob(os.path.join(fake_path, "*.png"))) > 0:
                    self.samples.append((fake_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        imgs_path = sorted(glob.glob(os.path.join(video_path, "*.png")))
        frame_count = len(imgs_path)
        
        clips_tensor = []
        # 提取 3 个不重叠或均匀分布的 8 帧片段
        for i in range(self.num_clips):
            # 不同的片段起始点偏移
            offset = i * (frame_count // self.num_clips)
            end_offset = (i + 1) * (frame_count // self.num_clips)
            
            if end_offset - offset >= self.num_frames:
                indices = np.linspace(offset, end_offset - 1, self.num_frames, dtype=int)
            else:
                indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int) # 退化为全局均匀
                
            frames = []
            for idx in indices:
                try:
                    img = Image.open(imgs_path[idx]).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames.append(img)
                except Exception as e:
                    frames.append(torch.zeros(3, 224, 224))
            clips_tensor.append(torch.stack(frames, dim=0))
            
        # 返回维度: [3(Clips), 8(Frames), 3, 224, 224]
        return torch.stack(clips_tensor, dim=0), label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_clip_path = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

    print(f"--- 初始化视频时空模型 (Multi-Clip 评估) ---")
    model = M2F2Det(
        clip_text_encoder_name=local_clip_path, clip_vision_encoder_name=local_clip_path,
        deepfake_encoder_name='efficientnet_b4', hidden_size=1792,
    ).to(device)

    # 如果你跑完了带权重的训练，记得把这里的路径改成 stage_1_video_weighted 下的最佳权重
    ckpt_path = 'checkpoints/stage_1/current_model_2.pth' 
    print(f"加载最佳权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model']
    model.load_state_dict({ (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }, strict=False)
    
    vt_path = './utils/weights/vision_tower.pth'
    if os.path.exists(vt_path):
        vt_dict = {k.replace("vision_tower.", ""): v for k, v in torch.load(vt_path, map_location='cpu').items()}
        target_sd = model.clip_vision_encoder.model.state_dict()
        filtered = {k: v for k, v in vt_dict.items() if k in target_sd or f"vision_model.{k}" in target_sd}
        model.clip_vision_encoder.model.load_state_dict(filtered, strict=False)

    model.eval()
    data_root = "/data/tangchengwen/Deepfake视频检测/Dataset/FFPP_Faces"
    val_split = './utils/FFPP_split/test.json'
    methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    print("\n" + "="*85)
    print(f"{'Method':<20} | {'AUC':<10} | {'AP':<10} | {'Best ACC':<10} | {'Opt Thresh':<10}")
    print("-" * 85)
    
    # ================= 1. 分类别独立测试 =================
    for method in methods:
        dataset = VideoMultiClipDataset(data_root, get_val_cfg(), val_split, target_method=method, num_frames=8, num_clips=3)
        if len(dataset) == 0: continue
            
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for clips_batch, labels in tqdm(loader, desc=f"Testing {method}", leave=False):
                B, num_clips, T, C, H, W = clips_batch.shape
                flat_clips = clips_batch.view(-1, T, C, H, W).to(device)
                
                out = model(flat_clips, return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1] 
                video_probs = probs.view(B, num_clips).mean(dim=1)
                
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


    # ================= 2. 混合全量测试 (Aggregate) =================
    print("\n" + "="*85)
    print(f"👉 评估全部混合伪造数据集 (All Methods Mixed - 真假比 1:5)")
    print("-" * 85)
    
    # target_method=None 会自动加载所有类别的假脸
    all_dataset = VideoMultiClipDataset(data_root, get_val_cfg(), val_split, target_method=None, num_frames=8, num_clips=3)
    if len(all_dataset) > 0:
        loader = DataLoader(all_dataset, batch_size=2, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for clips_batch, labels in tqdm(loader, desc="Testing ALL MIXED", leave=False):
                B, num_clips, T, C, H, W = clips_batch.shape
                flat_clips = clips_batch.view(-1, T, C, H, W).to(device)
                
                out = model(flat_clips, return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                video_probs = probs.view(B, num_clips).mean(dim=1)
                
                y_true.extend(labels.numpy())
                y_scores.extend(video_probs.cpu().numpy())
        
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            # 计算默认 0.5 阈值下的 ACC
            acc_05 = accuracy_score(y_true, (np.array(y_scores) >= 0.5).astype(int))
            
            best_acc = 0
            opt_thresh = 0.5
            for thresh in np.arange(0.1, 0.9, 0.01):
                acc = accuracy_score(y_true, (np.array(y_scores) >= thresh).astype(int))
                if acc > best_acc:
                    best_acc = acc
                    opt_thresh = thresh
            
            print(f"{'ALL MIXED':<20} | {auc:.4f}     | {ap:.4f}     | {best_acc:.4f}     | {opt_thresh:.2f}")
            print(f">>> [重要指标] 默认阈值 (0.5) 下的整体 ACC: {acc_05:.4f}")
            print("="*85)

if __name__ == "__main__":
    run_verify()
    