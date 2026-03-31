import torch
import os
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

# --- Celeb-DF 单帧均匀采样视频级 Dataset ---
class CelebDFSingleFrameDataset(Dataset):
    def __init__(self, root_dir, transform_cfg, frames_per_video=15):
        self.root_dir = root_dir
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.frames_per_video = frames_per_video
        self.samples = []
        
        print(f"正在扫描数据目录: {root_dir}")
        
        real_root = os.path.join(root_dir, "Celeb-real")
        if os.path.exists(real_root):
            for folder in os.listdir(real_root):
                full_folder = os.path.join(real_root, folder)
                if os.path.isdir(full_folder) and len(glob.glob(os.path.join(full_folder, "*.png")) + glob.glob(os.path.join(full_folder, "*.jpg"))) > 0:
                    self.samples.append((full_folder, 0))
        
        fake_root = os.path.join(root_dir, "Celeb-synthesis")
        if os.path.exists(fake_root):
            for folder in os.listdir(fake_root):
                full_folder = os.path.join(fake_root, folder)
                if os.path.isdir(full_folder) and len(glob.glob(os.path.join(full_folder, "*.png")) + glob.glob(os.path.join(full_folder, "*.jpg"))) > 0:
                    self.samples.append((full_folder, 1))

        print(f"加载完成! 总视频数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        folder_path, label = self.samples[index]
        imgs = sorted(glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg")))
        frame_count = len(imgs)
        
        if frame_count >= self.frames_per_video:
            indices = np.linspace(0, frame_count - 1, self.frames_per_video, dtype=int)
        else:
            indices = np.pad(np.arange(frame_count), (0, self.frames_per_video - frame_count), mode='edge')
            
        frames = []
        for idx in indices:
            try:
                img = Image.open(imgs[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                frames.append(torch.zeros(3, 224, 224))
                
        video_tensor = torch.stack(frames, dim=0) # [15, 3, 224, 224]
        return video_tensor, label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "/data/tangchengwen/Deepfake视频检测/Dataset/Celeb-DF-v2-Faces"
    LOCAL_CLIP_PATH = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    print(f"--- 启动 Celeb-DF v2 验证 (单帧平均策略) ---")
    model = M2F2Det(clip_text_encoder_name=LOCAL_CLIP_PATH, clip_vision_encoder_name=LOCAL_CLIP_PATH, deepfake_encoder_name='efficientnet_b4', hidden_size=1792).to(device)

    # 兼容自动找权重的逻辑
    ckpt_dir = 'checkpoints/stage_1_freq_spatial_dual'
    ckpt_path = ''
    if os.path.exists(ckpt_dir):
        list_of_files = glob.glob(f'{ckpt_dir}/current_model_*.pth')
        if list_of_files: ckpt_path = max(list_of_files, key=os.path.getctime)
            
    if not ckpt_path or not os.path.exists(ckpt_path):
        list_of_files = glob.glob('checkpoints/stage_1_spatial/current_model_*.pth') 
        if list_of_files: ckpt_path = max(list_of_files, key=os.path.getctime)

    print(f"正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else (checkpoint['model'] if 'model' in checkpoint else checkpoint)
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = CelebDFSingleFrameDataset(DATA_ROOT, get_val_cfg(), frames_per_video=15)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    
    y_true, y_scores = [], []
    with torch.no_grad():
        for imgs_batch, labels in tqdm(loader, desc="Testing Celeb-DF"):
            B, N, C, H, W = imgs_batch.shape
            flat_imgs = imgs_batch.view(-1, C, H, W).to(device)
            out = model(flat_imgs, return_dict=True)
            probs = torch.softmax(out['pred'], dim=1)[:, 1]
            video_probs = probs.view(B, N).mean(dim=1) # 15帧取平均
            y_true.extend(labels.numpy())
            y_scores.extend(video_probs.cpu().numpy())
            
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    acc_05 = accuracy_score(y_true, (np.array(y_scores) >= 0.5).astype(int))
    best_acc = 0; opt_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        acc = accuracy_score(y_true, (np.array(y_scores) >= thresh).astype(int))
        if acc > best_acc: best_acc, opt_thresh = acc, thresh
            
    print("\n" + "="*50)
    print(f"Celeb-DF v2 (Frame Average): AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"Best ACC: {best_acc:.4f} (Thresh: {opt_thresh:.2f}) | Default ACC: {acc_05:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_verify()