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

# --- 兼容视频模型的 DDVQA Dataset (单图复制 8 份) ---
class DDVQAVideoMockDataset(Dataset):
    def __init__(self, images_root, json_path, transform_cfg, num_frames=8):
        self.images_root = images_root
        self.num_frames = num_frames
        
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.samples = []
        
        has_train_dir = os.path.isdir(os.path.join(images_root, 'train'))
        has_test_dir = os.path.isdir(os.path.join(images_root, 'test'))
        
        for item in self.data_list:
            if 'image' not in item: continue
            img_name = item['image']
            
            label = -1
            conversations = item.get('conversations', [])
            if not conversations: continue
            last_response = conversations[-1]['value'].lower().strip()
            
            if 'fake' in last_response: label = 1
            elif 'real' in last_response: label = 0
            else: continue 
            
            full_path = os.path.join(self.images_root, img_name)
            if not os.path.exists(full_path) and has_train_dir:
                full_path = os.path.join(self.images_root, 'train', img_name)
            if not os.path.exists(full_path) and has_test_dir:
                full_path = os.path.join(self.images_root, 'test', img_name)
                
            if os.path.exists(full_path):
                self.samples.append((full_path, label))

        print(f"DDVQA 数据加载完成! 有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
                
            # 【核心修改】：复制单张图片 num_frames 次，欺骗模型这是个静止的视频
            video_tensor = img.unsqueeze(0).repeat(self.num_frames, 1, 1, 1) # [8, 3, 224, 224]
            return video_tensor, label
        except Exception as e:
            return torch.zeros(self.num_frames, 3, 224, 224), label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_cross_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DDVQA_IMAGES_ROOT = "./utils/DDVQA_images/c40" 
    DDVQA_JSON_PATH = "./utils/DDVQA_split/c40/train_DDVQA_format.json"
    CLIP_PATH = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # 指向最佳的视频级权重
    CKPT_PATH = "checkpoints/stage_1_video_level/current_model_5.pth" 

    print("--- 初始化模型 ---")
    model = M2F2Det(
        clip_text_encoder_name=CLIP_PATH,
        clip_vision_encoder_name=CLIP_PATH,
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    ).to(device)

    print(f"加载权重: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model']
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print(f"\n{'='*20} 开始 DDVQA 兼容性测试 {'='*20}")
    dataset = DDVQAVideoMockDataset(DDVQA_IMAGES_ROOT, DDVQA_JSON_PATH, get_val_cfg(), num_frames=8)
    
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Testing DDVQA"):
                out = model(imgs.to(device), return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                y_true.extend(labels.numpy())
                y_scores.extend(probs.cpu().numpy())
        
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        y_pred = (np.array(y_scores) >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print(f"Dataset: DDVQA (Static Frame Test)")
        print(f"AUC: {auc:.4f}  |  AP: {ap:.4f}  |  ACC: {acc:.4f}")
        print("="*50)

if __name__ == "__main__":
    run_cross_test()