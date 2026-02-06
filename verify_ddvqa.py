import torch
import os
import json
import glob
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

# --- 专用于 DDVQA 的 Dataset 类 ---
class DDVQADataset(Dataset):
    def __init__(self, images_root, json_path, transform_cfg):
        self.images_root = images_root
        
        print(f"正在加载 DDVQA 标注文件: {json_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到标注文件: {json_path}")
            
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.samples = []
        
        print("正在解析数据标签并匹配图片路径...")
        missing_count = 0
        
        # 预先检查子目录是否存在，减少后续 IO
        has_train_dir = os.path.isdir(os.path.join(images_root, 'train'))
        has_test_dir = os.path.isdir(os.path.join(images_root, 'test'))
        
        for item in self.data_list:
            # 1. 获取图片文件名
            if 'image' not in item: continue
            img_name = item['image']
            
            # 2. 解析标签
            label = -1
            conversations = item.get('conversations', [])
            if not conversations: continue
            last_response = conversations[-1]['value'].lower().strip()
            
            if 'fake' in last_response:
                label = 1
            elif 'real' in last_response:
                label = 0
            else:
                continue 
            
            # 3. 智能路径搜索 (核心修改)
            # 优先级 1: 直接在根目录下找
            full_path = os.path.join(self.images_root, img_name)
            
            if not os.path.exists(full_path) and has_train_dir:
                # 优先级 2: 在 train 子目录下找
                full_path = os.path.join(self.images_root, 'train', img_name)
                
            if not os.path.exists(full_path) and has_test_dir:
                # 优先级 3: 在 test 子目录下找
                full_path = os.path.join(self.images_root, 'test', img_name)
                
            if os.path.exists(full_path):
                self.samples.append((full_path, label))
            else:
                missing_count += 1
                # 仅在缺失大量文件时打印部分日志，防止刷屏
                if missing_count < 5:
                    print(f"缺失图片: {img_name} (尝试路径: {full_path})")
                
        print(f"加载完成!")
        print(f" - 有效样本数: {len(self.samples)}")
        print(f" - 缺失图片数: {missing_count}")
        
        if len(self.samples) == 0:
            print(f"!!! 严重错误: 依然未找到任何图片，请确认 {self.images_root} 下是否有 train/test 文件夹 !!!")

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
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 224, 224), label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}

def run_cross_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 配置路径 (请根据你的实际情况修改) ---
    
    # 1. DDVQA 图片根目录 (解压后的文件夹)
    # 假设你在项目根目录下解压了 utils/DDVQA_images/c40.zip
    DDVQA_IMAGES_ROOT = "./utils/DDVQA_images/c40" 
    
    # 2. DDVQA 标注文件
    DDVQA_JSON_PATH = "./utils/DDVQA_split/c40/train_DDVQA_format.json"
    
    # 3. CLIP 本地路径 (沿用之前的)
    CLIP_PATH = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # 4. 训练好的模型权重
    CKPT_PATH = "checkpoints/stage_1/current_model_2.pth" # 使用你刚才测试效果最好的那个权重

    # --- 初始化 ---
    if not os.path.exists(DDVQA_IMAGES_ROOT):
        print(f"错误: 图片目录不存在: {DDVQA_IMAGES_ROOT}")
        print("请确认你已经解压了 DDVQA 数据集，并修改代码中的路径。")
        return

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
    
    # Vision Tower 加载
    vt_path = './utils/weights/vision_tower.pth'
    if os.path.exists(vt_path):
        vt_weights = torch.load(vt_path, map_location='cpu')
        vt_dict = {k.replace("vision_tower.", ""): v for k, v in vt_weights.items()}
        target_sd = model.clip_vision_encoder.model.state_dict()
        filtered = {k: v for k, v in vt_dict.items() if k in target_sd or f"vision_model.{k}" in target_sd}
        model.clip_vision_encoder.model.load_state_dict(filtered, strict=False)

    model.eval()

    # --- 开始测试 ---
    print(f"\n{'='*20} 开始 DDVQA 跨库测试 {'='*20}")
    
    dataset = DDVQADataset(DDVQA_IMAGES_ROOT, DDVQA_JSON_PATH, get_val_cfg())
    
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Testing DDVQA"):
                out = model(imgs.to(device), return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                y_true.extend(labels.numpy())
                y_scores.extend(probs.cpu().numpy())
        
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        print("\n" + "="*50)
        print(f"Dataset: DDVQA (c40)")
        print(f"Total Samples: {len(y_true)}")
        print(f"AUC: {auc:.4f}")
        print(f"AP:  {ap:.4f}")
        print("="*50)
    else:
        print("无法进行测试：数据加载为空。")

if __name__ == "__main__":
    run_cross_test()