import torch
import os
import json
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 1. 强制离线模式 & 显卡设置
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入模型组件
from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 修正后的 Dataset (确保加载所有伪造类型) ---
class ImageFolderJSONDataset(Dataset):
    def __init__(self, data_root, transform_cfg, split_fn, target_method=None):
        self.data_root = data_root
        with open(split_fn, 'r') as f:
            self.folder_list = json.load(f)
        
        self.transform = get_image_transformation_from_cfg(transform_cfg)
        self.samples = []
        
        # 如果指定目标方法则只验证该方法，否则验证列表中的方法
        methods = [target_method] if target_method else ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
        
        for pair in self.folder_list:
            id1, id2 = pair[0], pair[1]
            # 1. 真脸
            real_path = os.path.join(self.data_root, "original_sequences/youtube/c23/videos", id1)
            if os.path.exists(real_path):
                imgs = sorted(glob.glob(os.path.join(real_path, "*.png")))
                for img_path in imgs[:15]: # 采样加速验证
                    self.samples.append((img_path, 0))
            
            # 2. 假脸
            fake_folder_name = f"{id1}_{id2}"
            for m in methods:
                fake_path = os.path.join(self.data_root, "manipulated_sequences", m, "c23/videos", fake_folder_name)
                if os.path.exists(fake_path):
                    imgs = sorted(glob.glob(os.path.join(fake_path, "*.png")))
                    for img_path in imgs[:15]:
                        self.samples.append((img_path, 1))
                    # 注意：这里不加 break，或者因为外层控制了 methods 列表所以无所谓

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

def run_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动定位 CLIP 路径 (沿用你之前的成功路径)
    base_snapshot_path = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    LOCAL_CLIP_PATH = base_snapshot_path if os.path.exists(base_snapshot_path) else ""
    
    if not LOCAL_CLIP_PATH:
        print("警告：未找到 CLIP 路径，请手动修改代码中的 base_snapshot_path")
        return

    print(f"--- 初始化模型 ---")
    model = M2F2Det(
        clip_text_encoder_name=LOCAL_CLIP_PATH,
        clip_vision_encoder_name=LOCAL_CLIP_PATH,
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    ).to(device)

    # --- 修改这里：填入你刚才 ls 找到的那个最好的模型路径 ---
    ckpt_path = 'checkpoints/stage_1/best_model.pth' # 或者 current_model_xxx.pth
    
    if not os.path.exists(ckpt_path):
        # 尝试找最新的 current_model
        list_of_files = glob.glob('checkpoints/stage_1/current_model_2.pth') 
        if list_of_files:
            ckpt_path = max(list_of_files, key=os.path.getctime)
        else:
            print("错误：找不到模型权重文件！")
            return

    print(f"正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 权重加载逻辑
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else (checkpoint['model'] if 'model' in checkpoint else checkpoint)
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    
    # Vision Tower 再次确认加载
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
    
    print("\n" + "="*60)
    print(f"{'Method':<20} | {'AUC':<10} | {'AP':<10}")
    print("-" * 60)
    
    for method in methods:
        dataset = ImageFolderJSONDataset(data_root, get_val_cfg(), val_split, target_method=method)
        if len(dataset) == 0: continue
            
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Testing {method}", leave=False):
                out = model(imgs.to(device), return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                y_true.extend(labels.numpy())
                y_scores.extend(probs.cpu().numpy())
        
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            print(f"{method:<20} | {auc:.4f}     | {ap:.4f}")
        else:
            print(f"{method:<20} | 无法计算 (数据单一)")

if __name__ == "__main__":
    run_verify()