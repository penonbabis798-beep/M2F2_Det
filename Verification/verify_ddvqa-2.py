import torch
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from sequence.models.M2F2_Det.models.model import M2F2Det
from dataset.process import get_image_transformation_from_cfg

# --- 纯单帧 DDVQA Dataset ---
class DDVQASingleFrameDataset(Dataset):
    def __init__(self, images_root, json_path, transform_cfg):
        self.images_root = images_root
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
            return torch.zeros(3, 224, 224), label

def get_val_cfg():
    return {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}, 'noise': {'prob': 0.0, 'var': [0.01]}}, 'flip': False}
  
def run_cross_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DDVQA_IMAGES_ROOT = "./utils/DDVQA_images/c40" 
    DDVQA_JSON_PATH = "./utils/DDVQA_split/c40/train_DDVQA_format.json"
    LOCAL_CLIP_PATH = "/data/tangchengwen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    print("--- 初始化模型 (静态图像评估) ---")
    model = M2F2Det(clip_text_encoder_name=LOCAL_CLIP_PATH, clip_vision_encoder_name=LOCAL_CLIP_PATH, deepfake_encoder_name='efficientnet_b4', hidden_size=1792).to(device)

    import glob
    ckpt_dir = 'checkpoints/stage_1_freq_spatial_dual'
    ckpt_path = ''
    if os.path.exists(ckpt_dir):
        list_of_files = glob.glob(f'{ckpt_dir}/current_model_*.pth')
        if list_of_files: ckpt_path = max(list_of_files, key=os.path.getctime)
            
    if not ckpt_path or not os.path.exists(ckpt_path):
        list_of_files = glob.glob('checkpoints/stage_1_spatial/current_model_*.pth') 
        if list_of_files: ckpt_path = max(list_of_files, key=os.path.getctime)

    print(f"加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model']
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = DDVQASingleFrameDataset(DDVQA_IMAGES_ROOT, DDVQA_JSON_PATH, get_val_cfg())
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Testing DDVQA"):
                out = model(imgs.to(device), return_dict=True)
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                y_true.extend(labels.numpy())
                y_scores.extend(probs.cpu().numpy())
        
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        acc_05 = accuracy_score(y_true, (np.array(y_scores) >= 0.5).astype(int))
        best_acc = 0; opt_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            acc = accuracy_score(y_true, (np.array(y_scores) >= thresh).astype(int))
            if acc > best_acc: best_acc, opt_thresh = acc, thresh
        
        print("\n" + "="*50)
        print(f"DDVQA (Static Frame): AUC: {auc:.4f} | AP: {ap:.4f}")
        print(f"Best ACC: {best_acc:.4f} (Thresh: {opt_thresh:.2f}) | Default ACC: {acc_05:.4f}")
        print("="*50)

if __name__ == "__main__":
    run_cross_test()