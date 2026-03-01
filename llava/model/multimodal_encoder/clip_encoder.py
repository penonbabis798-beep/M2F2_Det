# import torch
# import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


# class CLIPVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()

#         self.is_loaded = False

#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

#         # === 新增：无论是否 delay_load，都先加载处理器 ===
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         # =================================================
        
#         if not delay_load:
#             self.load_model()
#         elif getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.load_model()
#         else:
#             self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
#             return

#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
#         self.vision_tower.requires_grad_(False)

#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(f'Unexpected select feature: {self.select_feature}')
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else: 
#             image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches_per_side(self):
#         return self.config.image_size // self.config.patch_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2

# 修改后：
import torch
import torch.nn as nn
from transformers import  LIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        # 即使 delay_load，我们也先定义 vision_tower 为 None，防止 AttributeError
        self.vision_tower = None

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # 加载真实的 CLIP 视觉模型
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        # 【修复点】如果 forward 被调用时模型还没加载，强制加载它
        if not self.is_loaded or self.vision_tower is None:
            self.load_model()

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # 【修复点】增加安全检查
        if self.vision_tower is None: return torch.float16
        return self.vision_tower.dtype

    @property
    def device(self):
        # 【修复点】增加安全检查
        if self.vision_tower is None: return torch.device('cuda')
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded and self.vision_tower is not None:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2