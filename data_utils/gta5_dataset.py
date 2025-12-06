import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

# GTA5 (34 classes) -> Cityscapes (19 classes) 映射表
# 格式: GTA5_ID: Cityscapes_TrainID
# 255 表示忽略
GTA5_TO_CITYSCAPES_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,  # Road
    8: 1,  # Sidewalk
    9: 255, 10: 255,
    11: 2,  # Building
    12: 3,  # Wall
    13: 4,  # Fence
    14: 255, 15: 255, 16: 255,
    17: 5,  # Pole
    18: 255,
    19: 6,  # Traffic light
    20: 7,  # Traffic sign
    21: 8,  # Vegetation
    22: 9,  # Terrain
    23: 10,  # Sky
    24: 11,  # Person
    25: 12,  # Rider
    26: 13,  # Car
    27: 14,  # Truck
    28: 15,  # Bus
    29: 255, 30: 255,
    31: 16,  # Train
    32: 17,  # Motorcycle
    33: 18,  # Bicycle
    -1: 255
}


class GTA5Dataset(Dataset):
    def __init__(self, root_dir, img_size=(384, 384)):
        """
        Args:
            root_dir: GTA5 数据集根目录 (e.g., '../mtl_dataset/gta5')
            img_size: Resize 尺寸
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size

        # 兼容 folder 命名大小写 (Images/images, Labels/labels)
        self.images_dir = os.path.join(root_dir, 'images')
        if not os.path.exists(self.images_dir):
            self.images_dir = os.path.join(root_dir, 'Images')

        self.labels_dir = os.path.join(root_dir, 'labels')
        if not os.path.exists(self.labels_dir):
            self.labels_dir = os.path.join(root_dir, 'Labels')

        self.images = []
        self.targets = []

        # 查找所有图片
        # GTA5 文件名通常是 00001.png
        search_pattern = os.path.join(self.images_dir, "*.png")
        files = glob.glob(search_pattern)

        for img_path in files:
            file_name = os.path.basename(img_path)
            # 标签文件名通常和图片一致
            label_path = os.path.join(self.labels_dir, file_name)

            if os.path.exists(label_path):
                self.images.append(img_path)
                self.targets.append(label_path)

        self.images.sort()
        self.targets.sort()

        # 预计算映射数组
        self.mapping = np.zeros(256, dtype=np.int64) + 255
        for k, v in GTA5_TO_CITYSCAPES_MAPPING.items():
            if k >= 0:
                self.mapping[k] = v

        print(f"[GTA5] Found {len(self.images)} samples in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.targets[idx]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # Resize (Nearest for label)
        img = img.resize(self.img_size, Image.BILINEAR)
        label = label.resize(self.img_size, Image.NEAREST)

        # 随机水平翻转 (Sim-to-Real 训练时增强很重要)
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)

        # 转 Tensor
        to_tensor = transforms.ToTensor()
        rgb_tensor_unnormalized = to_tensor(img).float()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_tensor_normalized = normalize(rgb_tensor_unnormalized)

        # Label Mapping
        label_np = np.array(label, dtype=np.int64)
        # GTA5 有时有些标签会越界，clip一下或者mask
        label_np[label_np > 255] = 255
        label_mapped = self.mapping[label_np]
        seg_tensor = torch.from_numpy(label_mapped).long()

        # GTA5 只有图片和分割，没有深度 (除非特定版本)
        # 我们返回全零深度，并在 config 里把 lambda_depth 设为 0
        depth_tensor = torch.zeros((1, self.img_size[1], self.img_size[0]), dtype=torch.float32)

        return {
            'rgb': rgb_tensor_normalized,
            'depth': depth_tensor,  # 占位
            'segmentation': seg_tensor,
            'scene_type': torch.tensor(0),  # 占位
            'appearance_target': rgb_tensor_unnormalized
        }