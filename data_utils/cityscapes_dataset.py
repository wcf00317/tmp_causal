import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(384, 384)):
        super().__init__()
        self.root = os.path.expanduser(root_dir)
        self.split = split
        self.img_size = img_size

        # LibMTL 文件夹结构映射
        folder_split = 'train' if split == 'train' else 'val'
        self.data_path = os.path.join(self.root, folder_split)

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path not found: {self.data_path}")

        self.index_list = fnmatch.filter(os.listdir(os.path.join(self.data_path, 'image')), '*.npy')
        self.index_list = [int(x.replace('.npy', '')) for x in self.index_list]
        self.index_list.sort()

        self.num_samples = len(self.index_list)
        print(f"[{split.upper()}] Found {self.num_samples} samples in {self.data_path}")

        # 标准化参数 (ImageNet)
        self.normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, i):
        index = self.index_list[i]

        # 1. 读取 .npy (float64 [0,1])
        img_np = np.load(os.path.join(self.data_path, 'image', f'{index}.npy'))
        label_np = np.load(os.path.join(self.data_path, 'label', f'{index}.npy'))
        depth_np = np.load(os.path.join(self.data_path, 'depth', f'{index}.npy'))

        # LibMTL 存储格式通常是 [H, W, C]，需转为 [C, H, W]
        image = torch.from_numpy(np.moveaxis(img_np, -1, 0)).float()
        semantic = torch.from_numpy(label_np).float()
        depth = torch.from_numpy(np.moveaxis(depth_np, -1, 0)).float()

        # 2. Resize
        # 保持插值方法一致：Image -> Bilinear, Label/Depth -> Nearest
        image = F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=True).squeeze(0)
        semantic = F.interpolate(semantic.unsqueeze(0).unsqueeze(0), size=self.img_size, mode='nearest').squeeze(
            0).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=self.img_size, mode='nearest').squeeze(0)

        # 3. 预处理
        # 直接使用，不再除以 255
        rgb_unnormalized = image.clone()
        rgb_input = self.normalize_fn(image)

        return {
            'rgb': rgb_input,
            'depth': depth,
            'segmentation': semantic.long(),
            'scene_type': torch.tensor(0, dtype=torch.long),
            'appearance_target': rgb_unnormalized,
            'depth_meters': depth
        }

    def __len__(self):
        return self.num_samples

    def close(self): pass