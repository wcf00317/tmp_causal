import os
import torch
import fnmatch
import numpy as np
from torch.utils.data import Dataset


class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        纯净版 NYUv2 数据集读取类。
        直接读取 LibMTL 预处理好的 .npy 文件。
        假设数据已经是：
        1. 尺寸对齐 (H, W 一致)
        2. 数值正确 (Image [0,1], Label 整数)
        3. 维度顺序为 NumPy 默认的 HWC
        不做任何额外的 Resize、Crop、Flip 或 Normalize。
        """
        super().__init__()
        self.root = os.path.expanduser(root_dir)
        self.mode = mode

        # LibMTL 文件夹结构: root/train/image/*.npy
        sub_dir = 'train' if mode == 'train' else 'val'
        self.data_path = os.path.join(self.root, sub_dir)

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path not found: {self.data_path}")

        # 获取文件列表
        image_dir = os.path.join(self.data_path, 'image')
        # 使用 fnmatch 过滤 .npy 文件
        self.index_list = fnmatch.filter(os.listdir(image_dir), '*.npy')
        # 提取文件名中的数字并排序，确保数据对齐
        self.index_list = [int(x.replace('.npy', '')) for x in self.index_list]
        self.index_list.sort()

        self.num_samples = len(self.index_list)
        print(f"[{mode.upper()}] Found {self.num_samples} samples in {self.data_path}")

    def __getitem__(self, i):
        index = self.index_list[i]

        # ==================================================
        # 1. 读取 .npy 文件 (NumPy 默认 HWC 格式)
        # ==================================================
        img_path = os.path.join(self.data_path, 'image', f'{index}.npy')
        label_path = os.path.join(self.data_path, 'label', f'{index}.npy')
        depth_path = os.path.join(self.data_path, 'depth', f'{index}.npy')
        normal_path = os.path.join(self.data_path, 'normal', f'{index}.npy')

        img_np = np.load(img_path)  # [H, W, 3]
        label_np = np.load(label_path)  # [H, W]
        depth_np = np.load(depth_path)  # [H, W] 或 [H, W, 1]
        normal_np = np.load(normal_path)  # [H, W, 3]

        # ==================================================
        # 2. 维度变换 (HWC -> CHW) & 转 Tensor
        # ==================================================
        # Image: [H, W, 3] -> [3, H, W]
        image = torch.from_numpy(np.moveaxis(img_np, -1, 0)).float()

        # Label: [H, W] -> [H, W] (语义分割标签不需要 Channel 维)
        semantic = torch.from_numpy(label_np).long()

        # Depth: [H, W] 或 [H, W, 1] -> [1, H, W]
        if depth_np.ndim == 2:
            depth = torch.from_numpy(depth_np).float().unsqueeze(0)
        else:
            depth = torch.from_numpy(np.moveaxis(depth_np, -1, 0)).float()

        # Normal: [H, W, 3] -> [3, H, W]
        normal = torch.from_numpy(np.moveaxis(normal_np, -1, 0)).float()

        # ==================================================
        # 3. 构造返回字典 (纯净数据)
        # ==================================================
        return {
            'rgb': image,  # [3, H, W], float32, 未归一化
            'depth': depth,  # [1, H, W], float32
            'segmentation': semantic,  # [H, W], int64
            'normal': normal,  # [3, H, W], float32
            'scene_type': torch.tensor(0, dtype=torch.long),  # 占位符

            # Causal Model 需要的额外键：
            'appearance_target': image,  # 重构目标 (与输入一致)
            'depth_meters': depth  # 深度 (与 depth 一致)
        }

    def __len__(self):
        return self.num_samples