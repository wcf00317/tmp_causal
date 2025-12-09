import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms


class RandomScaleCrop(object):
    """
    完全对齐 LibMTL 的随机缩放裁剪逻辑。
    Ref: LibMTL/examples/nyu/create_dataset.py
    """

    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        # img: [C, H, W]
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        # 1. Image: Bilinear
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear',
                             align_corners=True).squeeze(0)

        # 2. Label: Nearest
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(
            0).squeeze(0)

        # 3. Depth: Nearest + 除以 scale (LibMTL 特性)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        depth_ = depth_ / sc

        # 4. Normal: Bilinear
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear',
                                align_corners=True).squeeze(0)

        return img_, label_, depth_, normal_


class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=(384, 384)):
        super().__init__()
        self.root = os.path.expanduser(root_dir)
        self.mode = mode
        self.img_size = img_size
        self.augmentation = (mode == 'train')

        # 路径适配 LibMTL 结构 (root/train/image/*.npy)
        sub_dir = 'train' if mode == 'train' else 'val'
        self.data_path = os.path.join(self.root, sub_dir)

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path not found: {self.data_path}")

        # 获取文件列表
        self.index_list = fnmatch.filter(os.listdir(os.path.join(self.data_path, 'image')), '*.npy')
        self.index_list = [int(x.replace('.npy', '')) for x in self.index_list]
        self.index_list.sort()

        self.num_samples = len(self.index_list)
        print(f"[{mode.upper()}] Found {self.num_samples} samples in {self.data_path}")

        # 标准化参数 (ImageNet) - 依然需要，因为这是 Backbone (ResNet/ViT) 的输入假设
        self.normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, i):
        index = self.index_list[i]

        # 1. 读取 (LibMTL 格式: [H, W, C] -> 转为 [C, H, W])
        # 数据已经是 float64 且 [0, 1]
        img_np = np.load(os.path.join(self.data_path, 'image', f'{index}.npy'))
        label_np = np.load(os.path.join(self.data_path, 'label', f'{index}.npy'))
        depth_np = np.load(os.path.join(self.data_path, 'depth', f'{index}.npy'))
        normal_np = np.load(os.path.join(self.data_path, 'normal', f'{index}.npy'))

        image = torch.from_numpy(np.moveaxis(img_np, -1, 0)).float()
        semantic = torch.from_numpy(label_np).float()
        depth = torch.from_numpy(np.moveaxis(depth_np, -1, 0)).float()
        normal = torch.from_numpy(np.moveaxis(normal_np, -1, 0)).float()

        # 2. 增强 (严格对齐 LibMTL)
        if self.augmentation:
            # Scale & Crop
            image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)

            # Flip (概率 0.5)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                # LibMTL 特性：法线翻转后，x轴分量取反
                normal[0, :, :] = - normal[0, :, :]

        # 3. 统一 Resize 到训练分辨率
        # 这是为了 batch 训练必须的步骤，保持插值模式一致
        image = F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=True).squeeze(0)
        semantic = F.interpolate(semantic.unsqueeze(0).unsqueeze(0), size=self.img_size, mode='nearest').squeeze(
            0).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=self.img_size, mode='nearest').squeeze(0)
        normal = F.interpolate(normal.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=True).squeeze(0)

        # 4. 预处理
        # 既然已经是 [0, 1]，直接 clone 一份做 target，一份做 normalize 输入
        rgb_unnormalized = image.clone()
        rgb_input = self.normalize_fn(image)

        return {
            'rgb': rgb_input,
            'depth': depth,
            'segmentation': semantic.long(),
            'normal': normal,
            'scene_type': torch.tensor(0, dtype=torch.long),  # 占位
            'appearance_target': rgb_unnormalized
        }

    def __len__(self):
        return self.num_samples

    def close(self):
        pass