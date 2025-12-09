import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class CityscapesCDataset(Dataset):
    """
    专门用于读取 Cityscapes-C (Corrupted) 数据集，
    并对齐 LibMTL 的预处理方式 (Resize to 128x256 + Norm).
    """

    def __init__(self, root_dir, gt_dir, corruption='fog', severity=1, img_size=(128, 256)):
        """
        Args:
            root_dir: Cityscapes-C 的根目录 (包含 fog, noise 等子文件夹)
            gt_dir: 原始 Clean Cityscapes 的 gtFine 目录 (用于读取 Label)
            corruption: 腐蚀类型 (e.g., 'gaussian_noise', 'fog')
            severity: 严重程度 (1-5)
            img_size: 必须与训练时的设置一致 (LibMTL 默认 [128, 256])
        """
        super().__init__()
        self.root_dir = root_dir
        self.gt_dir = gt_dir
        self.img_size = img_size

        # Cityscapes-C 目录结构: root/corruption/severity/city/image.png
        self.image_dir = os.path.join(root_dir, corruption, str(severity))

        # 搜索图片
        self.images = []
        self.targets = []

        # 遍历所有城市文件夹
        for city in os.listdir(self.image_dir):
            city_dir = os.path.join(self.image_dir, city)
            if not os.path.isdir(city_dir): continue

            for file_name in os.listdir(city_dir):
                if file_name.endswith('.png'):
                    img_path = os.path.join(city_dir, file_name)

                    # 推导 Label 路径 (需要对应到原始 Cityscapes 的 gtFine)
                    # Image: frankfurt_000000_000294_leftImg8bit.png
                    # Label: frankfurt_000000_000294_gtFine_labelIds.png (注意后缀可能不同)
                    label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    label_path = os.path.join(self.gt_dir, 'val', city, label_name)

                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.targets.append(label_path)

        print(f"[Cityscapes-C] {corruption} (Sev {severity}): Found {len(self.images)} images.")

        # 标准化 (ImageNet) - 必须与训练时一致
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 读取图片
        img = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])  # Label

        # 2. Resize (关键步骤：对齐 LibMTL 的 128x256)
        # Image 用 Bilinear
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        # Label 用 Nearest
        target = target.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        # 3. 转 Tensor & Normalize
        img_tensor = transforms.ToTensor()(img)  # [0, 1]
        img_tensor = self.normalize(img_tensor)

        target_np = np.array(target, dtype=np.int64)
        # LibMTL 的 Cityscapes 处理可能已经把 label 映射过了，
        # 如果这里读取的是原始 labelIds (0-33)，需要映射到 TrainID (0-18)。
        # 这里假设你已经有了 encode_target 函数，或者在外部做映射。
        # 为简化，这里暂时直接转 Tensor，你可能需要根据实际情况加 Label Mapping。
        target_tensor = torch.from_numpy(target_np).long()

        # 构造 dummy depth (Cityscapes-C 通常只评测分割)
        # 如果需要评测深度，需要加载原始的 disparity 并 resize
        depth_tensor = torch.zeros((1, self.img_size[0], self.img_size[1]))

        return {
            'rgb': img_tensor,
            'segmentation': target_tensor,
            'depth': depth_tensor,
            'normal': torch.zeros_like(img_tensor),  # 占位
            'scene_type': torch.tensor(0)
        }