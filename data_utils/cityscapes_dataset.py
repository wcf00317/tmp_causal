import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

# --- Cityscapes 19类 官方映射表 ---
# 格式: id: (trainId, name)
# trainId=255 表示忽略该类 (void/ignore)
CITYSCAPES_MAPPING = {
    0: (255, 'unlabeled'), 1: (255, 'ego vehicle'), 2: (255, 'rectification border'),
    3: (255, 'out of roi'), 4: (255, 'static'), 5: (255, 'dynamic'),
    6: (255, 'ground'), 7: (0, 'road'), 8: (1, 'sidewalk'),
    9: (255, 'parking'), 10: (255, 'rail track'), 11: (2, 'building'),
    12: (3, 'wall'), 13: (4, 'fence'), 14: (255, 'guard rail'),
    15: (255, 'bridge'), 16: (255, 'tunnel'), 17: (5, 'pole'),
    18: (255, 'polegroup'), 19: (6, 'traffic light'), 20: (7, 'traffic sign'),
    21: (8, 'vegetation'), 22: (9, 'terrain'), 23: (10, 'sky'),
    24: (11, 'person'), 25: (12, 'rider'), 26: (13, 'car'),
    27: (14, 'truck'), 28: (15, 'bus'), 29: (255, 'caravan'),
    30: (255, 'trailer'), 31: (16, 'train'), 32: (17, 'motorcycle'),
    33: (18, 'bicycle'), -1: (255, 'license plate')
}


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(384, 384)):
        """
        Args:
            root_dir (str): 数据集根目录 (e.g., '/data/cityscapes')
            split (str): 'train', 'val', or 'test'
            img_size (tuple): 目标分辨率 (height, width)
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root_dir, 'gtFine', split)

        self.images = []
        self.targets = []

        # 1. 遍历目录收集文件路径 (Cityscapes 有子文件夹结构)
        # search pattern: root/leftImg8bit/train/*/*.png
        search_pattern = os.path.join(self.images_dir, "*", "*_leftImg8bit.png")
        for img_path in glob.glob(search_pattern):
            # 推导对应的 label 路径
            # img:   .../aachen/aachen_000000_000019_leftImg8bit.png
            # label: .../aachen/aachen_000000_000019_gtFine_labelIds.png
            file_name = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))

            # 注意：标签文件在 gtFine 下，且文件名后缀不同
            target_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            target_path = os.path.join(self.targets_dir, city, target_name)

            if os.path.exists(target_path):
                self.images.append(img_path)
                self.targets.append(target_path)

        # 排序以保证多卡训练时顺序一致
        self.images.sort()
        self.targets.sort()

        print(f"[{split.upper()}] Found {len(self.images)} images in {self.root_dir}")

        # 预计算映射表加速查找 (0-255 -> 0-18/255)
        self.mapping = np.zeros(256, dtype=np.int64) + 255
        for k, v in CITYSCAPES_MAPPING.items():
            if k >= 0:
                self.mapping[k] = v[0]

        # 提取城市名作为场景类别 (Cityscapes 没有显式的 scene label，这里用城市代替)
        self.cities = sorted(os.listdir(self.images_dir))
        self.city_to_idx = {c: i for i, c in enumerate(self.cities)}
        self.scene_classes = self.cities  # 供可视化脚本读取

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 读取图片 (RGB)
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        # 2. 读取分割标签 (Label)
        target_path = self.targets[idx]
        target = Image.open(target_path)

        # 3. 读取并计算深度 (Depth from Disparity)
        # 路径推导: leftImg8bit/train/city/xxx.png -> disparity/train/city/xxx_disparity.png
        # 需要处理路径替换，Cityscapes 的 disparity 文件名后缀是 '_disparity.png'
        disp_path = img_path.replace('leftImg8bit', 'disparity').replace('.png', '.png')
        disp_path = disp_path.replace('_leftImg8bit', '_disparity')

        # 准备深度图数据 (numpy)
        # 默认初始化为全零 (H, W)
        depth_np = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)
        # 2. ★新增：评估用的真实物理深度 (米)，初始化为 -1 (表示无效值)
        depth_meters_np = np.full((self.img_size[1], self.img_size[0]), -1.0, dtype=np.float32)

        if os.path.exists(disp_path):
            disp = Image.open(disp_path)
            # Resize 深度图
            disp = disp.resize(self.img_size, Image.NEAREST)
            disp_np = np.array(disp).astype(np.float32)

            mask = disp_np > 0

            # (A) 训练用的归一化视差 (保持不变)
            depth_np[mask] = disp_np[mask] / 65535.0

            # (B) ★新增：计算物理深度 (米)
            # Cityscapes 公式: depth = (baseline * focal) / (disp_val / 256.0)
            # 常数 0.209375 * 2262.52 * 256.0 ≈ 121246.75
            depth_meters_np[mask] = 121246.75 / disp_np[mask]

        # 4. Resize RGB 和 Label
        img = img.resize(self.img_size, Image.BILINEAR)
        target = target.resize(self.img_size, Image.NEAREST)

        # 5. 随机增强 (训练时) - 必须保证 img, target, depth 同步变换
        if self.split == 'train':
            if random.random() < 0.5:
                # 水平翻转
                img = transforms.functional.hflip(img)
                target = transforms.functional.hflip(target)
                # depth 是 numpy array，用 np.fliplr 翻转
                depth_np = np.fliplr(depth_np).copy()
                # ★新增：物理深度图也要同步翻转！
                depth_meters_np = np.fliplr(depth_meters_np).copy()

        # 6. 转 Tensor 和 归一化
        to_tensor = transforms.ToTensor()

        # RGB
        rgb_tensor_unnormalized = to_tensor(img).float()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_tensor_normalized = normalize(rgb_tensor_unnormalized)

        # Label (Mapping)
        target_np = np.array(target, dtype=np.int64)
        target_mapped = self.mapping[target_np]
        seg_tensor = torch.from_numpy(target_mapped).long()

        # Depth
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()  # [1, H, W]

        # 7. 返回
        return {
            'rgb': rgb_tensor_normalized,
            'depth': depth_tensor,
            'segmentation': seg_tensor,
            'scene_type': torch.tensor(0, dtype=torch.long),  # Cityscapes 只有一个场景类
            'appearance_target': rgb_tensor_unnormalized,
            'depth_meters': torch.from_numpy(depth_meters_np).unsqueeze(0).float()
        }

    def close(self):
        pass  # 不需要关闭文件句柄