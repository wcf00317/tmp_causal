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
        # 注意: disparity 文件名后缀是 '_disparity.png'
        disp_path = img_path.replace('leftImg8bit', 'disparity').replace('.png', '.png')
        # Cityscapes 文件名通常是: aachen_0000_leftImg8bit.png -> aachen_0000_disparity.png
        # 需要仔细对一下文件名替换规则，通常是把 _leftImg8bit 换成 _disparity
        disp_path = disp_path.replace('_leftImg8bit', '_disparity')

        # 相机参数 (用于计算深度)
        # camera/train/city/xxx_camera.json
        cam_path = img_path.replace('leftImg8bit', 'camera').replace('_leftImg8bit.png', '_camera.json')

        # 加载 Disparity (16bit png)
        # 官方公式: depth = (baseline * focal) / disparity
        # 这里的 disparity 存储为 uint16，实际值 = pixel_value / 256.0
        # 简单的平均基线 * 焦距常数 ≈ 0.209313 * 2262.52 ≈ 473.6
        # 为了简化读取 json 的 I/O 开销，我们可以用平均常数，或者由网络学习相对深度

        if os.path.exists(disp_path):
            disp = Image.open(disp_path)
            # Resize 深度图 (Nearest)
            disp = disp.resize(self.img_size, Image.NEAREST)
            disp_np = np.array(disp).astype(np.float32)
            # 过滤无效值 (0 表示无穷远)
            mask = disp_np > 0
            depth_np = np.zeros_like(disp_np)
            # Cityscapes 标准转换: depth = 0.22 * 2262 / (disp / 256)
            # 简化处理：我们直接把 disp 归一化作为"逆深度"使用，或者直接作为深度监督
            # 这里我们输出归一化的逆深度 (Inverse Depth)，更适合深度学习
            depth_np[mask] = disp_np[mask] / 65535.0  # 简单归一化到 0-1
        else:
            # 如果没找到，给个全零
            depth_np = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)

        # ... (后续的数据增强、ToTensor 逻辑与之前一致) ...
        # 注意对 depth_np 做同样的翻转增强

        # 4. 转 Tensor
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()  # [1, H, W]
        # ...

        return {
            'rgb': rgb_tensor_normalized,
            'depth': depth_tensor,
            'segmentation': seg_tensor,
            'scene_type': torch.tensor(0, dtype=torch.long),  # 占位
            'appearance_target': rgb_tensor_unnormalized
        }
    def close(self):
        pass  # 不需要关闭文件句柄