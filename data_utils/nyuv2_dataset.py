import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.transform import resize
from skimage.color import rgb2lab


class NYUv2Dataset(Dataset):
    def __init__(self, mat_file_path, img_size=(224, 224)):
        super().__init__()
        self.mat_file_path = mat_file_path
        self.img_size = img_size

        print(f"Loading NYUv2 dataset from: {self.mat_file_path}...")
        self.db = h5py.File(self.mat_file_path, 'r')

        self.images = self.db['images']
        self.depths = self.db['depths']
        self.labels = self.db['labels']
        self.scene_types_str = self.db['sceneTypes']  # 读取场景类别 (字符串)

        # 创建场景类别字符串到整数ID的映射
        self.scene_classes = sorted(list(set(s[0] for s in self.scene_types_str)))
        self.scene_class_to_idx = {cls: i for i, cls in enumerate(self.scene_classes)}

        print(f"Found {len(self.scene_classes)} unique scene classes.")
        print("Dataset loaded successfully.")
        print(f"Found {len(self)} samples.")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        rgb_image = self.images[idx].transpose(2, 1, 0)
        depth_map = self.depths[idx].transpose(1, 0)
        seg_mask = self.labels[idx].transpose(1, 0)

        # 获取场景类别并转换为整数ID
        scene_type_str_ref = self.scene_types_str[idx][0]
        scene_type_str = "".join(chr(c[0]) for c in self.db[scene_type_str_ref])
        scene_label = self.scene_class_to_idx[scene_type_str]

        rgb_resized = resize(rgb_image, self.img_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        depth_resized = resize(depth_map, self.img_size, anti_aliasing=True)
        seg_mask_resized = resize(seg_mask, self.img_size, order=0, preserve_range=True, anti_aliasing=False).astype(
            np.int64)

        lab_image = rgb2lab(rgb_resized)
        lab_ab_channels = lab_image[:, :, 1:]

        to_tensor = transforms.ToTensor()
        normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        rgb_tensor = normalize_rgb(to_tensor(rgb_resized.copy()).float())
        depth_tensor = to_tensor(depth_resized.copy()).float()
        seg_mask_tensor = torch.from_numpy(seg_mask_resized.copy()).long()
        appearance_target_tensor = to_tensor(lab_ab_channels.copy()).float()

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'segmentation': seg_mask_tensor,
            'scene_type': torch.tensor(scene_label, dtype=torch.long),  # 返回场景标签
            'appearance_target': appearance_target_tensor
        }

    def close(self):
        self.db.close()