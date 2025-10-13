import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.transform import resize
from skimage.color import rgb2lab

# --- START: 标准 NYUv2 40类标签映射 ---
# 这是学术界公认的标准映射表之一。
# 关键：请确保您的评估代码也使用相同的类别->ID映射顺序。
_NYU40_CLASSES = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 2, 22, 1
]


def _remap_nyu40_labels(labels_raw):
    """
    一个高效的函数，用于将原始的、稀疏的NYUv2标签映射到标准的40个类别。
    不在映射表中的标签将被赋值为 255，以便在计算损失时被忽略。
    """
    max_label_id = labels_raw.max()
    if max_label_id <= 0:
        return labels_raw.astype(np.int64)

    mapping_array = np.full(shape=(max_label_id + 1,), fill_value=255, dtype=np.int64)

    for new_label_idx, old_label_id in enumerate(_NYU40_CLASSES):
        if old_label_id <= max_label_id:
            mapping_array[old_label_id] = new_label_idx

    remapped_labels = mapping_array[labels_raw]
    return remapped_labels


# --- END: 标签映射部分 ---


class NYUv2Dataset(Dataset):
    def __init__(self, mat_file_path, scene_types_list, img_size=(224, 224)):
        super().__init__()
        self.mat_file_path = mat_file_path
        self.img_size = img_size
        self.db = None

        self.scene_types_list = scene_types_list
        self.num_samples = len(self.scene_types_list)

        self.scene_classes = sorted(list(set(self.scene_types_list)))
        self.scene_class_to_idx = {cls: i for i, cls in enumerate(self.scene_classes)}

        print(f"Dataset initialized. Found {len(self.scene_classes)} unique scene classes.")
        print(f"Found {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.db is None:
            # 懒加载HDF5文件句柄，保证多进程安全
            self.db = h5py.File(self.mat_file_path, 'r', libver='latest', swmr=True)

        rgb_image = self.db['images'][idx].transpose(2, 1, 0)
        depth_map = self.db['depths'][idx].transpose(1, 0)
        seg_mask_raw = self.db['labels'][idx].transpose(1, 0)

        seg_mask_remapped = _remap_nyu40_labels(seg_mask_raw)

        scene_type_str = self.scene_types_list[idx]
        scene_label = self.scene_class_to_idx[scene_type_str]

        # --- 图像预处理 ---
        rgb_resized = resize(rgb_image, self.img_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)

        depth_resized = resize(depth_map, self.img_size, anti_aliasing=True).astype(np.float32)

        seg_mask_resized = resize(seg_mask_remapped, self.img_size, order=0, preserve_range=True,
                                  anti_aliasing=False).astype(np.int64)

        # --- 为重构任务计算 appearance target ---
        # 根据项目配置，此项为必需，因此予以保留
        # lab_image = rgb2lab(rgb_resized)
        # lab_ab_channels = lab_image[:, :, 1:]
        rgb_resized = resize(rgb_image, self.img_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        depth_resized = resize(depth_map, self.img_size, anti_aliasing=True).astype(np.float32)
        seg_mask_resized = resize(seg_mask_remapped, self.img_size, order=0, preserve_range=True,
                                  anti_aliasing=False).astype(np.int64)

        # --- 转换为Tensor ---
        to_tensor = transforms.ToTensor()
        normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_tensor_normalized = normalize_rgb(to_tensor(rgb_resized.copy()).float())  # 用于输入模型
        rgb_tensor_unnormalized = to_tensor(rgb_resized.copy()).float()

        depth_tensor = to_tensor(depth_resized.copy()).float()
        seg_mask_tensor = torch.from_numpy(seg_mask_resized.copy()).long()
        # 将 ab 通道的值从 [-128, 128] 范围归一化到 [-1, 1] 范围
        # normalized_lab_ab = lab_ab_channels / 128.0
        # appearance_target_tensor = to_tensor(normalized_lab_ab.copy()).float()

        return {
            'rgb': rgb_tensor_normalized,
            'depth': depth_tensor,
            'segmentation': seg_mask_tensor,
            'scene_type': torch.tensor(scene_label, dtype=torch.long),
            'appearance_target': rgb_tensor_unnormalized
            #'appearance_target': appearance_target_tensor
        }

    def close(self):
        if self.db is not None:
            self.db.close()