import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse
import yaml

# --- 导入您的模型定义 ---
# 确保脚本在项目根目录下运行，或者将项目根目录添加到 PYTHONPATH
from models.causal_model import CausalMTLModel


# =========================================================================
# 1. 严格复刻的数据集定义 (完全遵守您的 LibMTL 代码格式)
# =========================================================================
class CityscapesC_Dataset(Dataset):
    def __init__(self, images_dir, gt_root):
        self.gt_root = gt_root
        self.img_paths = []

        # 严格字典序读取
        if os.path.exists(images_dir):
            subfolders = sorted([d for d in os.listdir(images_dir)
                                 if os.path.isdir(os.path.join(images_dir, d))])
            if len(subfolders) > 0:
                for city in subfolders:
                    city_path = os.path.join(images_dir, city)
                    files = sorted([f for f in os.listdir(city_path) if f.endswith('.png')])
                    for f in files:
                        self.img_paths.append(os.path.join(city_path, f))
            else:
                files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
                for f in files:
                    self.img_paths.append(os.path.join(images_dir, f))

        self.length = len(self.img_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # --- 输入处理：严格遵循 LibMTL 的 Resize(256, 128) -> Tensor ---
        # 注意：PIL resize 参数是 (W, H)，所以这里是 W=256, H=128
        img_pil = Image.open(img_path).convert('RGB')
        img_resized = img_pil.resize((256, 128), resample=Image.BILINEAR)

        # LibMTL 预处理：归一化到 [0, 1]
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # --- GT 读取：直接读取 NPY，不做任何缩放 ---
        # 假设 gt_root 下结构为 val/label/*.npy 和 val/depth/*.npy
        label_path = os.path.join(self.gt_root, 'val', 'label', f'{idx}.npy')
        depth_path = os.path.join(self.gt_root, 'val', 'depth', f'{idx}.npy')

        try:
            label = torch.from_numpy(np.load(label_path)).long()
            if os.path.exists(depth_path):
                depth = torch.from_numpy(np.load(depth_path)).float()
            else:
                depth = torch.zeros_like(label).float()
        except Exception:
            # 异常处理：返回全零
            label = torch.zeros((128, 256)).long()
            depth = torch.zeros((128, 256)).float()

        return img_tensor, {'segmentation': label, 'depth': depth}


# =========================================================================
# 2. 评估逻辑 (适配 CausalMTLModel)
# =========================================================================
def process_preds_aligned(outputs):
    """
    将模型输出对齐到 (128, 256) 并重命名键值以匹配评估逻辑。
    """
    target_size = (128, 256)  # (H, W)
    processed = {}

    # 映射键名: CausalMTL (pred_seg) -> LibMTL (segmentation)
    key_map = {
        'pred_seg': 'segmentation',
        'pred_depth': 'depth'
    }

    for k_model, k_eval in key_map.items():
        if k_model in outputs:
            pred = outputs[k_model]
            # 插值到 128x256
            if pred.shape[-2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=True)
            processed[k_eval] = pred

    return processed


def evaluate(model, loader, device, num_classes=19):
    model.eval()

    # 根据模型类别数初始化混淆矩阵 (原代码是硬编码7，这里改为动态读取)
    conf_mat = np.zeros((num_classes, num_classes))

    depth_abs_err = 0.0
    depth_rel_err = 0.0
    depth_count = 0

    with torch.no_grad():
        for img, gts in tqdm(loader, leave=False, desc="Eval"):
            img = img.to(device)

            # 1. 模型推理
            outputs = model(img, stage=2)  # 使用 stage=2 (全模型) 进行推理

            # 2. 对齐处理 (Output=128x256)
            preds = process_preds_aligned(outputs)

            # --- Segmentation Evaluation ---
            if 'segmentation' in preds:
                s_pred = preds['segmentation'].argmax(1).cpu().numpy()
                s_gt = gts['segmentation'].numpy()

                # 过滤非法标签 (0 ~ num_classes-1)
                mask = (s_gt >= 0) & (s_gt < num_classes)
                if mask.sum() > 0:
                    # bincount 计算混淆矩阵
                    conf_mat += np.bincount(
                        num_classes * s_gt[mask].astype(int) + s_pred[mask],
                        minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)

            # --- Depth Evaluation ---
            if 'depth' in preds:
                d_pred = preds['depth']
                # 如果是 [B, 1, H, W] -> Squeeze 为 [B, H, W]
                if d_pred.dim() == 4:
                    d_pred = d_pred.squeeze(1)

                d_gt = gts['depth'].to(device)

                # 强制 d_gt 与 d_pred 形状一致
                if d_gt.shape != d_pred.shape:
                    d_gt = d_gt.view_as(d_pred)

                # 只在 GT > 0 的地方评估
                valid = d_gt > 0
                if valid.sum() > 0:
                    pred_valid = d_pred[valid]
                    gt_valid = d_gt[valid]

                    # Abs Error
                    diff = torch.abs(pred_valid - gt_valid)
                    depth_abs_err += diff.sum().item()

                    # Rel Error
                    depth_rel_err += (diff / (gt_valid + 1e-8)).sum().item()

                    depth_count += valid.sum().item()

    # --- 计算最终指标 ---

    # 1. Seg mIoU
    intersection = np.diag(conf_mat)
    union = conf_mat.sum(1) + conf_mat.sum(0) - intersection
    miou = np.nanmean(intersection / (union + 1e-10))

    # 2. Seg Pix Acc
    pix_acc = intersection.sum() / (conf_mat.sum() + 1e-10)

    # 3. Depth Metrics
    abs_err = depth_abs_err / (depth_count + 1e-10)
    rel_err = depth_rel_err / (depth_count + 1e-10)

    return miou, pix_acc, abs_err, rel_err


# =========================================================================
# 3. 主程序
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="模型配置文件路径 (.yaml)")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型权重路径 (.pth.tar)")
    parser.add_argument('--cc_dir', type=str, required=True, help="Cityscapes-C 数据集根目录")
    parser.add_argument('--gt_dir', type=str, required=True, help="预处理后的 GT 根目录 (包含 val/label/*.npy)")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--bs', '--batch_size', type=int, default=16, dest='bs')
    parser.add_argument('--output_txt', type=str, default='eval_cc_report.txt')
    args = parser.parse_args()

    # 设备设置
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # 1. 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 初始化模型
    print("⚙️ Building CausalMTLModel...")
    # 注意：这里可能会报 'data_config' key error，如果您的 config 结构不同。
    # 假设 config 包含 'model' 和 'data' 字段。
    model = CausalMTLModel(config['model'], config['data']).to(device)

    # 3. 加载权重
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded (Strict).")
    except Exception as e:
        print(f"⚠️ Strict load failed, trying non-strict... {e}")
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (Non-Strict).")

    # 获取类别数 (Cityscapes 通常是 19 类，但您的预处理数据如果是 7 类，这里需要匹配)
    # 优先读取 config 中的 num_seg_classes，默认为 19
    num_classes = config['model'].get('num_seg_classes', 19)
    print(f"ℹ️ Evaluating with {num_classes} segmentation classes.")

    # 4. 准备记录
    if not os.path.exists(args.cc_dir):
        print(f"❌ Error: Cannot find Cityscapes-C dir: {args.cc_dir}")
        return
    if not os.path.exists(os.path.join(args.gt_dir, 'val', 'label')):
        print(f"❌ Error: Cannot find GT labels at: {args.gt_dir}/val/label/")
        return

    corruptions = sorted([d for d in os.listdir(args.cc_dir) if os.path.isdir(os.path.join(args.cc_dir, d))])

    f_log = open(args.output_txt, 'w')

    def log(msg):
        print(msg)
        f_log.write(msg + '\n')
        f_log.flush()

    log(f"🚀 Start Eval on Cityscapes-C")
    log(f"Config: {args.config}")
    log(f"Checkpoint: {args.checkpoint}")
    log(f"-" * 60)

    # 5. 循环评测
    for corruption in corruptions:
        log(f"\n[Corruption: {corruption}]")
        metrics_sum = {'mIoU': 0, 'PixAcc': 0, 'AbsErr': 0, 'RelErr': 0}

        for severity in range(1, 6):
            # 构造数据集
            dataset = CityscapesC_Dataset(
                images_dir=os.path.join(args.cc_dir, corruption, str(severity)),
                gt_root=args.gt_dir
            )

            if len(dataset) == 0:
                print(f"  Warning: No images found for {corruption} level {severity}")
                continue

            loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

            # 执行评估
            miou, pix_acc, abs_err, rel_err = evaluate(model, loader, device, num_classes=num_classes)

            log(f"  Level {severity}: mIoU={miou:.4f} | PixAcc={pix_acc:.4f} | AbsErr={abs_err:.4f} | RelErr={rel_err:.4f}")

            metrics_sum['mIoU'] += miou
            metrics_sum['PixAcc'] += pix_acc
            metrics_sum['AbsErr'] += abs_err
            metrics_sum['RelErr'] += rel_err

        log(f"  >> Avg ({corruption}): mIoU={metrics_sum['mIoU'] / 5:.4f} | AbsErr={metrics_sum['AbsErr'] / 5:.4f}")

    f_log.close()
    print(f"\n✨ Evaluation Finished. Results saved to {args.output_txt}")


if __name__ == "__main__":
    main()