import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 项目模块导入 ---
# 确保您已经按之前的建议修正了 data_utils/gta5_dataset.py 中的 resize 逻辑
from data_utils.gta5_dataset import GTA5Dataset
from models.causal_model import CausalMTLModel
from utils.general_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Test CausalMTL Model on GTA5 Validation Set")

    # 必须参数
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the training config (e.g., configs/resnet/gta5_to_cityscapes.yaml)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint (e.g., runs/.../model_best.pth.tar)")

    # 可选参数 (有默认值)
    parser.add_argument('--gta5_val_dir', type=str, default="/data/chengfengwu/alrl/mtl_dataset/gta5/val",
                        help="Path to GTA5 validation set (root dir containing images/ and labels/)")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_id', type=int, default=0)

    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device, num_classes):
    model.eval()

    # 初始化混淆矩阵
    conf_mat = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating GTA5 Val"):
            # GTA5Dataset 返回的字典 keys: 'rgb', 'segmentation', ...
            img = batch['rgb'].to(device)
            target = batch['segmentation'].numpy()  # Keep on CPU for numpy calculation

            # 推理 (使用 stage=2 保证全参数工作)
            outputs = model(img, stage=2)

            # 获取分割预测
            pred_logits = outputs['pred_seg']

            # 对齐尺寸 (如果模型输出与输入不一致，通常 ResNet 解码器会对齐，但也防万一)
            if pred_logits.shape[-2:] != img.shape[-2:]:
                pred_logits = F.interpolate(pred_logits, size=img.shape[-2:],
                                            mode='bilinear', align_corners=False)

            # Argmax 得到类别
            pred_label = pred_logits.argmax(dim=1).cpu().numpy()

            # --- 更新混淆矩阵 ---
            # 过滤掉 ignore_index (通常是 255)
            mask = (target >= 0) & (target < num_classes)

            # 展平并计算
            conf_mat += np.bincount(
                num_classes * target[mask].astype(int) + pred_label[mask],
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # --- 计算指标 ---
    # Intersection = 对角线元素
    intersection = np.diag(conf_mat)
    # Union = 行求和 + 列求和 - 对角线
    union = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - intersection

    # mIoU (忽略从未出现的类别的 NaN)
    iou = intersection / (union + 1e-10)
    miou = np.nanmean(iou)

    # Pixel Accuracy
    pixel_acc = intersection.sum() / (conf_mat.sum() + 1e-10)

    return miou, pixel_acc, iou


def main():
    args = parse_args()

    # 1. 设备设置
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 2. 加载配置
    print(f"📂 Loading config: {args.config}")
    config = load_config(args.config)

    # 3. 构建模型 (CausalMTLModel 会根据 config['model']['encoder_name'] 自动加载 ResNet50)
    print("⚙️ Building Model...")
    model = CausalMTLModel(config['model'], config['data']).to(device)

    # 4. 加载权重
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded (Strict).")
    except Exception as e:
        print(f"⚠️ Strict load failed, trying non-strict... {e}")
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (Non-Strict).")

    # 5. 准备数据
    # 注意：这里我们直接用 GTA5Dataset 读取验证集目录
    # 必须确保 config['data']['img_size'] 是 [128, 256] (H, W)
    # 并且 data_utils/gta5_dataset.py 已经修复了 resize 顺序
    img_size = config['data']['img_size']
    print(f"📏 Using Image Size from config: {img_size} (H, W)")

    val_dataset = GTA5Dataset(root_dir=args.gta5_val_dir, img_size=img_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"📚 GTA5 Val Dataset: {len(val_dataset)} images")

    # 6. 获取类别数 (从 config 读取，通常是 7 或 19)
    num_classes = config['model'].get('num_seg_classes', 7)  # 默认为19，防止报错
    if 'gta5_to_cityscapes' in config['data']['type']:
        # 如果是 G2C 任务且 config 里明确写了 7
        print(f"ℹ️ Task is GTA5 -> Cityscapes. Evaluating on {num_classes} classes.")

    # 7. 开始评估
    print("🎬 Starting Evaluation...")
    miou, pix_acc, class_iou = evaluate(model, val_loader, device, num_classes)

    # 8. 打印结果
    print("\n" + "=" * 40)
    print(f"🏆 GTA5 Validation Results")
    print(f"   (Source Domain Performance)")
    print("-" * 40)
    print(f"   mIoU      : {miou * 100:.2f}%")
    print(f"   Pixel Acc : {pix_acc * 100:.2f}%")
    print("-" * 40)
    print("Per-Class IoU:")
    for i, iou_score in enumerate(class_iou):
        print(f"   Class {i:<2}: {iou_score * 100:.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()