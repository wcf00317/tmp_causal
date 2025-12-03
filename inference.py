import os
import argparse
import yaml
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# --- 项目模块导入 ---
from models.causal_model import CausalMTLModel
from data_utils.nyuv2_dataset import NYUv2Dataset
# --- 修改：导入所有需要的可视化函数，保持一致性 ---
from engine.visualizer import (
    _visualize_microscope,
    _visualize_mixer,
    _visualize_depth_task,  # 新增导入
    denormalize_image
)
from utils.general_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Causal MTL Inference & Visualization Standalone Script")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the config file used for training")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument('--dataset_path', type=str, default=None, help="Override dataset path")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use")
    parser.add_argument('--batch_size', type=int, default=1, help="Inference batch size")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 加载配置
    print(f"📂 Loading config from: {args.config}")
    config = load_config(args.config)
    if args.dataset_path:
        config['data']['dataset_path'] = args.dataset_path

    # 2. 复现数据划分
    seed = config['training']['seed']
    set_seed(seed)
    print(f"🌱 Re-seeding with {seed} ...")

    print("📚 Initializing Dataset...")
    with h5py.File(config['data']['dataset_path'], 'r') as db:
        scene_type_refs = db['sceneTypes']
        scene_types_list = []
        for i in range(scene_type_refs.shape[1]):
            ref = scene_type_refs[0, i]
            scene_str = "".join(chr(c[0]) for c in db[ref])
            scene_types_list.append(scene_str)

    full_dataset = NYUv2Dataset(
        mat_file_path=config['data']['dataset_path'],
        img_size=tuple(config['data']['img_size']),
        scene_types_list=scene_types_list
    )

    g = torch.Generator()
    g.manual_seed(seed)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

    print(f"✅ Data split reproduced. Val size: {len(val_dataset)}")

    # 3. Loader
    # 强制 batch_size 至少为 2 以支持 Mixer (Swap) 测试
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(2, args.batch_size),
        shuffle=True,  # 随机打乱以生成多样化图片
        num_workers=2,
        pin_memory=True
    )

    # 4. Model
    print("⚙️ Building Model...")
    model = CausalMTLModel(
        model_config=config['model'],
        data_config=config['data']
    ).to(device)

    # 5. Checkpoint
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Weights loaded (Strict).")
    except RuntimeError as e:
        print(f"⚠️ Strict load failed, trying non-strict... {e}")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✅ Weights loaded (Non-Strict).")

    model.eval()

    # 6. Output Directories
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    run_root = os.path.dirname(ckpt_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_root = os.path.join(run_root, f"inference_results_{timestamp}")

    dir_microscope = os.path.join(output_root, "1_Causal_Microscope")
    dir_mixer = os.path.join(output_root, "2_Causal_Mixer_Swap")
    dir_depth = os.path.join(output_root, "3_Depth_Decoupling_Analysis")

    os.makedirs(dir_microscope, exist_ok=True)
    os.makedirs(dir_mixer, exist_ok=True)
    os.makedirs(dir_depth, exist_ok=True)
    print(f"📂 Saving results to: {output_root}")

    # 7. Loop
    print("🎬 Starting Inference Loop...")

    # 获取场景类别映射（如果存在）
    scene_class_map = getattr(full_dataset, 'scene_classes', None)

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Generating")):
        # 全局禁止梯度计算，防止 OOM 和报错
        with torch.no_grad():
            if batch['rgb'].shape[0] < 2:
                continue

            fname = f"sample_{batch_idx:04d}.png"

            try:
                # Task 1: Microscope (Basic Recon)
                save_path_micro = os.path.join(dir_microscope, fname)
                _visualize_microscope(model, batch, device, save_path_micro, scene_class_map)

                # Task 2: Mixer (Swap)
                # 构造单样本 batch 传入 visualizer
                batch_a = {k: v[0:1] for k, v in batch.items()}
                batch_b = {k: v[1:2] for k, v in batch.items()}
                save_path_mixer = os.path.join(dir_mixer, fname)
                _visualize_mixer(model, batch_a, batch_b, device, save_path_mixer, scene_class_map)

                # Task 3: Depth Analysis
                # 修改：直接调用 engine.visualizer 中的函数，保持一致性
                save_path_depth = os.path.join(dir_depth, fname)
                _visualize_depth_task(model, batch, device, save_path_depth)

            except Exception as e:
                print(f"❌ Batch {batch_idx} error: {e}")
                # 打印详细错误栈以便调试
                import traceback
                traceback.print_exc()
                continue

    if hasattr(full_dataset, "close"):
        full_dataset.close()
    print(f"\n✨ All Done! Results saved in: {output_root}")


if __name__ == "__main__":
    main()