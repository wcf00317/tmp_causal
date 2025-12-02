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
# 我们只导入基础的 visualizer 函数，深度分析函数我们在本文件重写
from engine.visualizer import _visualize_microscope, _visualize_mixer, denormalize_image
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


# --- 【核心修改】将深度解耦分析逻辑直接写在这里，确保包含 Zs-only ---
# --- 【修复版】本地定义的深度分析函数 ---
def local_visualize_depth_task(model, batch, device, save_path):
    """
    本地定义的深度分析函数，强制包含 Zs-Only 和 Zp-Only 的对比。
    【修复】适配 GatedSegDepthDecoder 的双参数接口 (main_feat, z_p_feat)。
    """
    model.eval()
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        # 1. 手动拆解模型前向过程
        # 编码
        features = model.encoder(rgb_tensor)  # List[Tensor]
        combined_feat = torch.cat(features, dim=1)  # [1, C*4, 14, 14]

        # 投影
        f_proj = model.proj_f(combined_feat)

        # Z_s 分支
        z_s_map = model.projector_s(combined_feat)
        zs_proj = model.proj_z_s(z_s_map)

        # Z_p 分支 (Depth)
        z_p_depth_map = model.projector_p_depth(combined_feat)
        zp_depth_proj = model.proj_z_p_depth(z_p_depth_map)

        # --- 构造输入 ---
        # GatedDecoder 需要两个输入: (main_feat, z_p_feat)
        # main_feat 通常是 f_proj 和 zs_proj 的拼接
        main_feat = torch.cat([f_proj, zs_proj], dim=1)

        # (A) Main Prediction: 完整模型 (z_p 参与门控)
        pred_main = model.predictor_depth(main_feat, zp_depth_proj)

        # (B) Zs Only: 屏蔽 z_p (传入全零作为门控条件)
        # 这将测试仅靠 f 和 zs 能恢复多少结构
        zeros_zp = torch.zeros_like(zp_depth_proj)
        pred_zs = model.predictor_depth(main_feat, zeros_zp)

        # (C) Zp Only: 仅外观 (应该是一团糟/噪声)
        # 使用专门的辅助解码器 decoder_zp_depth (它只接受 z_p_map)
        pred_zp = model.decoder_zp_depth(z_p_depth_map)

    # 2. 数据转换
    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    d_main = pred_main[0].squeeze().cpu().numpy()
    d_zs = pred_zs[0].squeeze().cpu().numpy()
    d_zp = pred_zp[0].squeeze().cpu().numpy()

    # 误差图
    error_map = np.abs(d_main - gt_depth)

    # 3. 绘图 (1行6列)
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    fig.suptitle("Causal Depth Analysis: Can $z_s$ alone recover geometry?", fontsize=22)

    # 统一色阶
    vmin, vmax = np.percentile(gt_depth, [2, 98])

    # Col 1: RGB
    axes[0].imshow(input_rgb)
    axes[0].set_title("Input RGB", fontsize=16)

    # Col 2: GT
    axes[1].imshow(gt_depth, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth", fontsize=16)

    # Col 3: Main
    axes[2].imshow(d_main, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[2].set_title("Main Prediction\n($f + z_s + z_p$)", fontsize=16)

    # Col 4: Zs Only (重点!)
    axes[3].imshow(d_zs, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[3].set_title("Structure Only ($z_s$)\n(Should be clear)", fontsize=16)

    # Col 5: Zp Only
    axes[4].imshow(d_zp, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[4].set_title("Appearance Only ($z_p$)\n(Should be noise)", fontsize=16)

    # Col 6: Error
    im_err = axes[5].imshow(error_map, cmap='hot')
    axes[5].set_title("Prediction Error", fontsize=16)

    for ax in axes.flat: ax.axis('off')

    fig.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Depth Analysis: {save_path}")

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
    scene_class_map = full_dataset.scene_classes

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
                batch_a = {k: v[0:1] for k, v in batch.items()}
                batch_b = {k: v[1:2] for k, v in batch.items()}
                save_path_mixer = os.path.join(dir_mixer, fname)
                _visualize_mixer(model, batch_a, batch_b, device, save_path_mixer, scene_class_map)

                # Task 3: Depth Analysis (调用本地定义的函数!)
                save_path_depth = os.path.join(dir_depth, fname)
                local_visualize_depth_task(model, batch, device, save_path_depth)

            except Exception as e:
                print(f"❌ Batch {batch_idx} error: {e}")
                continue

    if hasattr(full_dataset, "close"):
        full_dataset.close()
    print(f"\n✨ All Done! Results saved in: {output_root}")


if __name__ == "__main__":
    main()