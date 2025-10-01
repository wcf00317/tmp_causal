import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.color import lab2rgb
from torch.utils.data import Subset


# -----------------------------------------------------------------------------
# 鲁棒的辅助函数 (保持不变)
# -----------------------------------------------------------------------------

def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    device = tensor.device
    dtype = tensor.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(3, 1, 1)
    img = tensor.cpu() * std_t.cpu() + mean_t.cpu()
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def lab_channels_to_rgb(lab_ab_channels, img_size, ab_range=128.0, L_value=70.0):
    if lab_ab_channels.dim() == 4:
        lab_ab = lab_ab_channels.squeeze(0)
    else:
        lab_ab = lab_ab_channels
    lab_ab = lab_ab.detach().cpu().numpy()
    if lab_ab.max() <= 1.0 + 1e-6 and lab_ab.min() >= -1.0 - 1e-6:
        lab_ab = lab_ab * ab_range
    H, W = img_size
    L = np.full((H, W), L_value, dtype=np.float32)
    lab_image = np.stack([L, lab_ab[0], lab_ab[1]], axis=-1)
    rgb = lab2rgb(lab_image.astype(np.float64))
    return np.clip(rgb, 0, 1)


def fuse_geom_and_app(geom_map, app_map):
    if isinstance(geom_map, torch.Tensor):
        geom_np = geom_map.detach().cpu().numpy()
    else:
        geom_np = np.array(geom_map)
    lo = np.percentile(geom_np, 2)
    hi = np.percentile(geom_np, 98)
    if hi - lo < 1e-6:
        geom_norm = (geom_np - geom_np.min()) / (geom_np.max() - geom_np.min() + 1e-6)
    else:
        geom_norm = (geom_np - lo) / (hi - lo)
    geom_norm = np.clip(geom_norm, 0, 1)
    geom_shading = np.stack([geom_norm] * 3, axis=-1)
    hybrid = geom_shading * app_map.astype(np.float32)
    return np.clip(hybrid, 0.0, 1.0)


def _visualize_microscope(model, batch, device, save_path, scene_class_map):
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)
    outputs = model(rgb_tensor)

    # --- 核心修复 1: 只取重构的第一个输出 (final) ---
    recon_geom_final = outputs['recon_geom']
    recon_app_final = outputs['recon_app']

    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()
    gt_scene_idx = batch['scene_type'][idx].item()
    pred_scene_idx = outputs['pred_scene'].argmax(dim=1)[0].item()
    gt_scene_name = scene_class_map[gt_scene_idx]
    pred_scene_name = scene_class_map[pred_scene_idx]

    recon_geom_raw = recon_geom_final[0].squeeze().cpu().numpy()
    recon_app_rgb = lab_channels_to_rgb(recon_app_final[0], gt_depth.shape[:2])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    title = f"Causal Microscope\nGT Scene: '{gt_scene_name}' | Pred Scene: '{pred_scene_name}' ({'Correct' if gt_scene_idx == pred_scene_idx else 'Wrong'})"
    fig.suptitle(title, fontsize=22)
    vmin, vmax = np.percentile(gt_depth, 2), np.percentile(gt_depth, 98)
    axes[0].imshow(input_rgb);
    axes[0].set_title("Input RGB", fontsize=16)
    axes[1].imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax);
    axes[1].set_title("Ground Truth: Depth (Geometry)", fontsize=16)
    axes[2].imshow(recon_geom_raw, cmap='viridis', vmin=vmin, vmax=vmax);
    axes[2].set_title("Model's Understanding:\nGeometry from $z_s$", fontsize=16)
    axes[3].imshow(recon_app_rgb);
    axes[3].set_title("Model's Understanding:\nAppearance from $z_{p,seg}$", fontsize=16)
    for ax in axes.flat: ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _visualize_mixer(model, batch_a, batch_b, device, save_path, scene_class_map):
    rgb_a = batch_a['rgb'][0].unsqueeze(0).to(device);
    out_a = model(rgb_a);
    z_s_a, z_p_a = out_a['z_s'], out_a['z_p_seg'];
    gt_scene_a = scene_class_map[batch_a['scene_type'][0].item()]
    rgb_b = batch_b['rgb'][0].unsqueeze(0).to(device);
    out_b = model(rgb_b);
    z_s_b, z_p_b = out_b['z_s'], out_b['z_p_seg'];
    gt_scene_b = scene_class_map[batch_b['scene_type'][0].item()]
    img_size = tuple(rgb_a.shape[2:])

    def create_hybrid(z_s, z_p):
        # --- 核心修复 2: 调用解码器后，只取第一个返回值 ---
        geom_final, _ = model.decoder_geom(z_s)
        app_final, _ = model.decoder_app(z_p)
        geom = geom_final.squeeze()
        app = lab_channels_to_rgb(app_final.squeeze(), img_size)
        return fuse_geom_and_app(geom, app)

    hybrid_A_A = create_hybrid(z_s_a, z_p_a);
    hybrid_A_B = create_hybrid(z_s_a, z_p_b);
    hybrid_B_A = create_hybrid(z_s_b, z_p_a);
    hybrid_B_B = create_hybrid(z_s_b, z_p_b)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    # ... (绘图的其余部分保持不变) ...
    fig.suptitle("Visualization Report 2: The Causal Mixer", fontsize=22, y=0.97)
    axes[0, 0].imshow(hybrid_A_A);
    axes[0, 0].set_title(f"Reconstructed Scene A\n(GT Scene: '{gt_scene_a}')", fontsize=14)
    axes[0, 1].imshow(hybrid_A_B);
    axes[0, 1].set_title(f"Hybrid: Geom A + App B", fontsize=14)
    axes[1, 0].imshow(hybrid_B_A);
    axes[1, 0].set_title(f"Hybrid: Geom B + App A", fontsize=14)
    axes[1, 1].imshow(hybrid_B_B);
    axes[1, 1].set_title(f"Reconstructed Scene B\n(GT Scene: '{gt_scene_b}')", fontsize=14)
    pad = 5
    axes[0, 0].annotate('Geometry from A', xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[0, 0].yaxis.label, textcoords='offset points', size='large', ha='right',
                        va='center', rotation=90)
    axes[1, 0].annotate('Geometry from B', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points', size='large', ha='right',
                        va='center', rotation=90)
    axes[0, 0].annotate('Appearance from A', xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction',
                        textcoords='offset points', size='large', ha='center', va='baseline')
    axes[0, 1].annotate('Appearance from B', xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction',
                        textcoords='offset points', size='large', ha='center', va='baseline')
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

@torch.no_grad()
def generate_visual_reports(model, data_loader, device, save_dir="visualizations_final", num_reports=3):
    """
    生成多份可视化报告的主调用函数。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data_loader.dataset, Subset):
        scene_class_map = data_loader.dataset.dataset.scene_classes
    else:
        scene_class_map = data_loader.dataset.scene_classes

    # 确保我们有足够的数据来生成报告
    try:
        it = iter(data_loader)
        samples = [next(it) for _ in range(num_reports * 2)]
    except StopIteration:
        raise RuntimeError(f"DataLoader must contain at least {num_reports * 2} batches for visualization.")

    print(f"Generating {num_reports} sets of visualization reports...")

    for i in range(num_reports):
        sample_a = samples[i * 2]
        sample_b = samples[i * 2 + 1]

        print(f"--- Generating Report Set {i + 1}/{num_reports} ---")

        # 为每个报告生成唯一的文件名
        microscope_path = os.path.join(save_dir, f"report_1_microscope_{i + 1}.png")
        mixer_path = os.path.join(save_dir, f"report_2_mixer_{i + 1}.png")

        _visualize_microscope(model, sample_a, device, microscope_path, scene_class_map)
        _visualize_mixer(model, sample_a, sample_b, device, mixer_path, scene_class_map)

    print(f"✅ Final visualization reports saved to '{save_dir}'.")