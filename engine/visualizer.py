import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.color import lab2rgb
from torch.utils.data import Subset
import torch.nn.functional as F
from models.layers.shading import shading_from_normals


# -----------------------------------------------------------------------------
# 鲁棒的辅助函数
# -----------------------------------------------------------------------------

def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    智能去归一化函数：
    1. 自动识别 0-255 格式 -> 除以 255
    2. 自动识别 0-1 格式 (您的数据情况) -> 直接返回
    3. 自动识别 Standard Norm 格式 -> 执行去归一化
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    device = tensor.device
    # 先 detach 转 CPU，避免后续计算影响梯度
    t_data = tensor.detach()

    t_max = t_data.max().item()
    t_min = t_data.min().item()

    # Case A: 0-255 格式 (数值很大)
    if t_max > 10.0:
        img = t_data / 255.0

    # Case B: 0-1 格式 (您的数据: min=0.03, max=0.8)
    # 只要最小值非负且最大值<=1，就认为是已经归一化好的 RGB
    elif t_min >= 0.0 and t_max <= 1.0:
        img = t_data

    # Case C: ImageNet 标准化格式 (通常包含负数，因为 mean=0.485)
    else:
        mean_t = torch.tensor(mean, device=device, dtype=tensor.dtype).view(3, 1, 1)
        std_t = torch.tensor(std, device=device, dtype=tensor.dtype).view(3, 1, 1)
        img = t_data * std_t + mean_t

    # 最后的安全钳制
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).cpu().numpy()


def _visualize_microscope(model, batch, device, save_path, scene_class_map):
    """
    Report 1: 基础重构能力检查 (Microscope)
    """
    model.eval()
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)

    with torch.no_grad():
        # [FIXED] 使用 extract_features 接口
        outputs = model(rgb_tensor)

    recon_geom_final = outputs['recon_geom']
    recon_app_final = outputs['recon_app']  # [1, 3, H, W] RGB

    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    # 场景名 (如果有)
    if 'scene_type' in batch and batch['scene_type'].dim() > 0:
        gt_scene_idx = batch['scene_type'][idx].item()
        gt_scene_name = scene_class_map[gt_scene_idx] if scene_class_map else str(gt_scene_idx)
    else:
        gt_scene_name = "N/A"

    recon_geom_raw = recon_geom_final[0].squeeze().cpu().numpy()
    recon_app_rgb = recon_app_final[0].detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    title = f"Report 1: Causal Microscope (Self-Reconstruction)\nGT Scene: '{gt_scene_name}'"
    fig.suptitle(title, fontsize=22)

    # === [FIX] 排除 0 值计算显示范围 ===
    valid_mask = gt_depth > 0.001
    if valid_mask.sum() > 0:
        vmin, vmax = np.percentile(gt_depth[valid_mask], [2, 98])
    else:
        vmin, vmax = 0, 1
    # =================================

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input RGB", fontsize=16)

    axes[1].imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth: Depth", fontsize=16)

    axes[2].imshow(recon_geom_raw, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title("Recon Geometry ($z_s$)\n(Should match Depth)", fontsize=16)

    axes[3].imshow(recon_app_rgb)
    axes[3].set_title("Recon Appearance ($z_p$)\n(Should match Color)", fontsize=16)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Microscope: {save_path}")


def _visualize_mixer(model, batch_a, batch_b, device, save_path, scene_class_map):
    """
    Report 2: Causal Mixer (Counterfactual Generation)
    采用物理本征分解 (Albedo x Shading) 以获得清晰的交换效果。
    """
    model.eval()

    rgb_a = batch_a['rgb'][0:1].to(device)
    rgb_b = batch_b['rgb'][0:1].to(device)

    with torch.no_grad():
        # 复用 forward 提取特征
        out_a = model(rgb_a, stage=2)
        out_b = model(rgb_b, stage=2)

        # === 提取物理分量 ===
        # 几何源: A
        Normal_A = model.normal_head(out_a['z_s_map'])  # [1, 3, H, W]

        # 外观源: B
        Albedo_B = model.albedo_head(out_b['z_p_seg_map'])  # [1, 3, H, W]

        # 光照源: B (需要重新提取 h)
        _, h_b, _ = model.extract_features(rgb_b)
        Light_B = model.light_head(h_b)  # [1, 27]

        # === 物理渲染交换 ===
        Shading_Mix = shading_from_normals(Normal_A, Light_B)

        # 对齐尺寸
        target_size = (rgb_a.shape[2], rgb_a.shape[3])

        if Albedo_B.shape[-2:] != target_size:
            Albedo_B = F.interpolate(Albedo_B, size=target_size, mode='bilinear', align_corners=False)
            Shading_Mix = F.interpolate(Shading_Mix, size=target_size, mode='bilinear', align_corners=False)
            Normal_A_Vis = F.interpolate(Normal_A, size=target_size, mode='bilinear', align_corners=False)
        else:
            Normal_A_Vis = Normal_A

        # 合成: I = Albedo * Shading
        I_swap = torch.clamp(Albedo_B * Shading_Mix, 0.0, 1.0)

    # --- 绘图 ---
    input_rgb_a = denormalize_image(batch_a['rgb'][0])
    input_rgb_b = denormalize_image(batch_b['rgb'][0])

    swap_result = I_swap[0].detach().cpu().permute(1, 2, 0).numpy()

    # Normal 显示归一化: [-1,1] -> [0,1]
    normal_a_vis = (Normal_A_Vis[0].detach().cpu().permute(1, 2, 0).numpy() + 1) / 2.0
    normal_a_vis = np.clip(normal_a_vis, 0, 1)

    albedo_b_vis = Albedo_B[0].detach().cpu().permute(1, 2, 0).numpy()
    shading_mix_vis = Shading_Mix[0].detach().cpu().permute(1, 2, 0).numpy()
    shading_mix_vis = np.clip(shading_mix_vis, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Report 2: Causal Mixer (Structure A + Style B)", fontsize=22)

    axes[0, 0].imshow(input_rgb_a)
    axes[0, 0].set_title("Content Source A (Input)", fontsize=14)

    axes[0, 1].imshow(normal_a_vis)
    axes[0, 1].set_title("Geometry A (Normal Map)\n(From $z_s^A$)", fontsize=14)

    axes[0, 2].imshow(shading_mix_vis, cmap='gray')
    axes[0, 2].set_title("Shading Mix\n(Geom A + Light B)", fontsize=14)

    axes[1, 0].imshow(input_rgb_b)
    axes[1, 0].set_title("Style Source B (Input)", fontsize=14)

    axes[1, 1].imshow(albedo_b_vis)
    axes[1, 1].set_title("Texture B (Albedo)\n(From $z_p^B$)", fontsize=14)

    axes[1, 2].imshow(swap_result)
    axes[1, 2].set_title("Final Swap Result\n(Albedo B $\\times$ Shading A)", fontsize=16, color='darkred',
                         fontweight='bold')

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Mixer: {save_path}")


def _visualize_depth_task(model, batch, device, save_path):
    """
    Report 3: Depth Decoupling Analysis
    """
    model.eval()
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)

    with torch.no_grad():
        combined_feat, h, _ = model.extract_features(rgb_tensor)
        f_proj = model.proj_f(combined_feat)
        z_s_map = model.projector_s(combined_feat)
        zs_proj = model.proj_z_s(z_s_map)
        z_p_depth_map = model.projector_p_depth(combined_feat)
        zp_depth_proj = model.proj_z_p_depth(z_p_depth_map)

        main_feat = torch.cat([f_proj, zs_proj], dim=1)
        pred_main = model.predictor_depth(main_feat, zp_depth_proj)

        zeros_zp = torch.zeros_like(zp_depth_proj)
        pred_zs = model.predictor_depth(main_feat, zeros_zp)

        pred_zp = model.decoder_zp_depth(z_p_depth_map)

    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    d_main = pred_main[0].squeeze().cpu().numpy()
    d_zs = pred_zs[0].squeeze().cpu().numpy()
    d_zp = pred_zp[0].squeeze().cpu().numpy()

    error_map = np.abs(d_main - gt_depth)

    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    fig.suptitle("Report 3: Depth Information Bottleneck (Does $z_p$ leak?)", fontsize=22)

    # === [FIX] 排除 0 值计算显示范围 ===
    valid_mask = gt_depth > 0.001
    if valid_mask.sum() > 0:
        vmin, vmax = np.percentile(gt_depth[valid_mask], [2, 98])
    else:
        vmin, vmax = 0, 1
    # =================================

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input RGB", fontsize=16)

    axes[1].imshow(gt_depth, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth Depth", fontsize=16)

    axes[2].imshow(d_main, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[2].set_title("Main Prediction\n($f + z_s + z_p$)", fontsize=16)

    axes[3].imshow(d_zs, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[3].set_title("Structure Only ($z_s$)\n(Should be clear)", fontsize=16)

    axes[4].imshow(d_zp, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[4].set_title("Appearance Only ($z_p$)\n(Should be noise/flat)", fontsize=16)

    im_err = axes[5].imshow(error_map, cmap='hot', vmin=0, vmax=vmax * 0.5)
    axes[5].set_title("Prediction Error", fontsize=16)

    for ax in axes.flat: ax.axis('off')
    fig.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Depth Analysis: {save_path}")


@torch.no_grad()
def generate_visual_reports(model, data_loader, device, save_dir="visualizations_final", num_reports=3):
    """
    生成多份可视化报告的主调用函数
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data_loader.dataset, Subset):
        dataset_obj = data_loader.dataset.dataset
    else:
        dataset_obj = data_loader.dataset
    scene_class_map = getattr(dataset_obj, 'scene_classes', None)

    try:
        it = iter(data_loader)
        samples = [next(it) for _ in range(num_reports * 2)]
    except StopIteration:
        print("Not enough samples for visualization.")
        return

    print(f"Generating {num_reports} sets of visualization reports...")

    for i in range(num_reports):
        sample_a = samples[i * 2]
        sample_b = samples[i * 2 + 1]

        microscope_path = os.path.join(save_dir, f"report_1_microscope_{i + 1}.png")
        mixer_path = os.path.join(save_dir, f"report_2_mixer_{i + 1}.png")
        depth_path = os.path.join(save_dir, f"report_3_depth_analysis_{i + 1}.png")

        try:
            _visualize_microscope(model, sample_a, device, microscope_path, scene_class_map)

            batch_a = {k: v[0:1] for k, v in sample_a.items()}
            batch_b = {k: v[0:1] for k, v in sample_b.items()}
            _visualize_mixer(model, batch_a, batch_b, device, mixer_path, scene_class_map)

            _visualize_depth_task(model, sample_a, device, depth_path)

        except Exception as e:
            print(f"Error generating report {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Final visualization reports saved to '{save_dir}'.")