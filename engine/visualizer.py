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
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    device = tensor.device
    dtype = tensor.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(3, 1, 1)

    tensor = tensor.detach()
    img = tensor * std_t + mean_t
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

    vmin, vmax = np.percentile(gt_depth, [2, 98])

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
        # [FIXED] 1. 使用 extract_features 提取对齐后的特征
        # 我们需要分别获取 A 和 B 的潜在变量 map
        # 为此，我们需要手动运行一遍 model 的前部分逻辑，或者复用 forward 的输出

        # 为了清晰，我们复用 forward，虽然稍微慢一点点
        out_a = model(rgb_a, stage=2)
        out_b = model(rgb_b, stage=2)

        # === 2. 提取物理分量 ===

        # 几何源 (Structure Source): A
        # 从 A 的 z_s 得到 Normal Map
        Normal_A = model.normal_head(out_a['z_s_map'])  # [1, 3, H, W]

        # 外观源 (Style Source): B
        # 从 B 的 z_p 得到 Albedo (材质颜色)
        Albedo_B = model.albedo_head(out_b['z_p_seg_map'])  # [1, 3, H, W]
        # 从 B 的 h (全局特征) 得到 Lighting (光照)
        # ⚠️ 关键修正：重新提取 B 的 h，确保维度正确
        _, h_b, _ = model.extract_features(rgb_b)
        Light_B = model.light_head(h_b)  # [1, 27]

        # 辅助: A 的 Albedo (用于展示)
        Albedo_A = model.albedo_head(out_a['z_p_seg_map'])

        # === 3. 物理渲染交换 (The Swap) ===
        # 计算混合后的 Shading: A 的形状 + B 的光照
        Shading_Mix = shading_from_normals(Normal_A, Light_B)

        # 对齐尺寸 (以 rgb_a 为准)
        target_size = (rgb_a.shape[2], rgb_a.shape[3])

        if Albedo_B.shape[-2:] != target_size:
            Albedo_B = F.interpolate(Albedo_B, size=target_size, mode='bilinear', align_corners=False)
            Albedo_A = F.interpolate(Albedo_A, size=target_size, mode='bilinear', align_corners=False)
            Shading_Mix = F.interpolate(Shading_Mix, size=target_size, mode='bilinear', align_corners=False)
            # Normal_A 用于显示，也 resize 一下
            Normal_A_Vis = F.interpolate(Normal_A, size=target_size, mode='bilinear', align_corners=False)
        else:
            Normal_A_Vis = Normal_A

        # 最终合成: I = Albedo * Shading
        I_swap = torch.clamp(Albedo_B * Shading_Mix, 0.0, 1.0)

    # --- 4. 转 Numpy 绘图 ---
    input_rgb_a = denormalize_image(batch_a['rgb'][0])
    input_rgb_b = denormalize_image(batch_b['rgb'][0])

    swap_result = I_swap[0].detach().cpu().permute(1, 2, 0).numpy()

    # 可视化中间件
    # Normal 需要归一化到 [0,1] 用于显示: (n+1)/2
    normal_a_vis = (Normal_A_Vis[0].detach().cpu().permute(1, 2, 0).numpy() + 1) / 2.0
    normal_a_vis = np.clip(normal_a_vis, 0, 1)

    albedo_b_vis = Albedo_B[0].detach().cpu().permute(1, 2, 0).numpy()
    shading_mix_vis = Shading_Mix[0].detach().cpu().permute(1, 2, 0).numpy()
    # Shading 只有亮度信息，通常显示为灰度，但维度是 [H,W,3]，可以直接显示
    shading_mix_vis = np.clip(shading_mix_vis, 0, 1)

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Report 2: Causal Mixer (Structure A + Style B)", fontsize=22)

    # 第一行: Source A (Structure)
    axes[0, 0].imshow(input_rgb_a)
    axes[0, 0].set_title("Content Source A (Input)", fontsize=14)

    axes[0, 1].imshow(normal_a_vis)
    axes[0, 1].set_title("Geometry A (Normal Map)\n(From $z_s^A$)", fontsize=14)

    axes[0, 2].imshow(shading_mix_vis, cmap='gray')
    axes[0, 2].set_title("Shading Mix\n(Geom A + Light B)", fontsize=14)

    # 第二行: Source B (Style) + Result
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
    验证 z_s 是否包含了所有的深度信息，以及 z_p 是否泄露信息。
    """
    model.eval()
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)

    with torch.no_grad():
        # [FIXED] 1. 使用统一接口提取特征 (兼容 ResNet/ViT)
        combined_feat, h, _ = model.extract_features(rgb_tensor)

        # 投影
        f_proj = model.proj_f(combined_feat)
        z_s_map = model.projector_s(combined_feat)
        zs_proj = model.proj_z_s(z_s_map)
        z_p_depth_map = model.projector_p_depth(combined_feat)
        zp_depth_proj = model.proj_z_p_depth(z_p_depth_map)

        # 构造输入
        main_feat = torch.cat([f_proj, zs_proj], dim=1)

        # (A) Main Prediction: 完整模型
        pred_main = model.predictor_depth(main_feat, zp_depth_proj)

        # (B) Zs Only: 屏蔽 z_p (传入全零)
        zeros_zp = torch.zeros_like(zp_depth_proj)
        pred_zs = model.predictor_depth(main_feat, zeros_zp)

        # (C) Zp Only: 仅外观 (应该是一团糟，或者也就是平面)
        pred_zp = model.decoder_zp_depth(z_p_depth_map)

    # 数据转换
    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    d_main = pred_main[0].squeeze().cpu().numpy()
    d_zs = pred_zs[0].squeeze().cpu().numpy()
    d_zp = pred_zp[0].squeeze().cpu().numpy()

    # 误差图
    error_map = np.abs(d_main - gt_depth)

    # 绘图
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    fig.suptitle("Report 3: Depth Information Bottleneck (Does $z_p$ leak?)", fontsize=22)

    # 统一 Scale
    vmin, vmax = np.percentile(gt_depth, [2, 98])

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

    im_err = axes[5].imshow(error_map, cmap='hot')
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
    生成多份可视化报告的主调用函数 (main.py 调用此入口)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 获取类别映射 (可选)
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
            # 1. Microscope
            _visualize_microscope(model, sample_a, device, microscope_path, scene_class_map)

            # 2. Mixer (Swap)
            # 构造 batch
            batch_a = {k: v[0:1] for k, v in sample_a.items()}
            batch_b = {k: v[0:1] for k, v in sample_b.items()}
            _visualize_mixer(model, batch_a, batch_b, device, mixer_path, scene_class_map)

            # 3. Depth Analysis
            _visualize_depth_task(model, sample_a, device, depth_path)

        except Exception as e:
            print(f"Error generating report {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Final visualization reports saved to '{save_dir}'.")