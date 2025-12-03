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

    # 【修复】增加 detach() 防止梯度报错
    tensor = tensor.detach()

    img = tensor * std_t + mean_t
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).cpu().numpy()


def _visualize_microscope(model, batch, device, save_path, scene_class_map):
    """
    基础重构能力检查
    """
    model.eval()
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(rgb_tensor)

    recon_geom_final = outputs['recon_geom']
    recon_app_final = outputs['recon_app']  # [1, 3, H, W] RGB

    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    # 场景名
    if 'scene_type' in batch and batch['scene_type'].dim() > 0:
        gt_scene_idx = batch['scene_type'][idx].item()
        gt_scene_name = scene_class_map[gt_scene_idx] if scene_class_map else str(gt_scene_idx)
    else:
        gt_scene_name = "Unknown"

    recon_geom_raw = recon_geom_final[0].squeeze().cpu().numpy()
    # 【修复】直接作为 RGB 处理
    recon_app_rgb = recon_app_final[0].detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    title = f"Causal Microscope\nGT Scene: '{gt_scene_name}'"
    fig.suptitle(title, fontsize=22)

    vmin, vmax = np.percentile(gt_depth, [2, 98])

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input RGB", fontsize=16)

    axes[1].imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth: Depth", fontsize=16)

    axes[2].imshow(recon_geom_raw, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title("Recon Geometry ($z_s$)", fontsize=16)

    axes[3].imshow(recon_app_rgb)
    axes[3].set_title("Recon Appearance ($z_p$)", fontsize=16)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Microscope: {save_path}")


def _visualize_mixer(model, batch_a, batch_b, device, save_path, scene_class_map):
    """
    Swap Test: 真正的物理交换 (Albedo * Shading)
    【修复版】恢复6张图布局，增加 Albedo A 的显示，并修复 Shading 警告。
    """
    model.eval()

    # --- 1. 提取 A 和 B 的特征 ---
    rgb_a = batch_a['rgb'][0:1].to(device)
    rgb_b = batch_b['rgb'][0:1].to(device)

    with torch.no_grad():
        # 获取特征图和潜变量
        out_a = model(rgb_a, stage=2)
        out_b = model(rgb_b, stage=2)

        # === 核心修改：真正的生成式交换 ===

        # 1. 获取材质 (Albedo)
        # B 的材质 (用于交换)
        Albedo_B = model.albedo_head(out_b['z_p_seg_map'])
        # 【新增】A 的材质 (用于填补右上角可视化，对比用)
        Albedo_A = model.albedo_head(out_a['z_p_seg_map'])

        # 2. 获取 A 的几何 (Normal) -> 来自 z_s^A
        Normal_A = model.normal_head(out_a['z_s_map'])

        # 3. 获取 B 的光照 (Light) -> 来自 h^B
        # 这里的 h_b 取自最后一层特征的均值
        h_b = model.encoder(rgb_b)[-1].mean(dim=[2, 3])
        Light_B = model.light_head(h_b)

        # 4. 计算新的 Shading (A的形状 + B的光)
        Shading_A_with_Light_B = shading_from_normals(Normal_A, Light_B)

        # 5. 上采样对齐
        target_size = (rgb_a.shape[2], rgb_a.shape[3])

        # 对所有需要可视化的图进行 Resize
        if Albedo_B.shape[-2:] != target_size:
            Albedo_B = F.interpolate(Albedo_B, size=target_size, mode='bilinear', align_corners=False)
            Albedo_A = F.interpolate(Albedo_A, size=target_size, mode='bilinear', align_corners=False)  # 【新增】
            Shading_A_with_Light_B = F.interpolate(Shading_A_with_Light_B, size=target_size, mode='bilinear',
                                                   align_corners=False)

        # 6. 合成最终图像 (I = A * S)
        I_swap = torch.clamp(Albedo_B * Shading_A_with_Light_B, 0.0, 1.0)

    # --- 2. 转 Numpy 用于绘图 ---
    input_rgb_a = denormalize_image(batch_a['rgb'][0])
    input_rgb_b = denormalize_image(batch_b['rgb'][0])

    # 结果转 numpy
    swap_result = I_swap[0].detach().cpu().permute(1, 2, 0).numpy()

    # 中间变量
    albedo_b_vis = Albedo_B[0].detach().cpu().permute(1, 2, 0).numpy()
    albedo_a_vis = Albedo_A[0].detach().cpu().permute(1, 2, 0).numpy()  # 【新增】

    # 【修复】Shading 可能 > 1.0，导致 matplotlib warning，这里手动 clip
    shading_mix_vis = Shading_A_with_Light_B[0].detach().cpu().permute(1, 2, 0).numpy()
    shading_mix_vis = np.clip(shading_mix_vis, 0.0, 1.0)

    # 获取场景名
    def get_scene_name(batch):
        if 'scene_type' in batch and batch['scene_type'].dim() > 0:
            idx = batch['scene_type'][0].item()
            return scene_class_map[idx] if scene_class_map else str(idx)
        return "Unknown"

    name_a = get_scene_name(batch_a)
    name_b = get_scene_name(batch_b)

    # --- 3. 绘图 (恢复 6 张图) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("True Causal Swap (Intrinsic Decomposition)", fontsize=22)

    # Row 1: Source A 相关
    # (0,0) 输入 A
    axes[0, 0].imshow(input_rgb_a)
    axes[0, 0].set_title(f"Content Source A\n({name_a})", fontsize=14)

    # (0,1) 结构 A + 光照 B (Shading)
    axes[0, 1].imshow(shading_mix_vis, cmap='gray')  # Shading 通常用灰度展示更好，或者保持默认
    axes[0, 1].set_title("Structure A + Light B\n(Shading from $z_s^A, L^B$)", fontsize=14)

    # (0,2) 【恢复位置】 A 的材质 (Albedo A)
    axes[0, 2].imshow(albedo_a_vis)
    axes[0, 2].set_title("Albedo A ($z_p^A$)\n(Intrinsic Texture of A)", fontsize=14)

    # Row 2: Source B 相关 + 结果
    # (1,0) 输入 B
    axes[1, 0].imshow(input_rgb_b)
    axes[1, 0].set_title(f"Style Source B\n({name_b})", fontsize=14)

    # (1,1) B 的材质 (Albedo B)
    axes[1, 1].imshow(albedo_b_vis)
    axes[1, 1].set_title("Albedo B ($z_p^B$)\n(Texture Source)", fontsize=14)

    # (1,2) 结果 (Swap)
    axes[1, 2].imshow(swap_result)
    axes[1, 2].set_title("Result: Albedo B $\\times$ Shading A\n(True Generative Swap)", fontsize=16, color='red')

    for ax in axes.flat:
        ax.axis('off')  # 统一关闭坐标轴

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Mixer: {save_path}")

def _visualize_depth_task(model, batch, device, save_path):
    """
    生成深度任务解耦分析报告。
    【修复版】适配多尺度 ViT 和 GatedDecoder 接口。
    """
    model.eval()
    idx = 0
    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)

    with torch.no_grad():
        # --- 1. 手动拆解模型前向过程 ---
        # 编码
        features = model.encoder(rgb_tensor)
        combined_feat = torch.cat(features, dim=1)

        # 投影
        f_proj = model.proj_f(combined_feat)
        z_s_map = model.projector_s(combined_feat)
        zs_proj = model.proj_z_s(z_s_map)
        z_p_depth_map = model.projector_p_depth(combined_feat)
        zp_depth_proj = model.proj_z_p_depth(z_p_depth_map)

        # --- 构造输入 ---
        main_feat = torch.cat([f_proj, zs_proj], dim=1)

        # (A) Main Prediction: 完整模型 (z_p 参与门控)
        pred_main = model.predictor_depth(main_feat, zp_depth_proj)

        # (B) Zs Only: 屏蔽 z_p (传入全零)
        zeros_zp = torch.zeros_like(zp_depth_proj)
        pred_zs = model.predictor_depth(main_feat, zeros_zp)

        # (C) Zp Only: 仅外观 (应该是一团糟)
        pred_zp = model.decoder_zp_depth(z_p_depth_map)

    # --- 2. 数据转换 ---
    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    d_main = pred_main[0].squeeze().cpu().numpy()
    d_zs = pred_zs[0].squeeze().cpu().numpy()
    d_zp = pred_zp[0].squeeze().cpu().numpy()

    error_map = np.abs(d_main - gt_depth)

    # --- 3. 绘图 (6列) ---
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    fig.suptitle("Causal Depth Analysis: Can $z_s$ alone recover geometry?", fontsize=22)

    vmin, vmax = np.percentile(gt_depth, [2, 98])

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input RGB", fontsize=16)

    axes[1].imshow(gt_depth, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth", fontsize=16)

    axes[2].imshow(d_main, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[2].set_title("Main Prediction\n($f + z_s + z_p$)", fontsize=16)

    axes[3].imshow(d_zs, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[3].set_title("Structure Only ($z_s$)\n(Should be clear)", fontsize=16)

    axes[4].imshow(d_zp, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[4].set_title("Appearance Only ($z_p$)\n(Should be noise)", fontsize=16)

    im_err = axes[5].imshow(error_map, cmap='hot')
    axes[5].set_title("Prediction Error", fontsize=16)

    for ax in axes.flat: ax.axis('off')
    fig.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved Visualization: {save_path}")


@torch.no_grad()
def generate_visual_reports(model, data_loader, device, save_dir="visualizations_final", num_reports=3):
    """
    生成多份可视化报告的主调用函数 (main.py 调用此入口)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 获取类别映射
    if isinstance(data_loader.dataset, Subset):
        dataset_obj = data_loader.dataset.dataset
    else:
        dataset_obj = data_loader.dataset

    scene_class_map = getattr(dataset_obj, 'scene_classes', None)

    # 确保有足够数据
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

        # 路径
        microscope_path = os.path.join(save_dir, f"report_1_microscope_{i + 1}.png")
        mixer_path = os.path.join(save_dir, f"report_2_mixer_{i + 1}.png")
        depth_path = os.path.join(save_dir, f"report_3_depth_analysis_{i + 1}.png")

        try:
            # 1. Microscope
            _visualize_microscope(model, sample_a, device, microscope_path, scene_class_map)

            # 2. Mixer (构造 batch_a, batch_b)
            # visualizer 内部只取 batch['rgb'][0], 所以可以直接传 sample
            # 但为了严谨，我们构造单样本 batch
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