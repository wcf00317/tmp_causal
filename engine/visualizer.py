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
    #recon_app_rgb = lab_channels_to_rgb(recon_app_final[0], gt_depth.shape[:2])
    recon_app_rgb = recon_app_final[0].detach().cpu().permute(1, 2, 0).numpy()
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _visualize_mixer(model, batch_a, batch_b, device, save_path, scene_class_map):
    model.eval()

    # --- 1. 提取 A 和 B 的特征 ---
    rgb_a = batch_a['rgb'][0].unsqueeze(0).to(device)
    rgb_b = batch_b['rgb'][0].unsqueeze(0).to(device)

    with torch.no_grad():
        # 获取分解后的 latent
        out_a = model(rgb_a)
        out_b = model(rgb_b)

        z_s_a = out_a['z_s']
        z_p_a = out_a['z_p_depth']  # 注意：这里用 depth 分支的 z_p 还是 seg 分支的视你的假设而定

        z_s_b = out_b['z_s']
        z_p_b = out_b['z_p_depth']

        # 获取 ViT 特征 f (用于主任务预测)
        # 注意：这里需要再次 forward encoder 或者从 out_a 中想办法获取 f
        # 为了简单，我们重新跑一遍 decoder 的部分逻辑
        # 假设我们只验证 z_s 和 z_p 在 Decoder 里的行为

        # --- 2. 交叉生成 (Swap) ---
        # 目标：看看 z_s(A) + z_p(B) 生成的 深度图 是什么样的
        # 预期：结构应该像 A，但细节可能带点 B 的噪声（如果解耦不完美）

        # 为了可视化，我们主要看重构解码器 (Decoder_Geom 和 Decoder_App)

        # 重构 A 的几何
        recon_geom_a, _ = model.decoder_geom(out_a['z_s_map'])
        # 重构 B 的外观 (直接当 RGB 处理，修正之前的 Lab 错误)
        recon_app_b_logits, _ = model.decoder_app(out_b['z_p_seg_map'])
        recon_app_b = torch.sigmoid(recon_app_b_logits)  # [1, 3, H, W] RGB

        # 重构 A 的外观
        recon_app_a_logits, _ = model.decoder_app(out_a['z_p_seg_map'])
        recon_app_a = torch.sigmoid(recon_app_a_logits)

    # --- 3. 数据转 Numpy 用于绘图 ---
    input_rgb_a = denormalize_image(batch_a['rgb'][0])
    input_rgb_b = denormalize_image(batch_b['rgb'][0])

    # 几何图 (Depth)
    geom_a = recon_geom_a.squeeze().cpu().numpy()

    # 外观图 (Appearance) - 修正后的直接 RGB
    app_a = recon_app_a.squeeze().permute(1, 2, 0).cpu().numpy()
    app_b = recon_app_b.squeeze().permute(1, 2, 0).cpu().numpy()

    # --- 4. 绘图：展示解耦的组件 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Causal Disentanglement Analysis (Swap Test)", fontsize=22)

    # 第一行：场景 A
    axes[0, 0].imshow(input_rgb_a)
    axes[0, 0].set_title(f"Input Image A\n({scene_class_map[batch_a['scene_type'][0].item()]})", fontsize=14)

    axes[0, 1].imshow(geom_a, cmap='plasma')
    axes[0, 1].set_title("Structure ($z_s^A$)\nShould contain Shape/Edges", fontsize=14)

    axes[0, 2].imshow(app_a)
    axes[0, 2].set_title("Appearance ($z_p^A$)\nShould contain Texture/Color", fontsize=14)

    # 第二行：场景 B 的外观混入
    axes[1, 0].imshow(input_rgb_b)
    axes[1, 0].set_title(f"Input Image B\n({scene_class_map[batch_b['scene_type'][0].item()]})", fontsize=14)

    axes[1, 1].imshow(app_b)
    axes[1, 1].set_title("Appearance ($z_p^B$)\nSource of Style", fontsize=14)

    # 这里放一个“合成”示意图，虽然我们没有训练 RGB 生成器
    # 我们可以简单的把 Geom A 和 App B 叠在一起展示，但不要相乘
    # 或者留白，或者展示 Task Prediction Swap（如果实现了的话）
    # 这里我们展示 0.5 * Geom A (Gray) + 0.5 * App B (Color) 来看是否对齐
    geom_a_norm = (geom_a - geom_a.min()) / (geom_a.max() - geom_a.min() + 1e-6)
    geom_a_rgb = np.stack([geom_a_norm] * 3, axis=-1)
    overlay = 0.6 * geom_a_rgb + 0.4 * app_b
    axes[1, 2].imshow(np.clip(overlay, 0, 1))
    axes[1, 2].set_title("Overlay: Struct A + App B\n(Check alignment)", fontsize=14)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _visualize_depth_task(model, batch, device, save_path):
    """
    生成一个独立的、专注于深度任务和 z_p_depth 解耦分析的可视化报告。
    【修复版】适配多尺度 ViT，不再手动执行 encoder/projector，而是直接利用 model 输出。
    """
    model.eval()
    idx = 0  # 仅可视化批次中的第一张图

    rgb_tensor = batch['rgb'][idx].unsqueeze(0).to(device)

    # --- 1. 获取模型输出 (使用 safe 的 forward) ---
    with torch.no_grad():
        outputs = model(rgb_tensor)

    # --- 2. 准备绘图所需的数据 ---
    input_rgb = denormalize_image(batch['rgb'][idx])
    gt_depth = batch['depth'][idx].squeeze().cpu().numpy()

    # 模型主干路的最终深度预测
    pred_depth_main = outputs['pred_depth'][0].squeeze().cpu().numpy()

    # 仅由 z_p_depth 解码出的深度图 (检查是否有 'pred_depth_from_zp'，没有则用 zeros 或跳过)
    if 'pred_depth_from_zp' in outputs:
        pred_depth_zp = outputs['pred_depth_from_zp'][0].squeeze().cpu().numpy()
    else:
        # 兼容性处理：如果模型没输出这个，就全黑
        pred_depth_zp = np.zeros_like(pred_depth_main)

    # 计算预测误差图
    error_map = np.abs(pred_depth_main - gt_depth)

    # --- 3. 开始绘图 ---
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle("Depth Task & $z_{p,depth}$ Decoupling Analysis", fontsize=22)

    # 使用百分位数来稳健地统一所有深度图的颜色范围
    vmin, vmax = np.percentile(gt_depth, [2, 98])

    # 图 1: 输入 RGB
    axes[0].imshow(input_rgb)
    axes[0].set_title("Input RGB", fontsize=16)

    # 图 2: 真实深度
    axes[1].imshow(gt_depth, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth Depth", fontsize=16)

    # 图 3: 主路预测深度
    axes[2].imshow(pred_depth_main, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[2].set_title("Main Predicted Depth", fontsize=16)

    # 图 4: 从 z_p_depth 解码的深度
    im_zp = axes[3].imshow(pred_depth_zp, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[3].set_title("Depth from $z_{p,depth}$ only\n(Should be empty/noise)", fontsize=16)

    # 图 5: 预测误差
    im_err = axes[4].imshow(error_map, cmap='hot')
    axes[4].set_title("Prediction Error", fontsize=16)

    # 美化图像
    for ax in axes.flat:
        ax.axis('off')

    # Colorbars
    fig.colorbar(im_zp, ax=axes[3], fraction=0.046, pad=0.04)
    fig.colorbar(im_err, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> 已保存深度分析可视化报告: {save_path}")

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

        # 原有的可视化文件路径
        microscope_path = os.path.join(save_dir, f"report_1_microscope_{i + 1}.png")
        mixer_path = os.path.join(save_dir, f"report_2_mixer_{i + 1}.png")

        # 为新的深度分析报告定义文件路径
        depth_analysis_path = os.path.join(save_dir, f"report_3_depth_analysis_{i + 1}.png")

        _visualize_microscope(model, sample_a, device, microscope_path, scene_class_map)
        _visualize_mixer(model, sample_a, sample_b, device, mixer_path, scene_class_map)

        # 增加对新函数的调用，使用 sample_a 进行分析
        _visualize_depth_task(model, sample_a, device, depth_analysis_path)


    print(f"✅ Final visualization reports saved to '{save_dir}'.")
