# 文件: engine/evaluator.py
# 版本：【已升级，增加可视化指标监控 + 稳健 quick_diagnose】

import os
import logging
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, stage):
    """
    评估模型，并返回所有任务及重构任务的指标。
    """
    model.eval()

    # --- 1. 初始化所有指标对象 ---
    num_seg_classes = model.predictor_seg.output_channels
    num_scene_classes = model.predictor_scene.out_features

    miou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=num_seg_classes, ignore_index=255).to(device)
    scene_acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_scene_classes).to(device)
    depth_mse_metric = torchmetrics.regression.MeanSquaredError().to(device)
    depth_mae_metric = torchmetrics.regression.MeanAbsoluteError().to(device)

    # --- 2. 跟踪损失 ---
    total_val_loss = 0.0
    total_recon_geom_loss = 0.0
    total_recon_app_loss = 0.0
    total_independence_loss = 0.0

    total_cka_seg = 0.0
    total_cka_depth = 0.0
    total_cka_scene = 0.0

    pbar = tqdm(val_loader, desc="Evaluating", leave=False)

    for batch in pbar:
        rgb = batch['rgb'].to(device)
        targets_on_device = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        outputs = model(rgb, stage=stage)

        crit_out = criterion(outputs, targets_on_device)
        if isinstance(crit_out, (tuple, list)):
            _, loss_dict = crit_out
        else:
            loss_dict = crit_out

        total_val_loss += loss_dict.get('total_loss', torch.tensor(0.0)).item()
        total_recon_geom_loss += loss_dict.get('recon_geom_loss', torch.tensor(0.0)).item()
        total_recon_app_loss += loss_dict.get('recon_app_loss', torch.tensor(0.0)).item()
        total_independence_loss += loss_dict.get('independence_loss', torch.tensor(0.0)).item()

        total_cka_seg += loss_dict.get('cka_seg', torch.tensor(0.0)).item()
        total_cka_depth += loss_dict.get('cka_depth', torch.tensor(0.0)).item()
        total_cka_scene += loss_dict.get('cka_scene', torch.tensor(0.0)).item()

        miou_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])
        scene_acc_metric.update(outputs['pred_scene'], targets_on_device['scene_type'])
        pred_depth = outputs['pred_depth']
        gt_depth = targets_on_device['depth']
        depth_mse_metric.update(pred_depth, gt_depth)
        depth_mae_metric.update(pred_depth, gt_depth)

    # --- 5. 平均 ---
    num_batches = max(1, len(val_loader))
    avg_val_loss = total_val_loss / num_batches
    avg_recon_geom_loss = total_recon_geom_loss / num_batches
    avg_recon_app_loss = total_recon_app_loss / num_batches
    avg_independence_loss = total_independence_loss / num_batches

    avg_cka_seg = total_cka_seg / num_batches
    avg_cka_depth = total_cka_depth / num_batches
    avg_cka_scene = total_cka_scene / num_batches

    # --- 6. 任务指标 ---
    final_miou = miou_metric.compute().item()
    final_scene_acc = scene_acc_metric.compute().item()
    final_rmse = torch.sqrt(depth_mse_metric.compute()).item()
    final_mae = depth_mae_metric.compute().item()

    # --- 7. 打印报告 ---
    logging.info("\n--- Validation Results ---")
    logging.info(f"Average Total Loss: {avg_val_loss:.4f}")
    logging.info("\n-- Causal & Reconstruction Losses --")
    logging.info(f"  - Independence Loss (Linear CKA): {avg_independence_loss:.4f}")
    logging.info(f"  - CKA (z_s vs z_p_seg):   {avg_cka_seg:.6f} (越低越好)")
    logging.info(f"  - CKA (z_s vs z_p_depth): {avg_cka_depth:.6f} (越低越好)")
    logging.info(f"  - CKA (z_s vs z_p_scene): {avg_cka_scene:.6f} (越低越好)")
    logging.info(f"  - Geometry Recon Loss: {avg_recon_geom_loss:.4f}  <-- 几何清晰度指标")
    logging.info(f"  - Appearance Recon Loss: {avg_recon_app_loss:.4f}  <-- 外观清晰度指标")
    logging.info("\n-- Downstream Task Metrics --")
    logging.info(f"  - Segmentation (mIoU): {final_miou:.4f}")
    logging.info(f"  - Depth (RMSE): {final_rmse:.4f}")
    logging.info(f"  - Scene Classification (Acc): {final_scene_acc:.4f}")
    logging.info("--------------------------")

    miou_metric.reset()
    scene_acc_metric.reset()
    depth_mse_metric.reset()
    depth_mae_metric.reset()

    return {
        'val_loss': avg_val_loss,
        'recon_geom_loss': avg_recon_geom_loss,
        'recon_app_loss': avg_recon_app_loss,
        'seg_miou': final_miou,
        'depth_rmse': final_rmse,
        'depth_mae': final_mae,
        'scene_acc': final_scene_acc
    }


# ======= 追加：quick diagnose 支撑函数 =======
import numpy as np


@torch.no_grad()
def _depth_rmse(pred, gt):
    pred = pred.float(); gt = gt.float()
    return torch.sqrt(F.mse_loss(pred, gt)).item()


def _seg_ce_loss(logits, target, ignore_index=255):
    # logits: [N,C,H,W], target: [N,H,W]
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


def _edge_mag_2d(x):
    # x: [N,1,H,W] 或 [N,C,H,W]（只用第 1 通道）
    if x.dim() == 4 and x.size(1) > 1:
        x = x[:, :1]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy)


def _boundary_map_from_seg(seg_labels):
    """
    seg_labels: [N, H, W] int/long
    返回 float 边界图 [N,1,H,W]，用 bool 比较生成，避免 bitwise_or 的 dtype 错误。
    """
    assert seg_labels.dim() == 3, "seg_labels should be [N,H,W]"
    N, H, W = seg_labels.shape
    seg = seg_labels
    # 邻域不等即边界（bool）
    b = torch.zeros((N, 1, H, W), dtype=torch.bool, device=seg.device)
    b[:, :, 1:, :]  |= (seg[:, 1:, :]  != seg[:, :-1, :]).unsqueeze(1)
    b[:, :, :-1, :] |= (seg[:, :-1, :] != seg[:, 1:, :]).unsqueeze(1)
    b[:, :, :, 1:]  |= (seg[:, :, 1:]  != seg[:, :, :-1]).unsqueeze(1)
    b[:, :, :, :-1] |= (seg[:, :, :-1] != seg[:, :, 1:]).unsqueeze(1)
    return b.float()


def _influence_ratio(main, alt):
    # 影响力 = ||main - alt|| / (||main|| + 1e-6)
    num = torch.norm(main - alt)
    den = torch.norm(main) + 1e-6
    return (num / den).item()


def _forward_decomposed(model, rgb_tensor):
    """
    注意：不加 no_grad 装饰器，确保后续可做 autograd.grad。
    """
    feature_map   = model.encoder(rgb_tensor)                      # [B,768,14,14]
    f_proj        = model.proj_f(feature_map)                      # [B,256,14,14]
    z_s_map       = model.projector_s(feature_map)                 # [B,ds,14,14]
    zs_proj       = model.proj_z_s(z_s_map)                        # [B,256,14,14]
    zps_map_seg   = model.projector_p_seg(feature_map)
    zps_proj_seg  = model.proj_z_p_seg(zps_map_seg)                # [B,256,14,14]
    zps_map_depth = model.projector_p_depth(feature_map)
    zps_proj_dep  = model.proj_z_p_depth(zps_map_depth)
    return f_proj, zs_proj, zps_proj_seg, zps_proj_dep, z_s_map, zps_map_seg, zps_map_depth


def _safe_grad_norm(loss, tensors, retain_graph=False):
    """
    用 autograd.grad 取梯度范数；张量可能无梯度（返回 None）时，记 0。
    """
    grads = torch.autograd.grad(loss, tensors, retain_graph=retain_graph, allow_unused=True)
    norms = []
    for g in grads:
        if g is None:
            norms.append(0.0)
        else:
            norms.append(g.detach().norm().item())
    return norms


def quick_diagnose(model, val_loader, device, save_dir="runs/_quick_diag"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()  # eval 模式，但允许计算梯度（不进入 no_grad）

    batch = next(iter(val_loader))
    # 标准键位
    rgb   = batch['rgb'][:1].to(device)                # [1,3,H,W]
    depth = batch['depth'][:1].to(device)              # [1,1,H,W]
    seg   = batch['segmentation'][:1].to(device)       # [1,H,W]

    # 1) 分解前向 & 三种变体（需要可追踪图）
    f_proj, zs_proj, zp_seg_proj, zp_dep_proj, z_s_map, z_p_seg_map, z_p_dep_map = _forward_decomposed(model, rgb)
    zeros = torch.zeros_like(zs_proj)

    # Depth
    pred_depth_main = model.predictor_depth(torch.cat([f_proj, zs_proj], 1), zp_dep_proj)
    pred_depth_zs   = model.predictor_depth(torch.cat([f_proj, zs_proj], 1), torch.zeros_like(zp_dep_proj))
    pred_depth_zp   = model.predictor_depth(torch.cat([f_proj, zeros],   1), zp_dep_proj)

    # Seg
    pred_seg_main = model.predictor_seg(torch.cat([f_proj, zs_proj], 1), zp_seg_proj)
    pred_seg_zs   = model.predictor_seg(torch.cat([f_proj, zs_proj], 1), torch.zeros_like(zp_seg_proj))
    pred_seg_zp   = model.predictor_seg(torch.cat([f_proj, zeros],   1), zp_seg_proj)

    # 影响力
    depth_inf_zs = _influence_ratio(pred_depth_main, pred_depth_zp)
    depth_inf_zp = _influence_ratio(pred_depth_main, pred_depth_zs)

    seg_prob_main = F.softmax(pred_seg_main, dim=1)
    seg_prob_zs   = F.softmax(pred_seg_zs,   dim=1)
    seg_prob_zp   = F.softmax(pred_seg_zp,   dim=1)
    seg_inf_zs = _influence_ratio(seg_prob_main, seg_prob_zs)
    seg_inf_zp = _influence_ratio(seg_prob_main, seg_prob_zp)

    # 单 batch 粗评估
    rmse_main = _depth_rmse(pred_depth_main, depth)
    rmse_zs   = _depth_rmse(pred_depth_zs,   depth)
    rmse_zp   = _depth_rmse(pred_depth_zp,   depth)

    pred_seg_label = pred_seg_main.argmax(1)  # [1,H,W]
    # 简单像素精度（忽略255）
    valid = (seg != 255)
    inter = ((pred_seg_label == seg) & valid).sum().item()
    union = valid.sum().item()
    pix_acc = inter / max(1, union)

    # seg 边界 vs recon_geom 边缘
    recon_geom_final, _ = model.decoder_geom(z_s_map)      # [1,1,H,W]
    geom_edge = _edge_mag_2d(recon_geom_final)             # [1,1,H,W]
    seg_edge  = _boundary_map_from_seg(pred_seg_label)     # [1,1,H,W]
    ge = geom_edge / (geom_edge.max() + 1e-6)
    se = seg_edge  / (seg_edge.max()  + 1e-6)
    edge_corr = F.cosine_similarity(ge.view(1, -1), se.view(1, -1)).item()

    # 2) 梯度探针（autograd.grad；无梯度视为0）
    loss_seg = _seg_ce_loss(pred_seg_main, seg)  # target 仍是 [1,H,W]
    g_zs_seg, g_zpseg_seg = _safe_grad_norm(loss_seg, [z_s_map, z_p_seg_map], retain_graph=True)

    loss_dep = F.mse_loss(pred_depth_main, depth)
    g_zs_dep, g_zpdep_dep = _safe_grad_norm(loss_dep, [z_s_map, z_p_dep_map], retain_graph=False)

    # 3) 打印诊断表
    print("\n==== QUICK DIAG ====")
    print(f"[Depth] RMSE(main)={rmse_main:.4f}  RMSE(zs)={rmse_zs:.4f}  RMSE(zp)={rmse_zp:.4f}")
    print(f"[Depth] Influence: zs={depth_inf_zs:.3f}  zp={depth_inf_zp:.3f}  (大→依赖该分支)")
    print(f"[Seg  ] PixelAcc(main)={pix_acc:.3f}")
    print(f"[Seg  ] Influence: zs={seg_inf_zs:.3f}  zp={seg_inf_zp:.3f}")
    print(f"[Edge ] seg–geom edge cosine={edge_corr:.3f}  (大→分割边缘受几何引导)")
    print(f"[Grad ] ||dL_seg/dz_s||={g_zs_seg:.3e}  ||dL_seg/dz_p_seg||={g_zpseg_seg:.3e}")
    print(f"[Grad ] ||dL_dep/dz_s||={g_zs_dep:.3e}  ||dL_dep/dz_p_dep||={g_zpdep_dep:.3e}")
    print("Tips: 若某分支 influence<0.10 或梯度范数≈0，说明该分支几乎没被用（可能被权重/门控压住或接线问题）。")
    print("====================\n")

    return {
        "depth_rmse_main": rmse_main, "depth_rmse_zs": rmse_zs, "depth_rmse_zp": rmse_zp,
        "depth_inf_zs": depth_inf_zs, "depth_inf_zp": depth_inf_zp,
        "seg_pix_acc": pix_acc, "seg_inf_zs": seg_inf_zs, "seg_inf_zp": seg_inf_zp,
        "edge_cosine": edge_corr,
        "g_zs_seg": g_zs_seg, "g_zpseg_seg": g_zpseg_seg,
        "g_zs_dep": g_zs_dep, "g_zpdep_dep": g_zpdep_dep,
    }
