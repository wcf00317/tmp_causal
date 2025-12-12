# 文件: engine/evaluator.py
# 版本：【LibMTL NYUv2 Alignment Version - Fixed Ignore Index (-1)】

import os
import logging
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, stage, data_type):
    """
    评估模型，并返回所有任务及重构任务的指标。
    已移除 Scene，新增 Normal CKA 及基于 data_type 的日志控制。
    """
    model.eval()

    # --- 1. 初始化所有指标对象 ---
    num_seg_classes = model.predictor_seg.output_channels

    # [FIXED] 将 ignore_index 从 255 改为 -1，以匹配 LibMTL 数据格式
    miou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=num_seg_classes, ignore_index=-1).to(device)

    pixel_acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_seg_classes, average='micro', ignore_index=-1).to(device)

    # [MODIFIED]: 使用自定义的 LibMTL 指标
    depth_metric = DepthMetric().to(device)
    normal_metric = NormalMetric().to(device)

    # --- 2. 跟踪损失 ---
    total_val_loss = 0.0
    total_recon_geom_loss = 0.0
    total_recon_app_loss = 0.0
    total_independence_loss = 0.0

    total_cka_seg = 0.0
    total_cka_depth = 0.0
    total_cka_normal = 0.0  # [NEW] 追踪 Normal 分支的 CKA

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

        # 累加基础 Loss
        total_val_loss += loss_dict.get('total_loss', torch.tensor(0.0)).item()
        total_recon_geom_loss += loss_dict.get('recon_geom_loss', torch.tensor(0.0)).item()
        total_recon_app_loss += loss_dict.get('recon_app_loss', torch.tensor(0.0)).item()
        total_independence_loss += loss_dict.get('independence_loss', torch.tensor(0.0)).item()

        # 累加 CKA Loss
        total_cka_seg += loss_dict.get('cka_seg', torch.tensor(0.0)).item()
        total_cka_depth += loss_dict.get('cka_depth', torch.tensor(0.0)).item()
        total_cka_normal += loss_dict.get('cka_normal', torch.tensor(0.0)).item()  # [NEW]

        # 更新任务指标
        miou_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])
        pixel_acc_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])

        # [NEW]: 法线指标更新
        if 'normal' in targets_on_device and 'normals' in outputs:
            normal_metric.update(outputs['normals'], targets_on_device['normal'])

        # [MODIFIED]: 深度指标更新
        if 'depth' in targets_on_device:
            depth_metric.update(outputs['pred_depth'], targets_on_device['depth'])

    # --- 5. 平均 ---
    num_batches = max(1, len(val_loader))
    avg_val_loss = total_val_loss / num_batches
    avg_recon_geom_loss = total_recon_geom_loss / num_batches
    avg_recon_app_loss = total_recon_app_loss / num_batches
    avg_independence_loss = total_independence_loss / num_batches

    avg_cka_seg = total_cka_seg / num_batches
    avg_cka_depth = total_cka_depth / num_batches
    avg_cka_normal = total_cka_normal / num_batches  # [NEW]

    # --- 6. 任务指标计算 ---
    final_miou = miou_metric.compute().item()
    final_pixel_acc = pixel_acc_metric.compute().item()

    # [MODIFIED]: 深度指标
    final_abs_err, final_rel_err = depth_metric.compute()

    # [NEW]: 法线指标 (安全计算，防空)
    if len(normal_metric.record) > 0:
        mean_angle, median_angle, acc_11, acc_22, acc_30 = normal_metric.compute()
    else:
        mean_angle, median_angle, acc_11, acc_22, acc_30 = 0.0, 0.0, 0.0, 0.0, 0.0

    # --- 7. 打印报告 (Log) ---
    logging.info("\n--- Validation Results ---")

    # 打印 Loss 概览
    log_loss = (f"Avg Loss: {avg_val_loss:.4f} | "
                f"Indep(CKA): {avg_independence_loss:.4f} | "
                f"Recon: G={avg_recon_geom_loss:.4f}, A={avg_recon_app_loss:.4f}")
    logging.info(log_loss)

    # 打印 CKA 详情 (包含 Normal)
    logging.info(f" - CKA (z_s vs z_p_seg): {avg_cka_seg:.6f} (越低越好)")
    logging.info(f" - CKA (z_s vs z_p_depth): {avg_cka_depth:.6f} (越低越好)")
    logging.info(f" - CKA (z_s vs z_p_normal): {avg_cka_normal:.6f} (越低越好)")

    logging.info("-- Downstream Task Metrics --")

    # 1. 基础任务 (Seg/Depth 总是打印)
    task_str = f"  - Seg:   mIoU={final_miou:.4f}, Pixel Acc={final_pixel_acc:.4f}\n"
    if 'gta5' not in str(data_type).lower():
        task_str += f"  - Depth: Abs Err={final_abs_err:.4f}, Rel Err={final_rel_err:.4f}"

    # 2. 法线 (Normal) - 根据 data_type 决定是否打印
    if 'nyuv2' in str(data_type).lower():
        task_str += (f"\n  - Normal: Mean={mean_angle:.2f}°, Med={median_angle:.2f}° | "
                     f"Acc: 11°={acc_11:.3f}, 22°={acc_22:.3f}, 30°={acc_30:.3f}")

    logging.info(task_str)
    logging.info("--------------------------")

    # Reset metrics
    miou_metric.reset()
    pixel_acc_metric.reset()
    depth_metric.reset()
    normal_metric.reset()

    # [CORRECTED RETURN]: 返回所有指标 (移除 Scene，包含 Normal)
    return {
        'val_loss': avg_val_loss,
        'recon_geom_loss': avg_recon_geom_loss,
        'recon_app_loss': avg_recon_app_loss,
        'seg_miou': final_miou,
        'seg_pixel_acc': final_pixel_acc,
        'depth_abs_err': final_abs_err,
        'depth_rel_err': final_rel_err,
        'normal_mean_angle': mean_angle,
        'normal_median_angle': median_angle,
        'normal_acc_11': acc_11,
        'normal_acc_22': acc_22,
        'normal_acc_30': acc_30
    }
# ======= 辅助函数 (QUICK DIAGNOSE) =======

@torch.no_grad()
def _depth_rmse(pred, gt):
    pred = pred.float();
    gt = gt.float()
    return torch.sqrt(F.mse_loss(pred, gt)).item()


def _seg_ce_loss(logits, target, ignore_index=255):
    # logits: [N,C,H,W], target: [N,H,W]
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


def _edge_mag_2d(x):
    # x: [N,1,H,W] 或 [N,C,H,W]（只用第 1 通道）
    if x.dim() == 4 and x.size(1) > 1:
        x = x[:, :1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
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
    b[:, :, 1:, :] |= (seg[:, 1:, :] != seg[:, :-1, :]).unsqueeze(1)
    b[:, :, :-1, :] |= (seg[:, :-1, :] != seg[:, 1:, :]).unsqueeze(1)
    b[:, :, :, 1:] |= (seg[:, :, 1:] != seg[:, :, :-1]).unsqueeze(1)
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
    # [MODIFIED] 使用 extract_features 接口
    combined_feat, _, _ = model.extract_features(rgb_tensor)

    # 投影
    f_proj = model.proj_f(combined_feat)
    z_s_map = model.projector_s(combined_feat)
    zs_proj = model.proj_z_s(z_s_map)
    zps_map_seg = model.projector_p_seg(combined_feat)
    zps_proj_seg = model.proj_z_p_seg(zps_map_seg)
    zps_map_depth = model.projector_p_depth(combined_feat)
    zps_proj_dep = model.proj_z_p_depth(zps_map_depth)
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
    rgb = batch['rgb'][:1].to(device)  # [1,3,H,W]
    depth = batch['depth'][:1].to(device)  # [1,1,H,W]
    seg = batch['segmentation'][:1].to(device)  # [1,H,W]

    # 1) 分解前向 & 三种变体（需要可追踪图）
    f_proj, zs_proj, zp_seg_proj, zp_dep_proj, z_s_map, z_p_seg_map, z_p_dep_map = _forward_decomposed(model, rgb)
    zeros = torch.zeros_like(zs_proj)

    # Depth
    pred_depth_main = model.predictor_depth(torch.cat([f_proj, zs_proj], 1), zp_dep_proj)
    pred_depth_zs = model.predictor_depth(torch.cat([f_proj, zs_proj], 1), torch.zeros_like(zp_dep_proj))
    pred_depth_zp = model.predictor_depth(torch.cat([f_proj, zeros], 1), zp_dep_proj)

    # Seg
    pred_seg_main = model.predictor_seg(torch.cat([f_proj, zs_proj], 1), zp_seg_proj)
    pred_seg_zs = model.predictor_seg(torch.cat([f_proj, zs_proj], 1), torch.zeros_like(zp_seg_proj))
    pred_seg_zp = model.predictor_seg(torch.cat([f_proj, zeros], 1), zp_seg_proj)

    # 影响力
    depth_inf_zs = _influence_ratio(pred_depth_main, pred_depth_zp)
    depth_inf_zp = _influence_ratio(pred_depth_main, pred_depth_zs)

    seg_prob_main = F.softmax(pred_seg_main, dim=1)
    seg_prob_zs = F.softmax(pred_seg_zs, dim=1)
    seg_prob_zp = F.softmax(pred_seg_zp, dim=1)
    seg_inf_zs = _influence_ratio(seg_prob_main, seg_prob_zs)
    seg_inf_zp = _influence_ratio(seg_prob_main, seg_prob_zp)

    # 单 batch 粗评估
    rmse_main = _depth_rmse(pred_depth_main, depth)
    rmse_zs = _depth_rmse(pred_depth_zs, depth)
    rmse_zp = _depth_rmse(pred_depth_zp, depth)

    pred_seg_label = pred_seg_main.argmax(1)  # [1,H,W]
    # 简单像素精度（忽略255）
    valid = (seg != 255)
    inter = ((pred_seg_label == seg) & valid).sum().item()
    union = valid.sum().item()
    pix_acc = inter / max(1, union)

    # seg 边界 vs recon_geom 边缘
    recon_geom_final, _ = model.decoder_geom(z_s_map)  # [1,1,H,W]
    geom_edge = _edge_mag_2d(recon_geom_final)  # [1,1,H,W]
    seg_edge = _boundary_map_from_seg(pred_seg_label)  # [1,1,H,W]
    ge = geom_edge / (geom_edge.max() + 1e-6)
    se = seg_edge / (seg_edge.max() + 1e-6)
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


# ==========================================================
# [NEW] LibMTL Aligned Metric Classes (for Depth & Normal)
# ==========================================================

class AbsMetric(object):
    """LibMTL AbsMetric 抽象基类，适配我们的 update/compute/reset 流程。"""

    def __init__(self):
        self.bs = []

    def update(self, *args):
        self.update_fun(*args)

    def compute(self):
        return self.score_fun()

    # [FIXED] 新增 to 方法，返回 self 以兼容 .to(device) 调用
    def to(self, device):
        return self

    def reinit(self):
        self.bs = []
        if hasattr(self, 'abs_record'): self.abs_record = []
        if hasattr(self, 'rel_record'): self.rel_record = []
        if hasattr(self, 'record'): self.record = []

    def reset(self):
        self.reinit()


class DepthMetric(AbsMetric):
    """
    对齐 LibMTL 的 DepthMetric，计算 Abs Err (MAE) 和 Rel Err。
    """

    def __init__(self):
        super(DepthMetric, self).__init__()
        self.abs_record = []
        self.rel_record = []
        self.bs = []

    def update_fun(self, pred, gt):
        # pred, gt 形状应为 [B, C, H, W]
        device = pred.device
        # 掩码: 过滤 GT 中全为 0 的像素
        binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(device)

        num_valid_pixels = torch.nonzero(binary_mask, as_tuple=False).size(0)
        if num_valid_pixels == 0:
            return

        pred = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)

        abs_err = torch.abs(pred - gt)
        rel_err = torch.abs(pred - gt) / torch.clamp(gt, min=1e-6)

        abs_err_mean = (torch.sum(abs_err) / num_valid_pixels).item()
        rel_err_mean = (torch.sum(rel_err) / num_valid_pixels).item()

        self.abs_record.append(abs_err_mean)
        self.rel_record.append(rel_err_mean)
        self.bs.append(num_valid_pixels)

    def score_fun(self):
        if not self.bs:
            return [0.0, 0.0]

        records = np.stack([np.array(self.abs_record), np.array(self.rel_record)])
        batch_size = np.array(self.bs)

        total_pixels = sum(batch_size)
        if total_pixels == 0:
            return [0.0, 0.0]

        # 计算加权平均 (误差 * 像素数 / 总像素数)
        weighted_abs_err = (records[0] * batch_size).sum() / total_pixels
        weighted_rel_err = (records[1] * batch_size).sum() / total_pixels

        return [float(weighted_abs_err), float(weighted_rel_err)]


class NormalMetric(AbsMetric):
    """
    对齐 LibMTL 的 NormalMetric，计算角度误差指标。
    """

    def __init__(self):
        super(NormalMetric, self).__init__()
        self.record = []  # 记录所有有效像素的角度误差 (度)

    def update_fun(self, pred, gt):
        # pred, gt 形状应为 [B, 3, H, W]

        # 1. 法线归一化 (pred)
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)

        # 2. 掩码
        binary_mask = (torch.sum(gt, dim=1) != 0)

        # 3. 计算点积 (cos(theta))
        # gt 已经是归一化的（根据 LibMTL 文档）
        dot_product = torch.sum(pred * gt, 1).masked_select(binary_mask)

        # 4. 角度误差 (acos(dot_product))
        error_rad = torch.acos(torch.clamp(dot_product, -1, 1))

        # 转换为角度 (度)
        error_deg = torch.rad2deg(error_rad).detach().cpu().numpy()

        self.record.append(error_deg)

    def score_fun(self):
        if not self.record:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        records = np.concatenate(self.record)

        # 5. 计算指标
        mean_angle = np.mean(records)
        median_angle = np.median(records)

        # 准确率 (Acc@T)
        acc_11 = np.mean((records < 11.25) * 1.0)
        acc_22 = np.mean((records < 22.5) * 1.0)
        acc_30 = np.mean((records < 30) * 1.0)

        return [float(mean_angle), float(median_angle), float(acc_11), float(acc_22), float(acc_30)]