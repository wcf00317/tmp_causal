# 文件: engine/evaluator.py
# 版本：【已升级，增加可视化指标监控】

import torch
import torchmetrics
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """
    【已升级】评估模型，并明确返回所有任务及重构任务的指标。
    """
    model.eval()

    # --- 1. 初始化所有指标对象 (保持不变) ---
    num_seg_classes = model.predictor_seg.output_channels
    num_scene_classes = model.predictor_scene.out_features

    miou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=num_seg_classes, ignore_index=255).to(device)
    scene_acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_scene_classes).to(device)
    depth_mse_metric = torchmetrics.regression.MeanSquaredError().to(device)
    depth_mae_metric = torchmetrics.regression.MeanAbsoluteError().to(device)

    # --- 2. 【核心修改】: 新增用于跟踪重构损失的累加器 ---
    total_val_loss = 0.0
    total_recon_geom_loss = 0.0
    total_recon_app_loss = 0.0
    total_independence_loss = 0.0
    # ----------------------------------------------------

    pbar = tqdm(val_loader, desc="Evaluating", leave=False)

    for batch in pbar:
        rgb = batch['rgb'].to(device)
        targets_on_device = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        outputs = model(rgb)

        crit_out = criterion(outputs, targets_on_device)
        if isinstance(crit_out, tuple) or isinstance(crit_out, list):
            _, loss_dict = crit_out
        else: # (为了兼容旧版，虽然现在统一返回tuple)
            loss_dict = crit_out

        # --- 3. 【核心修改】: 从loss_dict中提取并累加所有我们关心的损失 ---
        total_val_loss += loss_dict.get('total_loss', torch.tensor(0.0)).item()
        total_recon_geom_loss += loss_dict.get('recon_geom_loss', torch.tensor(0.0)).item()
        total_recon_app_loss += loss_dict.get('recon_app_loss', torch.tensor(0.0)).item()
        total_independence_loss += loss_dict.get('independence_loss', torch.tensor(0.0)).item()
        # --------------------------------------------------------------------

        # --- 4. 更新下游任务指标 (保持不变) ---
        miou_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])
        scene_acc_metric.update(outputs['pred_scene'], targets_on_device['scene_type'])
        pred_depth = outputs['pred_depth']
        gt_depth = targets_on_device['depth']
        depth_mse_metric.update(pred_depth, gt_depth)
        depth_mae_metric.update(pred_depth, gt_depth)

    # --- 5. 【核心修改】: 计算所有损失的平均值 ---
    num_batches = len(val_loader)
    avg_val_loss = total_val_loss / num_batches
    avg_recon_geom_loss = total_recon_geom_loss / num_batches
    avg_recon_app_loss = total_recon_app_loss / num_batches
    avg_independence_loss = total_independence_loss / num_batches
    # -------------------------------------------

    # --- 6. 计算下游任务最终指标 (保持不变) ---
    final_miou = miou_metric.compute().item()
    final_scene_acc = scene_acc_metric.compute().item()
    final_rmse = torch.sqrt(depth_mse_metric.compute()).item()
    final_mae = depth_mae_metric.compute().item()

    # --- 7. 【核心修改】: 打印一个更全面的报告 ---
    print("\n--- Validation Results ---")
    print(f"Average Total Loss: {avg_val_loss:.4f}")
    print("\n-- Causal & Reconstruction Losses --")
    print(f"  - Independence Loss (HSIC): {avg_independence_loss:.4f}")
    print(f"  - Geometry Recon Loss: {avg_recon_geom_loss:.4f}  <-- 几何清晰度指标")
    print(f"  - Appearance Recon Loss: {avg_recon_app_loss:.4f}  <-- 外观清晰度指标")
    print("\n-- Downstream Task Metrics --")
    print(f"  - Segmentation (mIoU): {final_miou:.4f}")
    print(f"  - Depth (RMSE): {final_rmse:.4f}")
    print(f"  - Scene Classification (Acc): {final_scene_acc:.4f}")
    print("--------------------------")
    # ---------------------------------------------

    miou_metric.reset()
    scene_acc_metric.reset()
    depth_mse_metric.reset()
    depth_mae_metric.reset()

    # --- 8. 【核心修改】: 返回包含新指标的字典 ---
    return {
        'val_loss': avg_val_loss,
        'recon_geom_loss': avg_recon_geom_loss,
        'recon_app_loss': avg_recon_app_loss,
        'seg_miou': final_miou,
        'depth_rmse': final_rmse,
        'depth_mae': final_mae,
        'scene_acc': final_scene_acc
    }