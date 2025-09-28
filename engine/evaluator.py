import torch
import torchmetrics
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set using standard metrics for all three tasks.
    - Segmentation: Mean Intersection over Union (mIoU)
    - Depth: RMSE and MAE
    - Scene Classification: Accuracy
    """
    model.eval()  # Set the model to evaluation mode

    # --- 1. Initialize All Metric Objects ---
    num_seg_classes = model.predictor_seg.output_channels
    num_scene_classes = model.predictor_scene.out_features

    miou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=num_seg_classes, ignore_index=255).to(device)
    scene_acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_scene_classes).to(device)
    depth_mse_metric = torchmetrics.regression.MeanSquaredError().to(device)
    depth_mae_metric = torchmetrics.regression.MeanAbsoluteError().to(device)  # 恢复MAE

    total_val_loss = 0.0

    pbar = tqdm(val_loader, desc="Evaluating", leave=False)

    for batch in pbar:
        # --- 2. Move data and perform forward pass ---
        rgb = batch['rgb'].to(device)
        targets_on_device = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        outputs = model(rgb)

        # --- 3. Calculate Loss (with robust handling) ---
        crit_out = criterion(outputs, targets_on_device)
        if isinstance(crit_out, tuple) or isinstance(crit_out, list):
            _, loss_dict = crit_out
        elif isinstance(crit_out, dict):
            loss_dict = crit_out
        else:
            raise ValueError("criterion must return dict or (total_loss, dict).")

        batch_loss = loss_dict.get('total_loss')
        if batch_loss is None:
            raise ValueError("Loss dictionary must contain a 'total_loss' key.")
        total_val_loss += batch_loss.item()

        # --- 4. Update All Metrics ---
        # Segmentation
        miou_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])

        # Scene Classification
        scene_acc_metric.update(outputs['pred_scene'], targets_on_device['scene_type'])

        # Depth
        pred_depth = outputs['pred_depth']
        gt_depth = targets_on_device['depth']
        depth_mse_metric.update(pred_depth, gt_depth)
        depth_mae_metric.update(pred_depth, gt_depth)

    # --- 5. Compute Final Metrics ---
    avg_val_loss = total_val_loss / len(val_loader)
    final_miou = miou_metric.compute().item()
    final_scene_acc = scene_acc_metric.compute().item()
    final_rmse = torch.sqrt(depth_mse_metric.compute()).item()
    final_mae = depth_mae_metric.compute().item()  # 恢复MAE

    # --- Print a comprehensive report ---
    print("\n--- Validation Results ---")
    print(f"Average Loss: {avg_val_loss:.4f}")
    print("\n-- Segmentation --")
    print(f"  - Mean IoU (mIoU): {final_miou:.4f}")
    print("\n-- Depth Estimation --")
    print(f"  - Root Mean Squared Error (RMSE): {final_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {final_mae:.4f}")  # 恢复MAE
    print("\n-- Scene Classification --")
    print(f"  - Accuracy: {final_scene_acc:.4f}")
    print("--------------------------")

    # Reset all metrics for the next evaluation
    miou_metric.reset()
    scene_acc_metric.reset()
    depth_mse_metric.reset()
    depth_mae_metric.reset()

    return {
        'val_loss': avg_val_loss,
        'seg_miou': final_miou,
        'depth_rmse': final_rmse,
        'depth_mae': final_mae,  # 恢复MAE
        'scene_acc': final_scene_acc
    }