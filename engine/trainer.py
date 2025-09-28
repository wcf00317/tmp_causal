import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
from .evaluator import evaluate
from utils.general_utils import save_checkpoint


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False)

    for batch in pbar:
        rgb = batch['rgb'].to(device)
        targets_on_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(rgb)

        # Support criterion that returns either dict or (total_loss, dict)
        crit_out = criterion(outputs, targets_on_device)
        if isinstance(crit_out, tuple) or isinstance(crit_out, list):
            total_loss, loss_dict = crit_out[0], crit_out[1]
        elif isinstance(crit_out, dict):
            loss_dict = crit_out
            total_loss = loss_dict.get('total_loss')
            if total_loss is None:
                raise ValueError("criterion returned dict but no 'total_loss' key found.")
        else:
            raise ValueError("criterion must return dict or (total_loss, dict).")

        total_loss.backward()
        optimizer.step()

        total_train_loss += float(total_loss.item())
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device):
    best_val_metric = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n----- Starting Epoch {epoch + 1}/{config['training']['epochs']} -----")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # --- Scheduler Advice ---
        # If using StepLR (current):
        scheduler.step()
        # If you switch to ReduceLROnPlateau in the config, change the call to:
        # scheduler.step(val_metrics['val_loss'])

        is_best = val_metrics['depth_rmse'] < best_val_metric
        if is_best:
            best_val_metric = val_metrics['depth_rmse']
            print(f"  -> New best model found with Depth RMSE: {best_val_metric:.4f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_metric': best_val_metric,
        }, is_best)

    print("\n----- Training Finished -----")
    print(f"Best model saved with Depth RMSE: {best_val_metric:.4f}")