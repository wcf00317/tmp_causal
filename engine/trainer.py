import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import os, logging
from .evaluator import evaluate
from utils.general_utils import save_checkpoint


# ----------------------------
# utils
# ----------------------------
def _set_requires_grad(module, requires_grad: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad

def _switch_stage_freeze(model, stage: int):
    """
    stage=1: 冻结 z_p 相关分支（私有投影/残差），只训练 z_s 与主干。
    stage=2: 全部解冻。
    """
    if stage == 1:
        _set_requires_grad(model.projector_p_seg,      False)
        _set_requires_grad(model.projector_p_depth,    False)
        _set_requires_grad(model.proj_z_p_seg,         False)
        _set_requires_grad(model.proj_z_p_depth,       False)
        _set_requires_grad(model.zp_seg_refiner,       False)
        _set_requires_grad(model.zp_depth_refiner,     False)
        _set_requires_grad(model.decoder_zp_depth,     False)
        logging.info("Stage-1: frozen private (z_p) branches.")
    else:
        _set_requires_grad(model.projector_p_seg,      True)
        _set_requires_grad(model.projector_p_depth,    True)
        _set_requires_grad(model.proj_z_p_seg,         True)
        _set_requires_grad(model.proj_z_p_depth,       True)
        _set_requires_grad(model.zp_seg_refiner,       True)
        _set_requires_grad(model.zp_depth_refiner,     True)
        _set_requires_grad(model.decoder_zp_depth,     True)
        logging.info("Stage-2: unfrozen private (z_p) branches.")


def _get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get("lr", None)

def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def _build_scheduler(optimizer, train_cfg):
    """
    读取 config['training'] 下的 lr_scheduler 配置：
      lr_scheduler:
        type: "cosine" | "step"          # 默认 cosine
        warmup_epochs: 3                 # 仅对 cosine 生效（线性 warm-up）
        min_lr_factor: 0.1               # eta_min = base_lr * min_lr_factor
        T_max: <int>                     # 可选；默认 = epochs - warmup_epochs
        step_size: 10                    # 仅对 step 生效
        gamma: 0.5                       # 仅对 step 生效
    """
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    sched_cfg = train_cfg.get("lr_scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "cosine")).lower()

    if sched_type == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 3))
        min_lr_factor = float(sched_cfg.get("min_lr_factor", 0.1))
        total_epochs  = int(train_cfg.get("epochs", 30))
        # 余弦部分的 T_max（排除 warmup）
        t_max = int(sched_cfg.get("T_max", max(1, total_epochs - warmup_epochs)))
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=base_lr * min_lr_factor
        )
        return {
            "type": "cosine",
            "warmup_epochs": warmup_epochs,
            "base_lr": base_lr,
            "cosine": cosine
        }

    # fallback: StepLR
    step_size = int(sched_cfg.get("step_size", 10))
    gamma     = float(sched_cfg.get("gamma", 0.5))
    step = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return {
        "type": "step",
        "step": step
    }


# ----------------------------
# train loops
# ----------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage: int):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False)

    # 物理 BS=4，累积 8次 = 逻辑 BS 32
    accumulation_steps = 8

    # 1. 在循环外先清空一次梯度
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(pbar):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rgb = batch['rgb']

        # ❌ 删除这行！不要在每次微步都清空！
        # optimizer.zero_grad(set_to_none=True)

        outputs = model(rgb, stage=stage)

        crit_out = criterion(outputs, batch)
        if isinstance(crit_out, (tuple, list)):
            total_loss, loss_dict = crit_out[0], crit_out[1]
        elif isinstance(crit_out, dict):
            loss_dict = crit_out
            total_loss = loss_dict.get('total_loss')
            if total_loss is None:
                raise ValueError("criterion returned dict but no 'total_loss' key found.")
        else:
            raise ValueError("criterion must return dict or (total_loss, dict).")

        # Loss 归一化，使得 8 次累加后的梯度幅值等效于一次大 Batch
        loss_normalized = total_loss / accumulation_steps
        loss_normalized.backward()

        # --- 梯度累积逻辑 ---
        # 只有达到累积步数时，才更新参数并清空梯度
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) # 更新完才清空！

        total_train_loss += float(total_loss.item())
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    # --- 【新增】处理循环结束时剩下的“尾巴” ---
    # 如果总 batch 数不是 8 的倍数，最后积累的梯度也需要更新一次
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = total_train_loss / max(1, len(train_loader))
    logging.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device, checkpoint_dir='checkpoints'):
    """
    注意：为了更好地控制 Cosine + warm-up，这里会忽略传入的 `scheduler`，
    统一根据 config['training']['lr_scheduler'] 在本函数内部创建与推进调度器。
    这样你无需修改 main.py。
    """
    best_val_metric = float('inf')
    train_cfg = config['training']
    stage1_epochs = int(train_cfg.get('stage1_epochs', 10))

    # 构建调度器（默认 cosine + 线性 warmup）
    sched = _build_scheduler(optimizer, train_cfg)
    logging.info(f"[LR Scheduler] {sched['type']}; base_lr={train_cfg.get('learning_rate', 1e-4)}")

    total_epochs = int(train_cfg.get('epochs', 30))
    base_lr = float(train_cfg.get("learning_rate", 1e-4))

    for epoch in range(total_epochs):
        stage = 1 if epoch < stage1_epochs else 2
        if epoch == 0 or epoch == stage1_epochs:
            _switch_stage_freeze(model, stage)

        # ---- Warm-up（仅 cosine 时生效）----
        if sched["type"] == "cosine":
            warmup_epochs = sched["warmup_epochs"]
            if epoch < warmup_epochs:
                # 线性从 10%*base_lr -> base_lr
                warmup_start = 0.1 * base_lr
                ratio = float(epoch + 1) / float(max(1, warmup_epochs))
                lr_now = warmup_start + (base_lr - warmup_start) * ratio
                _set_lr(optimizer, lr_now)
            else:
                # 确保进入余弦阶段前把 lr 设回 base_lr（第一步会立刻被 cosine 调度）
                if abs(_get_lr(optimizer) - base_lr) > 1e-12 and epoch == warmup_epochs:
                    _set_lr(optimizer, base_lr)

        cur_lr = _get_lr(optimizer)
        logging.info(f"\n----- Starting Epoch {epoch + 1}/{total_epochs} (Stage {stage}) | lr={cur_lr:.6f} -----")

        # --- Train ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage=stage)

        # --- Validate（验证阶段一律 stage=2）---
        val_metrics = evaluate(model, val_loader, criterion, device, stage=2)

        # Quick diagnose（按需）
        if os.environ.get("QUICK_DIAG", "0") == "1" and (epoch == 0 or epoch == stage1_epochs):
            from engine.evaluator import quick_diagnose
            quick_diagnose(model, val_loader, device)

        # --- Step Scheduler ---
        if sched["type"] == "cosine":
            warmup_epochs = sched["warmup_epochs"]
            if epoch >= warmup_epochs:
                sched["cosine"].step()
        else:
            # StepLR
            sched["step"].step()

        # --- Checkpoint ---
        is_best = val_metrics['depth_rmse'] < best_val_metric
        if is_best:
            best_val_metric = val_metrics['depth_rmse']
            logging.info(f"  -> New best model found with Depth RMSE: {best_val_metric:.4f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_metric': best_val_metric,
        }, is_best, checkpoint_dir=checkpoint_dir)

    logging.info("\n----- Training Finished -----")
    logging.info(f"Best model saved with Depth RMSE: {best_val_metric:.4f}")
