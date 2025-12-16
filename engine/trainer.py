import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import os, logging
import numpy as np  # 必需导入
from .evaluator import evaluate
from utils.general_utils import save_checkpoint
from torch.cuda.amp import autocast, GradScaler


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
    # 检查模型是否有这些属性（兼容 RawMTL）
    if not hasattr(model, 'projector_p_seg') or model.projector_p_seg is None:
        return

    def _switch_stage_freeze(model, stage: int):
        # 基础检查，防止传入不兼容的模型
        if not hasattr(model, 'projector_p_seg') or model.projector_p_seg is None:
            return

        # === Stage 0: 分解预热 (Decomposition Warmup) ===
        if stage == 0:
            # 1. 冻结下游任务头 (Seg, Depth, Normal)
            #    Stage 0 只想训练 Encoder 和 分解头(Albedo/Normal/Light)，
            #    防止随机初始化的任务头回传干扰梯度。
            _set_requires_grad(model.predictor_seg, False)
            _set_requires_grad(model.predictor_depth, False)
            # 使用 getattr 兼容可能没有 Normal 任务的旧 Config，但您确认有 Normal
            if hasattr(model, 'predictor_normal'):
                _set_requires_grad(model.predictor_normal, False)

            # 2. 冻结 z_p 私有分支 (同 Stage 1)
            #    Stage 0 专注于 z_s 的几何结构（通过 Normal Head 监督）
            _set_requires_grad(model.projector_p_seg, False)
            _set_requires_grad(model.projector_p_depth, False)
            _set_requires_grad(getattr(model, 'projector_p_normal', None), False)

            _set_requires_grad(model.proj_z_p_seg, False)
            _set_requires_grad(model.proj_z_p_depth, False)
            _set_requires_grad(getattr(model, 'proj_z_p_normal', None), False)

            _set_requires_grad(model.zp_seg_refiner, False)
            _set_requires_grad(model.zp_depth_refiner, False)
            _set_requires_grad(getattr(model, 'zp_normal_refiner', None), False)

            _set_requires_grad(model.decoder_zp_depth, False)
            _set_requires_grad(getattr(model, 'decoder_zp_normal', None), False)

            logging.info("Stage-0: Decomposition Warmup. Frozen Task Heads & z_p branches.")

        # === Stage 1: 结构预热 (Structure Warmup) ===
        elif stage == 1:
            # 1. [关键修改] 显式解冻任务头
            #    因为它们在 Stage 0 被冻结了，必须在这里解开！
            _set_requires_grad(model.predictor_seg, True)
            _set_requires_grad(model.predictor_depth, True)
            if hasattr(model, 'predictor_normal'):
                _set_requires_grad(model.predictor_normal, True)

            # 2. 继续冻结 z_p 私有分支 (保持原样)
            _set_requires_grad(model.projector_p_seg, False)
            _set_requires_grad(model.projector_p_depth, False)
            _set_requires_grad(getattr(model, 'projector_p_normal', None), False)

            _set_requires_grad(model.proj_z_p_seg, False)
            _set_requires_grad(model.proj_z_p_depth, False)
            _set_requires_grad(getattr(model, 'proj_z_p_normal', None), False)

            _set_requires_grad(model.zp_seg_refiner, False)
            _set_requires_grad(model.zp_depth_refiner, False)
            _set_requires_grad(getattr(model, 'zp_normal_refiner', None), False)

            _set_requires_grad(model.decoder_zp_depth, False)
            _set_requires_grad(getattr(model, 'decoder_zp_normal', None), False)

            logging.info("Stage-1: Structure Warmup. Frozen z_p branches, Unfrozen Task Heads.")

        # === Stage 2: 全面训练 (Full Training) ===
        else:
            # 1. 确保任务头是解冻的
            _set_requires_grad(model.predictor_seg, True)
            _set_requires_grad(model.predictor_depth, True)
            if hasattr(model, 'predictor_normal'):
                _set_requires_grad(model.predictor_normal, True)

            # 2. 解冻所有 z_p 私有分支
            _set_requires_grad(model.projector_p_seg, True)
            _set_requires_grad(model.projector_p_depth, True)
            _set_requires_grad(getattr(model, 'projector_p_normal', None), True)

            _set_requires_grad(model.proj_z_p_seg, True)
            _set_requires_grad(model.proj_z_p_depth, True)
            _set_requires_grad(getattr(model, 'proj_z_p_normal', None), True)

            _set_requires_grad(model.zp_seg_refiner, True)
            _set_requires_grad(model.zp_depth_refiner, True)
            _set_requires_grad(getattr(model, 'zp_normal_refiner', None), True)

            _set_requires_grad(model.decoder_zp_depth, True)
            _set_requires_grad(getattr(model, 'decoder_zp_normal', None), True)

            logging.info("Stage-2: unfrozen private (z_p) branches.")


def _get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get("lr", None)


def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _build_scheduler(optimizer, train_cfg):
    """
    自动构建调度器：Cosine 或 Step
    """
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    sched_cfg = train_cfg.get("lr_scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "cosine")).lower()

    if sched_type == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 3))
        min_lr_factor = float(sched_cfg.get("min_lr_factor", 0.1))
        total_epochs = int(train_cfg.get("epochs", 30))
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

    # fallback: StepLR (LibMTL 默认使用这个)
    step_size = int(sched_cfg.get("step_size", 100))
    gamma = float(sched_cfg.get("gamma", 0.5))
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

    # 自动计算累积步数 (保持原有逻辑)
    target_bs = 16
    physical_bs = train_loader.batch_size
    accumulation_steps = max(1, target_bs // physical_bs)

    optimizer.zero_grad(set_to_none=True)

    # [修改 1] 删除 scaler 初始化
    # scaler = GradScaler()  <-- 删除

    for i, batch in enumerate(pbar):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rgb = batch['rgb']

        # [修改 2] 删除 with autocast():，直接运行模型
        # with autocast():  <-- 删除这一行，下面的代码取消一级缩进
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

        loss_normalized = total_loss / accumulation_steps

        # [修改 3] 删除 scaler.scale，直接反向传播
        # scaler.scale(loss_normalized).backward() <-- 删除
        loss_normalized.backward()  # <-- 改为这样

        # 梯度更新
        if (i + 1) % accumulation_steps == 0:
            # [强烈推荐] 即使是 FP32，保留梯度裁剪也是防止训练崩溃的最佳实践
            # 之前提到的 unscale_ 也不需要了，直接 clip 即可
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # [修改 4] 删除 scaler.step 和 scaler.update，改回标准 step
            # scaler.step(optimizer)  <-- 删除
            # scaler.update()         <-- 删除
            optimizer.step()  # <-- 改为这样

            optimizer.zero_grad(set_to_none=True)

        # 记录 Loss (加个防止 NaN 的判断，虽在 FP32 下很难出现)
        loss_val = total_loss.item()
        if not np.isfinite(loss_val):
            print(f"Warning: Non-finite loss {loss_val} at step {i}")

        total_train_loss += float(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    # 处理剩余梯度 (如果有)
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = total_train_loss / max(1, len(train_loader))
    logging.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss

def calculate_improvement(base_metrics, current_metrics, data_type='nyuv2'):
    """相对提升率计算 (LibMTL 对齐)"""
    improvement = 0
    count = 0
    # 定义指标方向: 1=越大越好, 0=越小越好
    metric_meta = {
        'seg_miou': 1, 'seg_pixel_acc': 1,
        'depth_abs_err': 0, 'depth_rel_err': 0,
        'normal_mean_angle': 0, 'normal_acc_30': 1,
        'normal_median_angle': 0, 'normal_acc_11': 1, 'normal_acc_22': 1
    }
    if 'gta5' in data_type:
        # Sim-to-Real: 仅关注分割，忽略 Target 域上未经训练的深度/法线
        valid_keys = {'seg_miou', 'seg_pixel_acc'}

    elif  data_type == 'cityscapes':
        # Cityscapes MTL: 关注 分割 + 深度 (法线不存在)
        valid_keys = {'seg_miou', 'seg_pixel_acc', 'depth_abs_err', 'depth_rel_err'}

    else:  # Default (e.g., 'nyuv2')
        # Indoor MTL: 全都要
        valid_keys = set(metric_meta.keys())
    for k, direction in metric_meta.items():
        if k not in valid_keys:
            continue
        if k in base_metrics and k in current_metrics:
            base = base_metrics[k]
            curr = current_metrics[k]
            if base == 0: continue

            # 越小越好时：(Base - Curr) / Base
            # 越大越好时：(Curr - Base) / Base
            if direction == 1:
                imp = (curr - base) / base
            else:
                imp = (base - curr) / base
            improvement += imp
            count += 1

    return improvement / max(1, count)


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device,
          checkpoint_dir='checkpoints'):
    # 1. 提取 dataset_type (转小写以防万一)
    data_type = config['data'].get('type', 'nyuv2').lower()

    # 2. 读取训练配置
    train_cfg = config['training']
    stage1_epochs = int(train_cfg.get('stage1_epochs', 10))
    total_epochs = int(train_cfg.get('epochs', 30))
    base_lr = float(train_cfg.get("learning_rate", 1e-4))

    # 3. 初始化基准变量
    best_relative_score = -float('inf')
    baseline_metrics = None
    best_epoch = 0
    best_metrics_details = {}

    # 构建调度器
    sched = _build_scheduler(optimizer, train_cfg)
    logging.info(f"[LR Scheduler] {sched['type']}; base_lr={base_lr}")

    stage0_epochs = int(train_cfg.get('stage0_epochs', 0))
    for epoch in range(total_epochs):
        if epoch < stage0_epochs:
            stage = 0
        elif epoch < stage1_epochs:
            stage = 1
        else:
            stage = 2
        if epoch == 0 or epoch == stage0_epochs or epoch == stage1_epochs:
            _switch_stage_freeze(model, stage)
        target_ind_lambda = float(config['losses'].get('lambda_independence', 0.0))
        ind_warmup_epochs = int(train_cfg.get('ind_warmup_epochs', 0))

        current_ind_lambda = target_ind_lambda
        if stage < 2:
            # Stage 0/1 强制为 0 (虽然 Loss 内部也有判断，但这里显式控制更安全)
            current_ind_lambda = 0.0
        elif ind_warmup_epochs > 0:
            # Stage 2：开始 Warmup
            # 关键点：进度 = (当前Epoch - Stage2开始Epoch)
            progress = epoch - stage1_epochs

            # 限制比例在 0.0 到 1.0 之间
            ratio = min(1.0, max(0.0, progress / float(ind_warmup_epochs)))
            current_ind_lambda = target_ind_lambda * ratio
        if hasattr(criterion, 'weights'):
            criterion.weights['lambda_independence'] = torch.tensor(current_ind_lambda, device=device)
        # ---- Warm-up (Cosine only) ----
        if sched["type"] == "cosine":
            warmup_epochs = sched["warmup_epochs"]
            if epoch < warmup_epochs:
                warmup_start = 0.1 * base_lr
                ratio = float(epoch + 1) / float(max(1, warmup_epochs))
                lr_now = warmup_start + (base_lr - warmup_start) * ratio
                _set_lr(optimizer, lr_now)
            else:
                if abs(_get_lr(optimizer) - base_lr) > 1e-12 and epoch == warmup_epochs:
                    _set_lr(optimizer, base_lr)

        cur_lr = _get_lr(optimizer)
        logging.info(f"\n----- Starting Epoch {epoch + 1}/{total_epochs} (Stage {stage}) | lr={cur_lr:.6f} -----")

        # --- Train ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage=stage)

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, criterion, device, stage=stage,data_type=data_type)

        # --- Quick diagnose (Optional) ---
        if os.environ.get("QUICK_DIAG", "0") == "1" and (epoch == 0 or epoch == stage1_epochs):
            from engine.evaluator import quick_diagnose
            quick_diagnose(model, val_loader, device)

        # --- Step Scheduler ---
        if sched["type"] == "cosine":
            if epoch >= sched["warmup_epochs"]:
                sched["cosine"].step()
        else:
            sched["step"].step()

        # --- Best Model Selection Logic (LibMTL Aligned) ---
        if epoch < stage0_epochs:
            baseline_metrics = val_metrics
            is_best = True
            best_relative_score = 0.0
            logging.info("  -> Stage 0 won't service as Baseline for improvement calculation.")
        elif epoch == stage0_epochs:
            # Stage 1 Epoch 0 作为基准线
            baseline_metrics = val_metrics
            is_best = True
            best_relative_score = 0.0
            logging.info("  -> Stage 1 Epoch 0 set as Baseline for improvement calculation.")
        else:
            # 计算相对于 Epoch 0 的提升
            score = calculate_improvement(baseline_metrics, val_metrics,data_type=data_type)
            is_best = (score > best_relative_score)

            if is_best:
                best_relative_score = score
                best_epoch = epoch + 1
                best_metrics_details = val_metrics.copy()  # 保存最佳时刻的指标副本
                logging.info(f"  -> 🏆 New best model found! Avg Improvement vs Epoch 0: {score:.2%}")

                metrics_log = (
                    f"     [Tasks] Seg: mIoU={val_metrics.get('seg_miou', 0):.4f} Acc={val_metrics.get('seg_pixel_acc', 0):.4f} \n"
                )
                if "gta5" not in data_type:
                    metrics_log += (f"Depth: Abs={val_metrics.get('depth_abs_err', 0):.4f} Rel={val_metrics.get('depth_rel_err', 0):.4f} \n")
                # 法线 (Normal)：只在 NYUv2 下打印 (硬逻辑)
                if 'nyuv2' in data_type:
                    metrics_log += (
                        f"\n     [Normal] Mean={val_metrics.get('normal_mean_angle', 0):.2f}° Med={val_metrics.get('normal_median_angle', 0):.2f}° | "
                        f"Acc: 11°={val_metrics.get('normal_acc_11', 0):.3f} 22°={val_metrics.get('normal_acc_22', 0):.3f} 30°={val_metrics.get('normal_acc_30', 0):.3f}"
                    )
                logging.info(metrics_log)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_score': best_relative_score,
        }, is_best, checkpoint_dir=checkpoint_dir)

    # =========================================================
    # [FINAL LOG] 训练结束时的详细总结 (已修复数据类型判断)
    # =========================================================
    logging.info("\n" + "=" * 60)
    logging.info(f"🏆 Best Model Selection Summary (Epoch {best_epoch}):")
    logging.info(f"   Relative Improvement vs Epoch 0: {best_relative_score:.2%}")
    logging.info("-" * 60)
    logging.info("-- Best Epoch Downstream Task Metrics --")

    # 1. Segmentation (通用)
    miou = best_metrics_details.get('seg_miou', 0.0)
    pix_acc = best_metrics_details.get('seg_pixel_acc', 0.0)
    logging.info(f"  - Segmentation: mIoU={miou:.4f}, Pixel Acc={pix_acc:.4f}")

    # 2. Depth (通用)
    abs_err = best_metrics_details.get('depth_abs_err', 0.0)
    rel_err = best_metrics_details.get('depth_rel_err', 0.0)
    logging.info(f"  - Depth:        Abs Err={abs_err:.4f}, Rel Err={rel_err:.4f}")

    # 3. Normal (仅 NYUv2 输出)
    if 'nyuv2' in data_type:
        mean_ang = best_metrics_details.get('normal_mean_angle', 0.0)
        med_ang = best_metrics_details.get('normal_median_angle', 0.0)
        acc_11 = best_metrics_details.get('normal_acc_11', 0.0)
        acc_22 = best_metrics_details.get('normal_acc_22', 0.0)
        acc_30 = best_metrics_details.get('normal_acc_30', 0.0)
        logging.info(f"  - Normal:       Mean Ang={mean_ang:.2f}°, Median Ang={med_ang:.2f}°")
        logging.info(f"                  Acc@11.25°={acc_11:.4f}, Acc@22.5°={acc_22:.4f}, Acc@30°={acc_30:.4f}")

    # 4. Scene (已废弃，注释掉)
    # scene_acc = best_metrics_details.get('scene_acc', 1.0)
    # if scene_acc != 1.0:
    #     logging.info(f"  - Scene Classification (Acc): {scene_acc:.4f}")

    logging.info("=" * 60 + "\n")