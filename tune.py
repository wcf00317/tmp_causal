#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tune.py — Optuna 超参搜索（只搜 losses；值级封禁 + 比例守恒预剪枝 + 在线剪枝；屏幕可见日志）
从标准输出解析这些行（最后一次出现为准）：
  - CKA (z_s vs z_p_seg):   <float>
  - CKA (z_s vs z_p_depth): <float>
  - CKA (z_s vs z_p_scene): <float>
  - Segmentation (mIoU):    <float>
  - Depth (RMSE):           <float>
  - Scene Classification (Acc): <float>
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path

import yaml
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

# -------------------------
# I/O & utils
# -------------------------
def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))

def _get_last_float(pattern: str, text: str):
    """返回文本中 pattern 的最后一次匹配的 float；找不到返回 None。"""
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None

def parse_stdout_metrics(stdout_text: str):
    """
    解析关键指标。若 CKA 三项缺失则返回 None。
    """
    m = {}
    # CKA（三项必需）
    m["cka_seg"]   = _get_last_float(r"CKA\s*\(z_s\s*vs\s*z_p_seg\)\s*:\s*([0-9]*\.?[0-9]+)", stdout_text)
    m["cka_depth"] = _get_last_float(r"CKA\s*\(z_s\s*vs\s*z_p_depth\)\s*:\s*([0-9]*\.?[0-9]+)", stdout_text)
    m["cka_scene"] = _get_last_float(r"CKA\s*\(z_s\s*vs\s*z_p_scene\)\s*:\s*([0-9]*\.?[0-9]+)", stdout_text)
    if any(v is None for v in [m["cka_seg"], m["cka_depth"], m["cka_scene"]]):
        return None

    # 任务（可选）
    m["seg_miou"]   = _get_last_float(r"Segmentation\s*\(mIoU\)\s*:\s*([0-9]*\.?[0-9]+)", stdout_text)
    m["depth_rmse"] = _get_last_float(r"Depth\s*\(RMSE\)\s*:\s*([0-9]*\.?[0-9]+)", stdout_text)
    m["scene_acc"]  = _get_last_float(r"Scene\s*Classification\s*\(Acc\)\s*:\s*([0-9]*\.?[0-9]+)", stdout_text)
    return m

# -------------------------
# Scoring
# -------------------------
def calculate_final_score(metrics: dict) -> float:
    """
    必需：cka_seg / cka_depth / cka_scene ∈ [0,1]（越小越独立）
    可选：seg_miou ∈ [0,1]；depth_rmse（越小越好）；scene_acc ∈ [0,1]
    """
    for k in ["cka_seg", "cka_depth", "cka_scene"]:
        if k not in metrics or metrics[k] is None:
            raise optuna.TrialPruned(f"Missing CKA metric: {k}")

    avg_cka = (float(metrics["cka_seg"]) + float(metrics["cka_depth"]) + float(metrics["cka_scene"])) / 3.0
    decouple_score = clip01(1.0 - avg_cka)

    seg_score = None
    if metrics.get("seg_miou") is not None:
        miou = float(metrics["seg_miou"])
        if miou > 1.2:  # 百分比情况
            miou /= 100.0
        seg_score = clip01(miou)

    depth_score = None
    if metrics.get("depth_rmse") is not None:
        rmse = float(metrics["depth_rmse"])
        depth_score = 1.0 / (1.0 + max(0.0, rmse))

    scene_score = None
    if metrics.get("scene_acc") is not None:
        scene_score = clip01(float(metrics["scene_acc"]))

    # 聚合（只对存在的项分配权重）
    parts, weights = [], []
    parts.append(decouple_score); weights.append(0.35)
    if depth_score is not None: parts.append(depth_score); weights.append(0.25)
    if seg_score   is not None: parts.append(seg_score);   weights.append(0.20)
    if scene_score is not None: parts.append(scene_score); weights.append(0.20)

    s = sum(weights); weights = [w/s for w in weights]
    eps = 1e-8
    final = (sum(weights)) / sum(w / (p + eps) for w, p in zip(weights, parts))

    # 硬阈
    if depth_score is not None and depth_score < 0.2: final *= 0.2
    if seg_score   is not None and seg_score   < 0.2: final *= 0.2
    if scene_score is not None and scene_score < 0.2: final *= 0.2
    return float(final)

# -------------------------
# 值级别封禁（ban choices）：某取值屡次很差 -> 后续不再采样
# -------------------------
def _filter_bad_choices(trial: optuna.trial.Trial,
                        name: str,
                        choices: list,
                        min_trials_per_value: int = 1,
                        ban_if_best_below: float = 0.12,
                        verbose: bool = True) -> list:
    """
    若某个取值在历史中被尝试 >= min_trials_per_value 次，且该取值的最佳得分 < ban_if_best_below，
    则从候选集中剔除。若全部被剔除则回退到原 choices。
    """
    vals_by_choice = {c: [] for c in choices}
    for t in trial.study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        if name in t.params:
            vals_by_choice.setdefault(t.params[name], []).append(t.value)

    filtered, banned = [], []
    for c in choices:
        vals = vals_by_choice.get(c, [])
        best = max(vals) if vals else -1.0
        if len(vals) >= min_trials_per_value and best < ban_if_best_below:
            banned.append(c)
        else:
            filtered.append(c)

    if not filtered:
        filtered = choices

    if verbose and banned:
        print(f"[HPO] Param '{name}' banned (best<{ban_if_best_below}): {banned} -> {filtered}", flush=True)
    return filtered

# -------------------------
# Trial execution (TEE stdout/stderr + 在线剪枝)
# -------------------------
def try_import_main():
    try:
        from main import main as main_func
        return main_func
    except Exception:
        return None

def _pump_stdout_stream(stream, sink_file, buffer_list, trial, popen_ref, online_cfg, state):
    """
    stdout 线程：tee 到屏幕与文件；当捕获到一个完整的 Validation 块时，解析指标并做在线剪枝。
    online_cfg: {
        "min_epoch_before_prune": int,
        "max_avg_cka": float,
        "min_score": float
    }
    state: {"epoch": int, "last_metrics": dict|None}
    """
    in_val = False
    val_lines = []

    epoch_patterns = [
        re.compile(r"^-+\s*Starting\s*Epoch\s*(\d+)\s*/", re.IGNORECASE),
        re.compile(r"^Epoch\s+(\d+)\b", re.IGNORECASE),
    ]

    for line in iter(stream.readline, ''):
        sink_file.write(line); sink_file.flush()
        buffer_list.append(line)
        print(line, end='', flush=True)

        # 识别 epoch
        for pat in epoch_patterns:
            m = pat.search(line)
            if m:
                try:
                    state["epoch"] = int(m.group(1))
                except Exception:
                    pass
                break

        # 捕捉 Validation 块
        if not in_val and "--- Validation Results ---" in line:
            in_val = True
            val_lines = [line]
            continue

        if in_val:
            val_lines.append(line)
            if "--------------------------" in line:
                # 完整的 Validation 块结束
                chunk = ''.join(val_lines)
                in_val = False
                # 解析指标
                m = parse_stdout_metrics(chunk)
                if m:
                    state["last_metrics"] = m
                    # 计算 partial score
                    try:
                        score = calculate_final_score(m)
                        avg_cka = (m["cka_seg"] + m["cka_depth"] + m["cka_scene"]) / 3.0
                    except Exception:
                        score, avg_cka = 0.0, 1.0

                    # 中途上报与剪枝（达到最小 epoch 后）
                    step = state.get("epoch", 0)
                    if trial is not None:
                        trial.report(score, step=step)

                        # 规则 1：分数过低
                        bad_score = (score < online_cfg["min_score"])
                        # 规则 2：平均 CKA 很高
                        bad_cka = (avg_cka > online_cfg["max_avg_cka"])
                        # 规则 3：由 pruner 策略决定
                        bad_pruner = trial.should_prune() if step >= online_cfg["min_epoch_before_prune"] else False

                        if step >= online_cfg["min_epoch_before_prune"] and (bad_score or bad_cka or bad_pruner):
                            print(f"[OnlinePrune] epoch={step} score={score:.4f} avgCKA={avg_cka:.4f} -> terminate.", flush=True)
                            try:
                                if popen_ref[0] is not None:
                                    popen_ref[0].terminate()
                            except Exception:
                                pass
                            return  # 结束 stdout 线程，主线程稍后 join

def _pump_stderr_stream(stream, sink_file, buffer_list):
    for line in iter(stream.readline, ''):
        sink_file.write(line); sink_file.flush()
        buffer_list.append(line)
        print(line, end='', file=sys.stderr, flush=True)

def run_trial(config: dict, trial_dir: Path, gpus: str = None, trial: optuna.trial.Trial = None) -> dict:
    """
    优先 import main(config)->metrics；若失败则子进程执行 main.py。
    子进程采用 tee 模式，并在 stdout 线程中进行在线剪枝。
    """
    ensure_dir(trial_dir)
    env = os.environ.copy()
    env["HPO_TRIAL_DIR"] = str(trial_dir.resolve())
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)

    # 方案 1：import main（若你的 main 能返回 dict）
    main_func = try_import_main()
    if main_func is not None:
        try:
            metrics = main_func(config)
            if isinstance(metrics, dict):
                return metrics
        except Exception:
            pass  # 失败则走子进程

    # 方案 2：子进程，TEE 到屏幕与文件 + 在线剪枝
    cfg_path = trial_dir / "config.yaml"
    dump_yaml(config, str(cfg_path))

    cmd = [sys.executable, "main.py", "--config", str(cfg_path)]
    stdout_file = trial_dir / "stdout.txt"
    stderr_file = trial_dir / "stderr.txt"

    print(f"[run_trial] exec: {' '.join(cmd)}", flush=True)
    print(f"[run_trial] CWD : {os.getcwd()}", flush=True)

    stdout_lines, stderr_lines = [], []
    popen_ref = [None]  # 让线程能访问到 Popen 对象
    state = {"epoch": 0, "last_metrics": None}
    online_cfg = {
        "min_epoch_before_prune": 15,  # 至少跑到第2个验证再考虑剪枝
        "max_avg_cka": 0.60,          # 平均 CKA 超过 0.90 视为明显失败
        "min_score": 0.05,            # 你的综合分阈值
    }

    with open(stdout_file, "w") as out_f, open(stderr_file, "w") as err_f:
        p = subprocess.Popen(
            cmd, env=env, cwd=os.getcwd(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        popen_ref[0] = p
        t_out = threading.Thread(target=_pump_stdout_stream,
                                 args=(p.stdout, out_f, stdout_lines, trial, popen_ref, online_cfg, state))
        t_err = threading.Thread(target=_pump_stderr_stream,
                                 args=(p.stderr, err_f, stderr_lines))
        t_out.start(); t_err.start()
        p.wait()
        t_out.join(); t_err.join()

    if p.returncode != 0:
        print(f"[run_trial] WARNING: subprocess returned code {p.returncode}", flush=True)

    # 优先返回在线期间解析到的最近一次指标
    metrics = state.get("last_metrics")
    if metrics is None:
        # 兜底：用整段 stdout 再解析一次
        stdout_text = ''.join(stdout_lines)
        metrics = parse_stdout_metrics(stdout_text)

    if metrics is None:
        # 再兜底：若 main 将 metrics.json 写在 trial 目录
        for candidate in ["metrics.json", "final_metrics.json", "results.json"]:
            jf = trial_dir / candidate
            if jf.exists():
                try:
                    with open(jf, "r") as f:
                        metrics = json.load(f)
                        break
                except Exception:
                    pass

    if metrics is None:
        raise optuna.TrialPruned("No metrics found in stdout or files")

    return metrics

# -------------------------
# Search space（只搜 losses；粗粒度 + 值级封禁）
# -------------------------
def build_trial_config(base_cfg: dict, trial: optuna.trial.Trial) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # 深拷贝

    # ===== 任务主损失 =====
    ch = [6.0, 10.0, 14.0]
    ch = _filter_bad_choices(trial, "lambda_seg", ch)
    cfg["losses"]["lambda_seg"] = trial.suggest_categorical("lambda_seg", ch)

    ch = [3.0, 5.0, 8.0]
    ch = _filter_bad_choices(trial, "lambda_depth", ch)
    cfg["losses"]["lambda_depth"] = trial.suggest_categorical("lambda_depth", ch)

    ch = [3.0, 5.0, 8.0]
    ch = _filter_bad_choices(trial, "lambda_scene", ch)
    cfg["losses"]["lambda_scene"] = trial.suggest_categorical("lambda_scene", ch)

    # ===== 解耦与重构 =====
    ch = [ 0.5,1.0,1.5,2.0]
    ch = _filter_bad_choices(trial, "lambda_independence", ch)
    cfg["losses"]["lambda_independence"] = trial.suggest_categorical("lambda_independence", ch)

    ch = [0.5, 1.0, 1.5]
    ch = _filter_bad_choices(trial, "alpha_recon_geom", ch)
    cfg["losses"]["alpha_recon_geom"] = trial.suggest_categorical("alpha_recon_geom", ch)

    ch = [2.0, 3.0, 4.5, 6.0]
    ch = _filter_bad_choices(trial, "beta_recon_app", ch)
    cfg["losses"]["beta_recon_app"] = trial.suggest_categorical("beta_recon_app", ch)

    ch = [0.0, 1.0, 3.0]
    ch = _filter_bad_choices(trial, "lambda_l1_recon", ch)
    cfg["losses"]["lambda_l1_recon"] = trial.suggest_categorical("lambda_l1_recon", ch)

    ch = [0.0, 0.5, 1.0]
    ch = _filter_bad_choices(trial, "alpha_recon_geom_aux", ch)
    cfg["losses"]["alpha_recon_geom_aux"] = trial.suggest_categorical("alpha_recon_geom_aux", ch)

    ch = [0.0, 1.5, 2.5, 3.5]
    ch = _filter_bad_choices(trial, "beta_recon_app_aux", ch)
    cfg["losses"]["beta_recon_app_aux"] = trial.suggest_categorical("beta_recon_app_aux", ch)

    ch = [0.0, 0.5, 1.0]
    ch = _filter_bad_choices(trial, "lambda_depth_zp", ch)
    cfg["losses"]["lambda_depth_zp"] = trial.suggest_categorical("lambda_depth_zp", ch)

    # ===== 边缘一致性 =====
    ch = [0.0, 0.1, 0.2]
    ch = _filter_bad_choices(trial, "lambda_edge_consistency", ch)
    cfg["losses"]["lambda_edge_consistency"] = trial.suggest_categorical("lambda_edge_consistency", ch)

    ch = [0.05, 0.1, 0.2]
    ch = _filter_bad_choices(trial, "alpha_recon_geom_edges", ch)
    cfg["losses"]["alpha_recon_geom_edges"] = trial.suggest_categorical("alpha_recon_geom_edges", ch)

    ch = [0.05, 0.1, 0.2, 0.3]
    ch = _filter_bad_choices(trial, "beta_seg_edge_from_geom", ch)
    cfg["losses"]["beta_seg_edge_from_geom"] = trial.suggest_categorical("beta_seg_edge_from_geom", ch)

    # ===== HPO 阶段缩短 epochs =====
    cfg["training"]["epochs"] = min(cfg["training"].get("epochs", 30), 15)

    return cfg

def objective_factory(base_cfg: dict, gpus: str):
    def objective(trial: optuna.trial.Trial):
        cfg = build_trial_config(base_cfg, trial)

        # --- 比例守恒的开跑前剪枝（避免明显高 CKA 的组合） ---
        task_sum = (
            cfg["losses"]["lambda_seg"] +
            cfg["losses"]["lambda_depth"] +
            cfg["losses"]["lambda_scene"]
        )
        ratio = cfg["losses"]["lambda_independence"] / max(1e-6, task_sum)
        if ratio < 0.02:  # 经验范围：0.04~0.06；你可调整
            print(f"[PrePrune] too-weak independence ratio={ratio:.4f} (ind={cfg['losses']['lambda_independence']} vs task_sum={task_sum})", flush=True)
            raise optuna.TrialPruned()

        trial_id = f"trial_{trial.number:04d}_{uuid.uuid4().hex[:6]}"
        trial_dir = Path("hpo_runs") / trial_id
        ensure_dir(trial_dir)

        # 保存本次 trial 的配置（也便于你事后审计）
        dump_yaml(cfg, trial_dir / "config.yaml")
        print(f"[Trial {trial.number}] params = {trial.params}", flush=True)

        t0 = time.time()
        try:
            metrics = run_trial(cfg, trial_dir, gpus=gpus, trial=trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[Trial {trial.number}] ERROR: {repr(e)}", flush=True)
            raise optuna.TrialPruned()

        score = calculate_final_score(metrics)

        # 记录结果
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(trial_dir / "score.txt", "w") as f:
            f.write(str(score))

        # 结束上报（仍保留尾部低分剪枝以便 TPE 学习）
        trial.report(score, step=9999)
        if score < 0.05:
            print(f"[Trial {trial.number}] PRUNED (final score={score:.4f})", flush=True)
            raise optuna.TrialPruned()

        dt = time.time() - t0
        print(f"[Trial {trial.number}] DONE score={score:.4f} time={dt/60:.1f}m dir={trial_dir}", flush=True)
        return score
    return objective

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True, help="基础 YAML 配置路径")
    ap.add_argument("--trials", type=int, default=60, help="试验次数（粗搜建议 40~80）")
    ap.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES，例如 '0' 或 '0,1'")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    optuna.logging.set_verbosity(optuna.logging.INFO)
    args = parse_args()
    print(f"[CLI] base_config={args.base_config} trials={args.trials} gpus={args.gpus} seed={args.seed}", flush=True)

    base_cfg = load_yaml(args.base_config)
    ensure_dir(Path("hpo_runs"))

    sampler = TPESampler(seed=args.seed, n_startup_trials=10, multivariate=True)
    pruner = SuccessiveHalvingPruner(reduction_factor=3, min_resource=1, min_early_stopping_rate=0)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # （可选）预置前几次试验的“靠谱起点”，帮助 TPE 早期收敛
    for p in [
        {"lambda_independence": 0.5},
        {"lambda_independence": 1.0},
        {"lambda_independence": 1.5},
    ]:
        study.enqueue_trial(p)

    study.optimize(objective_factory(base_cfg, gpus=args.gpus),
                   n_trials=args.trials, gc_after_trial=True)

    print("\n=== HPO 完成 ===")
    print(f"Best Value: {study.best_value:.6f}")
    print("Best Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # 输出一份“最优配置”的 YAML（便于复现）
    best_cfg = build_trial_config(base_cfg, study.best_trial)
    dump_yaml(best_cfg, "hpo_best_config.yaml")
    print("已写出: hpo_best_config.yaml")

if __name__ == "__main__":
    main()
