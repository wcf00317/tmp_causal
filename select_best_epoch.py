import re
import numpy as np
import pandas as pd
import argparse
import os
import sys

# ==========================================
# 0. 全局路径配置
# ==========================================
# 这里定义日志所在的根目录
BASE_LOG_DIR = "/data/chengfengwu/alrl/causal_mtl/runs/"

# ==========================================
# 1. 配置区域：单任务 Baseline (STL) 数据
# ==========================================
STL_BASELINES = {
    'nyuv2': {
        # 越大越好的指标 (Direction = 1)
        'mIoU': {'val': 0.2343, 'dir': 1},
        'Pixel Acc': {'val': 0.8395, 'dir': 1},
        'Scene Accuracy': {'val': 0.2793, 'dir': 1},

        # 越小越好的指标 (Direction = -1)
        'RMSE': {'val': 0.6681, 'dir': -1},
        'MAE': {'val': 0.4276, 'dir': -1},
        'Abs Rel': {'val': 0.1773, 'dir': -1},
    },
    'cityscapes': {
        # 越大越好的指标 (Direction = 1)
        'mIoU': {'val': 0.6099, 'dir': 1},
        'Pixel Acc': {'val': 0.9310, 'dir': 1},

        # 越小越好的指标 (Direction = -1)
        'RMSE': {'val': 8.7620, 'dir': -1},
        'MAE': {'val': 3.9452, 'dir': -1},
        'Abs Rel': {'val': 0.1821, 'dir': -1},
    }
}


def extract_run_id(log_content):
    """
    从日志中提取实验路径 ID
    目标行示例: Loading best model from runs/2025-12-04_14-14/checkpoints...
    """
    match = re.search(r"Loading best model from (runs/[\d\-_]+)", log_content)
    if match:
        return match.group(1)
    return "Unknown Run ID"


def detect_dataset_type(log_content):
    """
    从日志中检测数据集类型
    """
    match = re.search(r"data:\s+type:\s+(\w+)", log_content)
    if match:
        return match.group(1).lower()

    if "nyu" in log_content.lower():
        return 'nyuv2'
    if "cityscape" in log_content.lower():
        return 'cityscapes'

    return 'unknown'


def parse_training_log(log_content):
    """
    解析日志，提取每个 Epoch 的指标
    """
    epoch_data = {}
    current_epoch = -1

    epoch_pattern = re.compile(r"Starting Epoch (\d+)/(\d+)")
    seg_pattern = re.compile(r"Segmentation: mIoU=([\d\.]+), Pixel Acc=([\d\.]+)")
    depth_pattern = re.compile(r"Depth:\s+RMSE=([\d\.]+), MAE=([\d\.]+), Abs Rel=([\d\.]+)")
    scene_pattern = re.compile(r"Scene Classification \(Acc\): ([\d\.|N/A]+)")

    lines = log_content.split('\n')

    for line in lines:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epoch_data[current_epoch] = {}
            continue

        if current_epoch == -1:
            continue

        seg_match = seg_pattern.search(line)
        if seg_match:
            epoch_data[current_epoch]['mIoU'] = float(seg_match.group(1))
            epoch_data[current_epoch]['Pixel Acc'] = float(seg_match.group(2))

        depth_match = depth_pattern.search(line)
        if depth_match:
            epoch_data[current_epoch]['RMSE'] = float(depth_match.group(1))
            epoch_data[current_epoch]['MAE'] = float(depth_match.group(2))
            epoch_data[current_epoch]['Abs Rel'] = float(depth_match.group(3))

        scene_match = scene_pattern.search(line)
        if scene_match:
            val_str = scene_match.group(1)
            val = 0.0 if val_str == 'N/A' else float(val_str)
            epoch_data[current_epoch]['Scene Accuracy'] = val

    return epoch_data


def find_best_epoch_stl_relative(epoch_data, dataset_name):
    """
    核心逻辑：基于 Single-Task Baseline 计算相对提升率
    """
    if dataset_name not in STL_BASELINES:
        print(f"⚠️ 警告：未找到数据集 '{dataset_name}' 的 Baseline 配置。")
        return None, None

    baseline_cfg = STL_BASELINES[dataset_name]

    if dataset_name == 'cityscapes' and baseline_cfg['mIoU']['val'] == 0:
        print("⚠️ 提示：Cityscapes Baseline 尚未配置，请更新代码。")
        return None, None

    results = []

    for epoch, metrics in epoch_data.items():
        total_rel_score = 0
        count = 0
        row_data = {'Epoch': epoch}

        for metric_name, cfg in baseline_cfg.items():
            if metric_name not in metrics:
                continue

            val_mtl = metrics[metric_name]
            val_stl = cfg['val']
            direction = cfg['dir']

            row_data[metric_name] = val_mtl

            # 计算相对提升率
            if direction == 1:  # 越大越好
                score = (val_mtl - val_stl) / val_stl
            else:  # 越小越好
                score = (val_stl - val_mtl) / val_stl

            total_rel_score += score
            count += 1

        if count > 0:
            avg_score = total_rel_score / count
            row_data['Relative Score'] = avg_score
            results.append(row_data)

    if not results:
        return None, None

    df = pd.DataFrame(results)
    df.set_index('Epoch', inplace=True)
    best_epoch = df['Relative Score'].idxmax()
    return best_epoch, df


# ================= 主程序入口 =================

if __name__ == "__main__":
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="根据 Run ID 自动分析 run.log 并选择最佳 Epoch")
    parser.add_argument("run_name", type=str, help="实验文件夹名称 (例如: 2025-12-04_14-14)")
    args = parser.parse_args()

    # 2. 拼接完整路径
    log_file_path = os.path.join(BASE_LOG_DIR, args.run_name, "run.log")

    print(f"📂 正在分析日志文件: {log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 3. 提取并打印 Run ID (本次实验的标识)
        run_id = extract_run_id(content)

        # 4. 自动识别数据集
        dataset_type = detect_dataset_type(content)
        print(f"🔍 检测到数据集类型: {dataset_type.upper()}")

        # 5. 打印当前使用的 Baseline 值
        if dataset_type in STL_BASELINES:
            print("\n📊 使用的 Single-Task Baseline 参考值:")
            print("-" * 40)
            print(f"{'Metric':<15} | {'STL Value':<10} | {'Direction'}")
            print("-" * 40)
            for k, v in STL_BASELINES[dataset_type].items():
                direction_str = "↑ (Higher is better)" if v['dir'] == 1 else "↓ (Lower is better)"
                print(f"{k:<15} | {v['val']:<10.4f} | {direction_str}")
            print("-" * 40)

        # 6. 解析数据
        data = parse_training_log(content)
        print(f"\n✅ 解析完成，共提取了 {len(data)} 个 Epoch 的数据。")

        # 7. 计算最佳 Epoch
        best_ep, df_res = find_best_epoch_stl_relative(data, dataset_type)

        if best_ep:
            print(f"🏆 综合推荐的最佳模型 (Epoch {best_ep})")
            print(f"   平均相对提升率: {df_res.loc[best_ep]['Relative Score']:.2%} (vs Single-Task)")
            print("=" * 60)
            print("\n[Output Dictionary]:")
            print(data[best_ep])
            print(f"📂 实验日志标识 (Run ID): {run_id}")

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件，请检查路径是否正确：\n{log_file_path}")
        sys.exit(1)