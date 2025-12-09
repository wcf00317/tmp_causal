import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os, argparse
import logging
from datetime import datetime

# --- 数据集导入 ---
from data_utils.nyuv2_dataset import NYUv2Dataset
from data_utils.gta5_dataset import GTA5Dataset
from data_utils.cityscapes_dataset import CityscapesDataset

# --- 模型与Loss导入 ---
from models.causal_model import CausalMTLModel
from losses.composite_loss import AdaptiveCompositeLoss
from losses.mtl_loss import MTLLoss
from models.baselines import RawMTLModel, SingleTaskModel
from losses.single_task_loss import SingleTaskLoss

# --- 引擎工具导入 ---
from engine.trainer import train
from engine.visualizer import generate_visual_reports
from engine.experiments import run_all_experiments
from utils.general_utils import set_seed, setup_logging


def main(config_path):
    """
    项目主函数（适配 LibMTL 数据格式版）。
    """
    # 1. 加载配置并设置随机种子
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        set_seed(config['training']['seed'])
    except Exception as e:
        logging.info(f"❌ Error loading config file: {e}")
        return

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir = os.path.join('runs', timestamp)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    vis_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    setup_logging(run_dir)
    logging.info("✅ Configuration loaded successfully.")
    logging.info(f"📂 All outputs for this run will be saved in: {run_dir}")

    # 2. 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Using device: {device}")

    # 3. 初始化数据集和数据加载器
    logging.info("\nInitializing dataset...")
    try:
        data_cfg = config['data']
        dataset_type = data_cfg.get('type', 'nyuv2').lower()
        img_size = tuple(data_cfg['img_size'])
        dataset_path = data_cfg.get('dataset_path')

        logging.info(f"📋 Dataset Type: {dataset_type}")
        logging.info(f"📂 Dataset Path: {dataset_path}")

        # === 数据集加载逻辑 ===
        if dataset_type == 'gta5_to_cityscapes':
            # ... (Sim-to-Real 逻辑保持不变，如果需要) ...
            train_path = data_cfg['train_dataset_path']
            val_path = data_cfg['val_dataset_path']
            train_dataset = GTA5Dataset(root_dir=train_path, img_size=img_size)
            val_dataset = CityscapesDataset(root_dir=val_path, split='val', img_size=img_size)
            # 兼容性引用
            full_dataset = train_dataset

        elif dataset_type == 'cityscapes':
            logging.info("🌍 Mode: Cityscapes (LibMTL format)")
            train_dataset = CityscapesDataset(root_dir=dataset_path, split='train', img_size=img_size)
            val_dataset = CityscapesDataset(root_dir=dataset_path, split='val', img_size=img_size)
            full_dataset = train_dataset

        elif dataset_type == 'nyuv2':
            logging.info("🏠 Mode: NYUv2 (LibMTL format - Folder based)")
            # [MODIFIED] 不再读取 HDF5，而是直接实例化 Train/Val Dataset
            # LibMTL 格式中，train 和 val 是分开的文件夹，通过 mode 参数控制
            train_dataset = NYUv2Dataset(root_dir=dataset_path, mode='train', img_size=img_size)
            val_dataset = NYUv2Dataset(root_dir=dataset_path, mode='val', img_size=img_size)
            full_dataset = train_dataset  # 仅用于获取属性，不影响逻辑

        else:
            raise ValueError(f"❌ Unsupported dataset type: '{dataset_type}'")

        # DataLoader 设置
        pin_memory = data_cfg.get('pin_memory', torch.cuda.is_available())

        train_loader = DataLoader(
            train_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory,
            drop_last=True  # 训练时丢弃最后一个不完整的 batch 有助于稳定 BN
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory
        )

        logging.info(f"📚 Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples.")

    except Exception as e:
        logging.info(f"❌ Error creating dataset/loaders: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 初始化模型
    logging.info("\nInitializing model...")
    model_type = config['model'].get('type', 'causal')
    base_lr = float(config['training']['learning_rate'])

    # 兼容性处理：为单任务/Baseline模型提供参数
    if model_type == 'raw_mtl':
        model = RawMTLModel(config['model'], config['data']).to(device)
        criterion = MTLLoss(config['losses'], use_uncertainty=(config['training'].get('strategy') == 'uncertainty')).to(
            device)
        # 简单优化器配置
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=config['training']['weight_decay'])

    elif model_type == 'single_task':
        model = SingleTaskModel(config['model'], config['data']).to(device)
        criterion = SingleTaskLoss(config['model']['active_task'], config['losses']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=config['training']['weight_decay'])

    else:
        # 默认 Causal Model
        model = CausalMTLModel(config['model'], config['data']).to(device)

        # 参数分组：Backbone vs Heads
        # 如果是 ResNetEncoder (wrapper)，它的参数在 model.encoder.backbone 里
        # 我们简单地按名称区分
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if 'encoder' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # LibMTL 默认配置: Adam, lr=1e-4, weight_decay=1e-5
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': base_lr},  # Backbone LR
            {'params': head_params, 'lr': base_lr * 10}  # Head LR 通常大一些 (可选，或者保持一致)
        ], lr=base_lr, weight_decay=config['training']['weight_decay'])

        criterion = AdaptiveCompositeLoss(config['losses']).to(device)

    logging.info(f"🔧 Optimizer: {config['training']['optimizer']}, LR: {base_lr}")

    # 5. 学习率调度器 (Scheduler)
    # 逻辑已移至 engine/trainer.py 中 _build_scheduler 内部处理，
    # 这里传 None 即可，trainer 会读取 config 自动构建
    scheduler = None

    # 6. 启动训练
    logging.info("\n----- Starting Training -----")
    if config['training'].get('enable_training', True):
        train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device, checkpoint_dir)
    else:
        logging.info("🏃 Training is disabled in config.")

    # 7. 实验性分析 (可选)
    exp_cfg = config.get('experiments', {})
    if exp_cfg.get('enable', False):
        logging.info("\n===== Running experiments =====")
        model.eval()
        run_all_experiments(model, val_loader, device)

    # 8. 可视化 (可选)
    logging.info("\n----- Running Final Visualizations -----")
    best_ckpt = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        generate_visual_reports(model, val_loader, device, save_dir=vis_dir, num_reports=5)

    if hasattr(train_dataset, "close"): train_dataset.close()
    if hasattr(val_dataset, "close"): val_dataset.close()

    logging.info("\n🎉 Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/resnet50_nyuv2.yaml')
    args = parser.parse_args()
    main(args.config)