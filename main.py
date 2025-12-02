import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os,argparse
from data_utils.nyuv2_dataset import NYUv2Dataset
import h5py,logging
# --- 必改1: 确认模块/文件名一致性 ---
# 请确保下面的导入路径与您项目中models/和losses/下的文件名完全一致
# 例如，如果文件名是 causal_models.py (复数)，则应改为:
# from models.causal_models import CausalMTLModel
from models.causal_model import CausalMTLModel
from losses.composite_loss import CompositeLoss,AdaptiveCompositeLoss
from datetime import datetime
from engine.trainer import train
from engine.visualizer import generate_visual_reports
from utils.general_utils import set_seed,setup_logging
from torch.utils.data import Subset

def main(config_path):
    """
    项目主函数（最终鲁棒版），集成了所有稳定性与可复现性修正。
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
    logging.info("🧩 Full configuration:\n" + yaml.dump(config, sort_keys=False, allow_unicode=True))
    logging.info(f"🌱 Random seed set to {config['training']['seed']}")
    logging.info(f"📂 All outputs for this run will be saved in: {run_dir}")
    # 2. 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Using device: {device}")

    # 3. 初始化数据集和数据加载器
    logging.info("\nInitializing dataset...")
    try:
        # 确保我们导入的Dataset类名与文件名中的类名一致

        logging.info("Pre-loading scene metadata from HDF5 file...")
        with h5py.File(config['data']['dataset_path'], 'r') as db:
            scene_type_refs = db['sceneTypes']  # shape is (1, 1449)
            scene_types_list = []

            for i in range(scene_type_refs.shape[1]):
                ref = scene_type_refs[0, i]
                scene_str = "".join(chr(c[0]) for c in db[ref])
                scene_types_list.append(scene_str)
        full_dataset = NYUv2Dataset(
            mat_file_path=config['data']['dataset_path'],
            img_size=tuple(config['data']['img_size']),scene_types_list=scene_types_list
        )

        # --- 必改7: 保证随机划分的可复现性 ---
        g = torch.Generator()
        g.manual_seed(config['training']['seed'])
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

        # --- 必改4: 根据设备情况设置pin_memory ---
        pin_memory = config['data'].get('pin_memory', torch.cuda.is_available())
        logging.info(f"💡 pin_memory set to: {pin_memory}")

        train_loader = DataLoader(
            train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
            num_workers=config['data']['num_workers'], pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['data']['batch_size'], shuffle=False,
            num_workers=config['data']['num_workers'], pin_memory=pin_memory
        )
        logging.info(f"📚 Dataset split into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    except Exception as e:
        logging.info(f"❌ Error creating dataset/loaders: {e}")
        return

    # 4. 初始化模型、优化器、调度器和损失函数
    logging.info("\nInitializing model and training components...")
    model = CausalMTLModel(
        model_config=config['model'],
        data_config=config['data']
    ).to(device)
    base_lr = float(config['training']['learning_rate'])  # 例如 1e-5

    # 2. 分离参数
    # 获取 encoder 的参数内存地址 ID
    encoder_params_ids = list(map(id, model.encoder.parameters()))

    # 过滤参数：不在 encoder 中的就是 head/decoder 参数
    backbone_params = model.encoder.parameters()
    head_params = [p for n, p in model.named_parameters() if id(p) not in encoder_params_ids]

    print(f"🔧 Optimizer setup: Backbone LR={base_lr}, Head/Decoder LR={base_lr * 10.0}")

    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': base_lr},  # 预训练部分保持小 LR
            {'params': head_params, 'lr': base_lr * 10.0}  # 新增部分放大 10 倍 LR
        ], weight_decay=config['training']['weight_decay'])
    else:
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': base_lr},
            {'params': head_params, 'lr': base_lr * 10.0}
        ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = AdaptiveCompositeLoss(loss_weights=config['losses']).to(device)
    logging.info("⚙️ Model, optimizer, scheduler, and loss function are ready.")
    # --- 必改6: 需确认CompositeLoss的返回接口与trainer兼容 ---
    # 我们已在上一版中统一 CompositeLoss 返回 (total_loss, loss_dict)，
    # 并且 trainer.py 中的代码已兼容此格式。

    # 6. 启动训练流程
    logging.info("\n----- Starting Training -----")
    if config['training'].get('enable_training', True):
        train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device, checkpoint_dir)
    else:
        logging.info("🏃 Training is disabled in the config. Skipping.")
    from engine.experiments import run_all_experiments
    exp_cfg = config.get('experiments', {})
    if exp_cfg.get('enable', False):
        logging.info("\n===== Running falsification experiments =====")
        model.eval()
        _ = run_all_experiments(
            model, val_loader, device,
            max_batches_swap=int(exp_cfg.get('max_batches_swap', 8)),
            max_batches_inv=int(exp_cfg.get('max_batches_inv', 8)),
            max_batches_cross=int(exp_cfg.get('max_batches_cross', 20)),
        )

    # 7. 最终可视化与分析
    logging.info("\n----- Running Final Visualizations & Analysis -----")
    best_checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_checkpoint_path):
        logging.info(f"🔍 Loading best model from {best_checkpoint_path} for visualization...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("✅ Loaded checkpoint state_dict successfully.")
        except RuntimeError as e:
            logging.info(f"⚠️ Warning: state_dict load error: {e}. Trying non-strict load.")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
        vis_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
        # --- 修改: 将新创建的 vis_dir 传递给可视化函数 ---
        generate_visual_reports(model, vis_loader, device, save_dir=vis_dir,num_reports=5)
    else:
        logging.info(f"⚠️ Could not find best model checkpoint at '{best_checkpoint_path}'. Skipping final analysis.")
    # --- 必改3: 安全地调用close方法 ---
    if hasattr(full_dataset, "close") and callable(full_dataset.close):
        logging.info("Closing dataset handler...")
        full_dataset.close()

    logging.info("\n🎉 Project execution finished.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Causal MTL Training")
    # 添加 --config 参数，如果没有提供，默认使用 base_full_model.yaml
    parser.add_argument('--config', type=str, default='configs/base_full_model.yaml', help='Path to the config file')

    args = parser.parse_args()

    # 使用命令行传入的参数
    config_file = args.config

    print(f"🚀 Loading configuration from: {config_file}")  # 打印一下以确认

    # 强烈建议: 在正式运行前，用一小部分数据进行冒烟测试(smoke test)
    main(config_file)