import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os,argparse
from data_utils.nyuv2_dataset import NYUv2Dataset
from data_utils.gta5_dataset import GTA5Dataset
import h5py,logging
from models.causal_model import CausalMTLModel
from losses.composite_loss import AdaptiveCompositeLoss
from losses.mtl_loss import MTLLoss
from models.baselines import RawMTLModel, SingleTaskModel
from losses.single_task_loss import SingleTaskLoss
from data_utils.cityscapes_dataset import CityscapesDataset
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
        data_cfg = config['data']
        dataset_type = data_cfg.get('type', 'nyuv2').lower()  # 获取类型，默认为 'nyuv2' 以兼容旧配置
        #dataset_path = data_cfg['dataset_path']
        img_size = tuple(data_cfg['img_size'])

        logging.info(f"📋 Dataset Type Configuration: {dataset_type}")
        if dataset_type == 'gta5_to_cityscapes':
            logging.info("🌍 Mode: Sim-to-Real (Train on GTA5, Val on Cityscapes)")
            # 1. 加载 GTA5 Train
            train_path = data_cfg['train_dataset_path']
            logging.info(f"   -> Loading Source Train: {train_path}")
            train_dataset = GTA5Dataset(root_dir=train_path, img_size=img_size)

            # 2. 加载 Cityscapes Val (Target)
            target_val_path = data_cfg['val_dataset_path']
            logging.info(f"   -> Loading Target Val: {target_val_path}")
            val_dataset = CityscapesDataset(root_dir=target_val_path, split='val', img_size=img_size)

            # 3. 加载 GTA5 Val (Source Held-out)
            source_val_path = data_cfg.get('source_val_path')
            if source_val_path and os.path.exists(source_val_path):
                logging.info(f"   -> Loading Source Val: {source_val_path}")
                source_val_dataset = GTA5Dataset(root_dir=source_val_path, img_size=img_size)
            else:
                logging.warning(f"⚠️ Source val path not found or empty: {source_val_path}")

            # 这里的 full_dataset 只是为了兼容后面的一行代码，可以指向 train_dataset
            full_dataset = train_dataset
        # === 显式分支逻辑 ===
        elif dataset_type == 'cityscapes':
            dataset_path = data_cfg['dataset_path']
            logging.info(f"📂 Loading CityscapesDataset from: {dataset_path}")
            full_dataset = CityscapesDataset(
                root_dir=dataset_path,
                split='train',
                img_size=img_size
            )

        elif dataset_type == 'nyuv2':
            dataset_path = data_cfg['dataset_path']
            logging.info(f"📄 Loading NYUv2Dataset (HDF5) from: {dataset_path}")

            # NYUv2 特有的预读取逻辑
            logging.info("Pre-loading scene metadata from HDF5 file...")
            with h5py.File(dataset_path, 'r') as db:
                scene_type_refs = db['sceneTypes']
                scene_types_list = []
                for i in range(scene_type_refs.shape[1]):
                    ref = scene_type_refs[0, i]
                    scene_str = "".join(chr(c[0]) for c in db[ref])
                    scene_types_list.append(scene_str)

            full_dataset = NYUv2Dataset(
                mat_file_path=dataset_path,
                img_size=img_size,
                scene_types_list=scene_types_list
            )

        else:
            # 遇到不支持的类型直接报错，而不是瞎猜
            raise ValueError(f"❌ Unsupported dataset type: '{dataset_type}'. "
                             f"Supported types are: ['cityscapes', 'nyuv2']")


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
    model_type = config['model'].get('type', 'causal')
    base_lr = float(config['training']['learning_rate'])
    if model_type == 'raw_mtl':
        logging.info("🏗️ Building Baseline: Raw MTL Model")
        model = RawMTLModel(
            model_config=config['model'],
            data_config=config['data']
        ).to(device)

        # Baseline 使用通用 Loss
        strategy = config['training'].get('strategy', 'fixed')
        use_uncertainty = (strategy == 'uncertainty')
        logging.info(f" Using Loss Strategy: {strategy}")
        criterion = MTLLoss(loss_weights=config['losses'], use_uncertainty=use_uncertainty).to(device)
        optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': base_lr},
            {'params': model.seg_head.parameters(), 'lr': base_lr * 10},
            {'params': model.depth_head.parameters(), 'lr': base_lr * 10},
            {'params': model.scene_mlp.parameters(), 'lr': base_lr * 10},
            {'params': model.shared_proj.parameters(), 'lr': base_lr * 10},
            # 如果使用 uncertainty，loss 中也有参数需要优化
            {'params': criterion.parameters(), 'lr': base_lr}
        ], weight_decay=config['training']['weight_decay'])
        scheduler = None
    elif model_type == 'single_task':
        logging.info(f"Building Single-Task Baseline: {config['model']['active_task']}")
        model = SingleTaskModel(
            model_config=config['model'],
            data_config=config['data']
        ).to(device)

        criterion = SingleTaskLoss(
            active_task=config['model']['active_task'],
            loss_weights=config['losses']
        ).to(device)

        # 只将 encoder, shared_proj 和当前任务的 head 放入优化器
        # 我们可以简单地用 model.parameters()，因为其他 head 根本没有被实例化
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=config['training']['weight_decay'])
        scheduler = None
    else:
        logging.info("Building Our Causal MTL Model")
        model = CausalMTLModel(
            model_config=config['model'],
            data_config=config['data']
        ).to(device)

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