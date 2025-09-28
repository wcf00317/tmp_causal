import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

# --- 必改1: 确认模块/文件名一致性 ---
# 请确保下面的导入路径与您项目中models/和losses/下的文件名完全一致
# 例如，如果文件名是 causal_models.py (复数)，则应改为:
# from models.causal_models import CausalMTLModel
from models.causal_model import CausalMTLModel
from losses.composite_loss import CompositeLoss

from engine.trainer import train
from engine.visualizer import generate_visual_reports
from utils.general_utils import set_seed


def main(config_path):
    """
    项目主函数（最终鲁棒版），集成了所有稳定性与可复现性修正。
    """
    # 1. 加载配置并设置随机种子
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully.")
        set_seed(config['training']['seed'])
        print(f"🌱 Random seed set to {config['training']['seed']}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return

    # 2. 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 3. 初始化数据集和数据加载器
    print("\nInitializing dataset...")
    try:
        # 确保我们导入的Dataset类名与文件名中的类名一致
        from data_utils.nyuv2_dataset import NYUv2Dataset
        full_dataset = NYUv2Dataset(
            mat_file_path=config['data']['dataset_path'],
            img_size=tuple(config['data']['img_size'])
        )

        # --- 必改7: 保证随机划分的可复现性 ---
        g = torch.Generator()
        g.manual_seed(config['training']['seed'])
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

        # --- 必改4: 根据设备情况设置pin_memory ---
        pin_memory = config['data'].get('pin_memory', torch.cuda.is_available())
        print(f"💡 pin_memory set to: {pin_memory}")

        train_loader = DataLoader(
            train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
            num_workers=config['data']['num_workers'], pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['data']['batch_size'], shuffle=False,
            num_workers=config['data']['num_workers'], pin_memory=pin_memory
        )
        print(f"📚 Dataset split into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    except Exception as e:
        print(f"❌ Error creating dataset/loaders: {e}")
        return

    # 4. 初始化模型、优化器、调度器和损失函数
    print("\nInitializing model and training components...")
    model = CausalMTLModel(
        model_config=config['model'],
        data_config=config['data']
    ).to(device)

    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = CompositeLoss(loss_weights=config['losses'])
    print("⚙️ Model, optimizer, scheduler, and loss function are ready.")
    # --- 必改6: 需确认CompositeLoss的返回接口与trainer兼容 ---
    # 我们已在上一版中统一 CompositeLoss 返回 (total_loss, loss_dict)，
    # 并且 trainer.py 中的代码已兼容此格式。

    # 6. 启动训练流程
    print("\n----- Starting Training -----")
    if config['training'].get('enable_training', True):
        train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device)
    else:
        print("🏃 Training is disabled in the config. Skipping.")

    # 7. 最终可视化与分析
    print("\n----- Running Final Visualizations & Analysis -----")
    best_checkpoint_path = 'checkpoints/model_best.pth.tar'
    if os.path.exists(best_checkpoint_path):
        print(f"🔍 Loading best model from {best_checkpoint_path} for visualization...")

        # --- 必改2: 加载checkpoint时使用map_location ---
        checkpoint = torch.load(best_checkpoint_path, map_location=device)

        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("✅ Loaded checkpoint state_dict successfully.")
        except RuntimeError as e:
            # 增加鲁棒性，如果key不完全匹配（例如在多卡和单卡模型间切换），尝试非严格加载
            print(f"⚠️ Warning: state_dict load error: {e}. Trying non-strict load.")
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        # 强烈建议: 加载后显式设置eval模式
        model.eval()

        # --- 必改5: 确保vis_loader能提供至少两个不同batch ---
        # 我们已在visualizer中修复了采样逻辑，这里确保loader配置合理即可
        vis_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

        generate_visual_reports(model, vis_loader, device, save_dir="visualizations_final")
    else:
        print(f"⚠️ Could not find best model checkpoint at '{best_checkpoint_path}'. Skipping final analysis.")

    # --- 必改3: 安全地调用close方法 ---
    if hasattr(full_dataset, "close") and callable(full_dataset.close):
        print("Closing dataset handler...")
        full_dataset.close()

    print("\n🎉 Project execution finished.")


if __name__ == '__main__':
    config_file = 'configs/base_full_model.yaml'
    # 强烈建议: 在正式运行前，用一小部分数据进行冒烟测试(smoke test)
    # 例如，可以在config中设置一个'debug_subset_size'参数，并在Dataset中实现只加载少量数据
    main(config_file)