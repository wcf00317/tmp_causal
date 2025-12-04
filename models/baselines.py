import torch
import torch.nn as nn
from .building_blocks import ViTEncoder, MLP
# 复用你在 causal_model.py 中定义的普通解码器
from .causal_model import SegDepthDecoder
import torch.nn.functional as F


class RawMTLModel(nn.Module):
    """
    Standard Hard Parameter Sharing Multi-Task Learning (Raw MTL).

    【工程适配版】
    包含特定的属性别名和占位符，以兼容 Causal MTL 项目现有的 trainer.py 和 evaluator.py，
    无需修改公共训练代码。
    """

    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config

        # 1. Shared Backbone (Same as Ours)
        self.encoder = ViTEncoder(
            name=model_config['encoder_name'],
            pretrained=model_config['pretrained'],
            img_size=data_config['img_size'][0]
        )
        encoder_feature_dim = self.encoder.feature_dim

        # 2. Shared Projection (Bottleneck)
        combined_feature_dim = encoder_feature_dim * 4
        self.shared_dim = 512
        self.shared_proj = nn.Conv2d(combined_feature_dim, self.shared_dim, kernel_size=1)

        # 3. Task Heads

        # --- Scene Head ---
        self.num_scene_classes = model_config['num_scene_classes']
        # 注意：为了兼容 evaluator 读取 .out_features，我们把 MLP 独立出来
        self.scene_mlp = MLP(self.shared_dim, self.num_scene_classes, hidden_dim=256)

        # --- Seg Head ---
        self.num_seg_classes = 40
        self.seg_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=self.num_seg_classes)

        # --- Depth Head ---
        self.depth_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=1)

        # =========================================================
        #   兼容性适配层 (Compatibility Layer) - 关键修改
        # =========================================================

        # 1. [For Evaluator] 提供别名，因为 evaluator.py 访问的是 predictor_xxx
        self.predictor_seg = self.seg_head
        self.predictor_depth = self.depth_head
        self.predictor_scene = self.scene_mlp  # evaluator 读取 .out_features

        # 2. [For Trainer] 提供占位符，欺骗 _switch_stage_freeze
        # trainer.py 会尝试访问这些属性并设置梯度。设为 None 可安全跳过。
        self.projector_p_seg = None
        self.projector_p_depth = None
        self.proj_z_p_seg = None
        self.proj_z_p_depth = None
        self.zp_seg_refiner = None
        self.zp_depth_refiner = None
        self.decoder_zp_depth = None

        # 3. [For GradNorm] 预留共享层接口
        self.layer_to_norm = self.shared_proj

    def forward(self, x, stage=None):
        # stage 参数被保留以兼容 trainer 接口调用，但此处忽略

        # 1. Encoder
        features = self.encoder(x)

        # 2. Aggregation
        combined_feat = torch.cat(features, dim=1)
        shared_feat = self.shared_proj(combined_feat)

        # 3. Task Predictions
        pred_seg = self.seg_head(shared_feat)
        pred_depth = self.depth_head(shared_feat)

        # Scene: Pool -> Flatten -> MLP
        h = F.adaptive_avg_pool2d(shared_feat, (1, 1)).flatten(1)
        pred_scene = self.scene_mlp(h)

        return {
            'pred_seg': pred_seg,
            'pred_depth': pred_depth,
            'pred_scene': pred_scene,
            # 'z_s', 'z_p' 等不需要返回，因为使用的是 MTLLoss 而非 CompositeLoss
        }

    def get_last_shared_layer(self):
        return self.layer_to_norm