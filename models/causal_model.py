import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# 请确保 building_blocks.py 在同级目录下，且包含这些类
from .building_blocks import ViTEncoder, ResNetEncoder, ResNetDecoderWithDeepSupervision
from .heads.albedo_head import AlbedoHead
from .heads.normal_head import NormalHead
from .heads.light_head import LightHead
from .layers.shading import shading_from_normals

# ==============================================================================
# 1. 基础组件 (ASPP, Refiners)
# ==============================================================================

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ZpSegRefiner(nn.Module):
    def __init__(self, c_in=256, out_classes=40, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, out_classes, 3, padding=1)
        )

    def forward(self, zp_proj, out_size=None):
        x = self.net(zp_proj)
        if out_size is not None and x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return self.alpha * x

class ZpDepthRefiner(nn.Module):
    def __init__(self, c_in=256, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, zp_proj):
        return self.alpha * self.net(zp_proj)

class SegDepthDecoder(nn.Module):
    """辅助用的简单解码器，用于 pred_depth_from_zp"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.output_channels = output_channels
        def _make_upsample_layer(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_c),
                nn.ReLU(inplace=True)
            )
        self.upsample1 = _make_upsample_layer(input_channels, 256)
        self.upsample2 = _make_upsample_layer(256, 128)
        self.upsample3 = _make_upsample_layer(128, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        return self.final_conv(x)

# ==============================================================================
# 2. 核心融合模块 (Cross Attention) - 必须定义在这里
# ==============================================================================

class CrossAttentionFusion(nn.Module):
    """
    [结构-风格融合]
    Query = 结构特征 (Main/Skip)
    Key/Value = 风格特征 (z_p)
    """
    def __init__(self, main_channels, style_channels, inter_channels=None):
        super().__init__()
        self.inter_channels = inter_channels if inter_channels else main_channels // 2

        # 1x1 Conv 投影
        self.query_conv = nn.Conv2d(main_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(style_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(style_channels, main_channels, kernel_size=1)

        # 初始为 0，防止破坏预训练特征
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_main, z_p):
        batch_size, C, H, W = x_main.size()

        # Q: [B, C_inter, HW] -> [B, HW, C_inter]
        proj_query = self.query_conv(x_main).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        # K: [B, C_inter, HW]
        proj_key = self.key_conv(z_p).view(batch_size, self.inter_channels, -1)

        # Attention Map: [B, HW, HW]
        # 注意显存：如果 H,W 很大(Layer1)，这里会比较吃显存。BS=32时注意。
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # V: [B, C, HW]
        proj_value = self.value_conv(z_p).view(batch_size, C, -1)

        # Out: [B, C, HW] -> [B, C, H, W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # Residual Connection
        out = self.gamma * out + x_main
        return out

# ==============================================================================
# 3. 核心解码器 (GatedSegDepthDecoder with Dual Attention)
# ==============================================================================

class GatedSegDepthDecoder(nn.Module):
    def __init__(self, main_in_channels: int, z_p_channels: int, out_channels: int,
                 low_level_in_channels: int = 256,
                 scale_cap: float = 1.0,
                 use_sigmoid=False):
        super().__init__()
        self.output_channels = out_channels

        # 1. 高层特征处理 (ASPP)
        aspp_out = 256
        self.aspp = ASPP(in_channels=main_in_channels, out_channels=aspp_out, atrous_rates=[12, 24, 36])

        # 2. High-Level Attention: 在语义层注入风格
        self.high_level_attention = CrossAttentionFusion(
            main_channels=aspp_out,
            style_channels=z_p_channels
        )

        # 3. Skip Connection 投影
        self.low_level_proj_channels = 48
        self.project_low_level = nn.Sequential(
            nn.Conv2d(low_level_in_channels, self.low_level_proj_channels, 1, bias=False),
            nn.GroupNorm(4, self.low_level_proj_channels),
            nn.ReLU(inplace=True)
        )

        # 4. Low-Level Attention: 在细节层注入风格 (Layer 1 特征在这里被增强)
        self.low_level_attention = CrossAttentionFusion(
            main_channels=self.low_level_proj_channels, # 48通道
            style_channels=z_p_channels,
            inter_channels=32 # 降维以节省显存
        )

        # 5. 预测头
        head_in_channels = aspp_out + self.low_level_proj_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in_channels, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, out_channels, 1)
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, main_feat, z_p_feat, low_level_feat=None, use_film: bool = True, detach_zp: bool = False):
        # 1. High Level
        x = self.aspp(main_feat)

        if use_film:
            if detach_zp: z_p_feat = z_p_feat.detach()
            # 融合增强后的风格
            x = self.high_level_attention(x, z_p_feat)

        # 2. Low Level (Skip Connection)
        if low_level_feat is not None:
            low = self.project_low_level(low_level_feat)

            if use_film:
                # 在细节层也进行 Cross-Attention
                low = self.low_level_attention(low, z_p_feat)

            # 上采样 High Level 并与 Low Level 拼接
            x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, low], dim=1)

        # 3. 输出
        out = self.head(x)

        # 上采样回原图
        scale_factor = 4 if (low_level_feat is not None) else 8
        out = F.interpolate(out, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out

# ==============================================================================
# 4. 主模型 (CausalMTLModel)
# ==============================================================================

class CausalMTLModel(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config
        self.data_config = data_config

        # --- Encoder ---
        encoder_name = model_config['encoder_name']
        if 'resnet' in encoder_name:
            # 使用 dilated ResNet
            self.encoder = ResNetEncoder(name=encoder_name, pretrained=model_config['pretrained'], dilated=True)

            # 将 ResNet 所有层投影到统一维度 (1024)
            target_dim = 1024
            self.resnet_adapters = nn.ModuleList([
                nn.Conv2d(in_c, target_dim, kernel_size=1) for in_c in self.encoder.feature_dims
            ])
            encoder_feature_dim = target_dim
            print(f"✅ Using ResNet50 Encoder with adapters -> {target_dim} dim")
        else:
            # ViT 兼容
            self.encoder = ViTEncoder(name=encoder_name, pretrained=model_config['pretrained'], img_size=data_config['img_size'][0])
            encoder_feature_dim = self.encoder.feature_dim
            self.resnet_adapters = None

        self.latent_dim_s = model_config['latent_dim_s']
        self.latent_dim_p = model_config['latent_dim_p']

        # ResNet 4层特征拼接作为主特征: 1024 * 4 = 4096
        combined_feature_dim = encoder_feature_dim * 4

        # --- Projectors ---
        self.projector_s = nn.Conv2d(combined_feature_dim, self.latent_dim_s, kernel_size=1)
        self.projector_p_seg = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)
        self.projector_p_depth = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)
        self.projector_p_normal = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)

        # --- Intermediate Projections (统一到 256) ---
        PROJ_CHANNELS = 256
        self.proj_f = nn.Conv2d(combined_feature_dim, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_s = nn.Conv2d(self.latent_dim_s, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_seg = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_depth = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_normal = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)

        # ======================================================================
        # [NEW] Cross-Task Style Injection Adapters (简单版)
        # ======================================================================
        # Seg -> Depth
        self.cross_seg_to_depth = nn.Sequential(
            nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, 1)
        )
        # Seg -> Normal
        self.cross_seg_to_normal = nn.Sequential(
            nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, 1)
        )
        self.cross_depth_to_normal = nn.Sequential(
            nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, 1)  # 甚至可以用 3x3 卷积来模拟求导过程
        )


        # 融合门控 (可学习)
        self.fusion_gate_depth = nn.Parameter(torch.tensor(0.0))
        self.fusion_gate_normal = nn.Parameter(torch.tensor(0.0))
        self.fusion_gate_depth_to_normal = nn.Parameter(torch.tensor(0.0))

        # --- Residual Refiners (Refiner 使用原始风格) ---
        self.zp_depth_refiner = ZpDepthRefiner(c_in=PROJ_CHANNELS, alpha=0.1)
        self.zp_normal_refiner = nn.Sequential(
            nn.Conv2d(PROJ_CHANNELS, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        num_seg_classes = model_config.get('num_seg_classes', 40)
        self.zp_seg_refiner = ZpSegRefiner(c_in=PROJ_CHANNELS, out_classes=num_seg_classes, alpha=0.2)

        # --- Decoders (使用 CrossAttentionFusion) ---
        decoder_main_dim = PROJ_CHANNELS * 2

        self.predictor_seg = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim,
            z_p_channels=PROJ_CHANNELS,
            out_channels=num_seg_classes
        )
        self.predictor_depth = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim,
            z_p_channels=PROJ_CHANNELS,
            out_channels=1,
            use_sigmoid=False
        )
        self.predictor_normal = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim,
            z_p_channels=PROJ_CHANNELS,
            out_channels=3
        )

        # --- Auxiliary & Decomp Heads ---
        self.decoder_zp_depth = SegDepthDecoder(self.latent_dim_p, 1)
        self.decoder_zp_normal = SegDepthDecoder(self.latent_dim_p, 3)
        self.decoder_geom = ResNetDecoderWithDeepSupervision(self.latent_dim_s, 1, tuple(data_config['img_size']))
        self.decoder_app = ResNetDecoderWithDeepSupervision(self.latent_dim_p, 3, tuple(data_config['img_size']))
        self.final_app_activation = nn.Sigmoid()

        self.albedo_head = AlbedoHead(self.latent_dim_p, hidden=128)
        self.normal_head = NormalHead(self.latent_dim_s, hidden=128)
        self.light_head = LightHead(in_ch=encoder_feature_dim)
        self._target_size = tuple(data_config['img_size'])

    def extract_features(self, x):
        raw_features = self.encoder(x)
        if self.resnet_adapters is not None:
            features = []
            target_h, target_w = raw_features[2].shape[-2:] # Align to Layer 3
            for i, feat in enumerate(raw_features):
                feat = self.resnet_adapters[i](feat)
                if feat.shape[-2:] != (target_h, target_w):
                    feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                features.append(feat)
        else:
            features = raw_features
        combined_feat = torch.cat(features, dim=1)
        last_feat = features[-1]
        h = last_feat.mean(dim=[2, 3])
        return combined_feat, h, features, raw_features

    def forward(self, x, stage: int = 2,
                override_zs_map=None, override_zp_seg_map=None, override_zp_depth_map=None):

        # 1. 特征提取
        # raw_features[0] 是 Layer 1 (stride 4)，用于 Low-Level Cross Attention
        combined_feat, h, features, raw_features = self.extract_features(x)
        low_level_feat = raw_features[0]
        H_map, W_map = combined_feat.shape[2], combined_feat.shape[3]
        grid_size = 4

        # 2. 潜变量计算
        if override_zs_map is None:
            z_s_map = self.projector_s(combined_feat)
        else:
            z_s_map = override_zs_map

        feat_grid = F.adaptive_avg_pool2d(combined_feat, (grid_size, grid_size))
        z_p_seg_grid = self.projector_p_seg(feat_grid)
        z_p_depth_grid = self.projector_p_depth(feat_grid)
        z_p_normal_grid = self.projector_p_normal(feat_grid)

        if override_zp_seg_map is None:
            z_p_seg_map = F.interpolate(z_p_seg_grid, size=(H_map, W_map), mode='nearest')
        else:
            z_p_seg_map = override_zp_seg_map

        if override_zp_depth_map is None:
            z_p_depth_map = F.interpolate(z_p_depth_grid, size=(H_map, W_map), mode='nearest')
        else:
            z_p_depth_map = override_zp_depth_map

        z_p_normal_map = F.interpolate(z_p_normal_grid, size=(H_map, W_map), mode='nearest')

        # Global Pooling
        z_s = z_s_map.mean(dim=[2, 3])
        z_p_seg = z_p_seg_grid.mean(dim=[2, 3])
        z_p_depth = z_p_depth_grid.mean(dim=[2, 3])
        z_p_normal = z_p_normal_grid.mean(dim=[2, 3])

        # 3. 统一投影 (256维)
        f_proj = self.proj_f(combined_feat)
        zs_proj = self.proj_z_s(z_s_map)
        zp_seg_proj = self.proj_z_p_seg(z_p_seg_map)
        zp_depth_proj = self.proj_z_p_depth(z_p_depth_map)
        zp_normal_proj = self.proj_z_p_normal(z_p_normal_map)

        use_film = (stage >= 2)
        use_zp_residual = (stage >= 2)

        # ======================================================================
        # [Step 4] 跨任务风格注入 (Cross-Task Style Injection)
        # ======================================================================

        # 1. 计算 Seg 带来的增益
        feat_seg_for_depth = self.cross_seg_to_depth(zp_seg_proj)
        feat_seg_for_normal = self.cross_seg_to_normal(zp_seg_proj)


        # 2. 融合 (Fusion)
        # Seg 不变
        zp_seg_fused = zp_seg_proj
        # Depth 和 Normal 接收 Seg 的信息
        zp_depth_fused = zp_depth_proj + torch.sigmoid(self.fusion_gate_depth) * feat_seg_for_depth
        #zp_normal_fused = zp_normal_proj + torch.sigmoid(self.fusion_gate_normal) * feat_seg_for_normal
        feat_depth_for_normal = self.cross_depth_to_normal(zp_depth_fused)
        zp_normal_fused = zp_normal_proj + \
                          torch.sigmoid(self.fusion_gate_normal) * feat_seg_for_normal + \
                          torch.sigmoid(self.fusion_gate_depth_to_normal) * feat_depth_for_normal
        # Decoder 主输入：结合了多层 ResNet 特征的 f_proj 和 几何 z_s
        task_input_feat = torch.cat([f_proj, zs_proj], dim=1)

        # Checkpoint 包装器
        def run_decoder(decoder_module, main, zp, low_level_feat, film, detach):
            return decoder_module(main, zp, low_level_feat, film, detach)

        # ======================================================================
        # [Step 5] 下游任务解码
        # ======================================================================

        # === Task 1: Segmentation ===
        if self.training and task_input_feat.requires_grad:
            pred_seg_main = checkpoint(run_decoder, self.predictor_seg,
                                     task_input_feat,
                                     zp_seg_fused,  # 使用融合后的风格
                                     low_level_feat, use_film, False, use_reentrant=False)
        else:
            pred_seg_main = self.predictor_seg(task_input_feat, zp_seg_fused,
                                             low_level_feat=low_level_feat, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            # Refiner 使用原始风格 (zp_seg_proj)
            seg_res = self.zp_seg_refiner(zp_seg_proj, out_size=pred_seg_main.shape[-2:])
            pred_seg = pred_seg_main + seg_res
        else:
            pred_seg = pred_seg_main

        # === Task 2: Depth ===
        if self.training and task_input_feat.requires_grad:
            pred_depth_main = checkpoint(run_decoder, self.predictor_depth,
                                       task_input_feat,
                                       zp_depth_fused, # 使用融合后的风格
                                       low_level_feat, use_film, False, use_reentrant=False)
        else:
            pred_depth_main = self.predictor_depth(task_input_feat, zp_depth_fused,
                                                 low_level_feat=low_level_feat, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            zp_residual = self.zp_depth_refiner(zp_depth_proj)
            zp_residual = F.interpolate(zp_residual, size=pred_depth_main.shape[-2:], mode='bilinear', align_corners=False)
            pred_depth = pred_depth_main + zp_residual
        else:
            pred_depth = pred_depth_main

        # === Task 3: Normal ===
        if self.training and task_input_feat.requires_grad:
            pred_normal_main = checkpoint(run_decoder, self.predictor_normal,
                                        task_input_feat,
                                        zp_normal_fused, # 使用融合后的风格
                                        low_level_feat, use_film, False, use_reentrant=False)
        else:
            pred_normal_main = self.predictor_normal(task_input_feat, zp_normal_fused,
                                                   low_level_feat=low_level_feat, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            if hasattr(self, 'zp_normal_refiner'):
                normal_res = self.zp_normal_refiner(zp_normal_proj) * 0.1
                normal_res = F.interpolate(normal_res, size=pred_normal_main.shape[-2:], mode='bilinear', align_corners=False)
                pred_normal = pred_normal_main + normal_res
            else:
                pred_normal = pred_normal_main
        else:
            pred_normal = pred_normal_main

        pred_normal = F.normalize(pred_normal, p=2, dim=1)

        # ---------- 辅助与重构 ----------
        pred_depth_from_zp = self.decoder_zp_depth(z_p_depth_map) if use_zp_residual else torch.zeros_like(pred_depth)

        if self.training and z_s_map.requires_grad:
            recon_geom_final, recon_geom_aux = checkpoint(self.decoder_geom, z_s_map, use_reentrant=False)
            recon_app_final_logits, recon_app_aux_logits = checkpoint(self.decoder_app, z_p_seg_map, use_reentrant=False)
        else:
            recon_geom_final, recon_geom_aux = self.decoder_geom(z_s_map)
            recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_p_seg_map)

        recon_app_final = self.final_app_activation(recon_app_final_logits)
        recon_app_aux = self.final_app_activation(recon_app_aux_logits)

        A = self.albedo_head(z_p_seg_map)
        N = self.normal_head(z_s_map)
        L = self.light_head(h)
        S = shading_from_normals(N, L)

        target_size = self._target_size
        if A.shape[-2:] != target_size:
            A = F.interpolate(A, size=target_size, mode='bilinear', align_corners=False)
            N = F.interpolate(N, size=target_size, mode='bilinear', align_corners=False)
            S = F.interpolate(S, size=target_size, mode='bilinear', align_corners=False)
        I_hat = torch.clamp(A * S, 0.0, 1.0)

        outputs = {
            'z_s': z_s, 'z_p_seg': z_p_seg, 'z_p_depth': z_p_depth, 'z_p_normal': z_p_normal,
            'z_s_map': z_s_map, 'z_p_seg_map': z_p_seg_map, 'z_p_depth_map': z_p_depth_map,
            'pred_seg': pred_seg, 'pred_depth': pred_depth, 'pred_normal': pred_normal, 'normals': pred_normal,
            'decomposition_normal': N, 'albedo': A, 'shading': S, 'sh_coeff': L, 'recon_decomp': I_hat,
            'pred_depth_from_zp': pred_depth_from_zp,
            'recon_geom': recon_geom_final, 'recon_app': recon_app_final,
            'recon_geom_aux': recon_geom_aux, 'recon_app_aux': recon_app_aux,
            'stage': stage,
        }
        return outputs