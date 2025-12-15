import torch
import torch.nn as nn
import torch.nn.functional as F

# [MODIFIED] 引入 ResNetEncoder
from .building_blocks import ViTEncoder, ResNetEncoder, MLP, ResNetDecoderWithDeepSupervision
from .heads.albedo_head import AlbedoHead
from .heads.normal_head import NormalHead
from .heads.light_head import LightHead
from .layers.shading import shading_from_normals
from torch.utils.checkpoint import checkpoint


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

class GatedFiLM(nn.Module):
    """用 z_p 对主特征做通道级缩放/平移；scale_cap 限制 z_p 的影响力。"""

    def __init__(self, c_main: int, c_zp: int, scale_cap: float = 1.0):
        super().__init__()
        self.scale_cap = scale_cap
        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,c_zp,H,W] -> [B,c_zp,1,1]
            nn.Conv2d(c_zp, c_main * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_main * 2, c_main * 2, 1),
        )

    def forward(self, h_main, z_p):
        gamma_beta = self.film(z_p)  # [B,2*c_main,1,1]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # 各 [B,c_main,1,1]
        gamma = torch.tanh(gamma) * self.scale_cap
        beta = torch.tanh(beta) * self.scale_cap
        return h_main * (1 + gamma) + beta


class ZpSegRefiner(nn.Module):
    """
    用 z_p_seg 生成一个很小的分割残差（每类一个通道），加到主分割 logits 上。
    alpha 建议 0.2 左右，避免 z_p 抢权。
    """

    def __init__(self, c_in=256, out_classes=40, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, out_classes, 3, padding=1)
        )

    def forward(self, zp_proj, out_size=None):
        x = self.net(zp_proj)  # [B,out_classes,14,14]
        if out_size is not None and x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return self.alpha * x


class ZpDepthRefiner(nn.Module):
    """
    用 z_p_depth 的特征生成一个很小的残差（alpha很小），加到主深度上。
    这样 z_s 主导结构，z_p 负责补细节，不会喧宾夺主。
    """

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


class GatedSegDepthDecoder(nn.Module):
    def __init__(self, main_in_channels: int, z_p_channels: int, out_channels: int, scale_cap: float = 1.0,
                 use_sigmoid=False):
        super().__init__()
        self.output_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # [1] ASPP 放在最前面，负责降维和提取上下文
        # 无论输入 main_in_channels 是多少 (1024/3072)，ASPP 都把它压到 256
        aspp_out = 256
        self.aspp = ASPP(in_channels=main_in_channels, out_channels=aspp_out, atrous_rates=[12, 24, 36])

        # [2] 核心融合层 (Gated FiLM)
        # 我们只需要做一次高质量的融合，不需要做4次
        self.film_fusion = GatedFiLM(aspp_out, z_p_channels, scale_cap)

        # [3] 最终预测头 (轻量化)
        # 包含 3x3 卷积整理特征 -> GN -> ReLU -> 1x1 输出
        self.head = nn.Sequential(
            nn.Conv2d(aspp_out, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),  # [关键] 使用 GroupNorm 防止 BS=2 崩溃
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, main_feat, z_p_feat, use_film: bool = True, detach_zp: bool = False):
        # 1. 提取特征 (H/8)
        x = self.aspp(main_feat)

        # 2. 因果干预 (融合 z_p)
        if use_film:
            x = self.film_fusion(x, z_p_feat)

        # 3. 生成预测 (H/8)
        out = self.head(x)

        # 4. 直接上采样回原图 (H) - 像 LibMTL 一样
        # scale_factor=8 对应 ResNet OS=8
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out

class SegDepthDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.output_channels = output_channels

        # 内部上采样块，使用老师推荐的 interpolate + conv 结构
        def _make_upsample_layer(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_c),
                nn.ReLU(inplace=True)
            )

        # 上采样路径
        self.upsample1 = _make_upsample_layer(input_channels, 256)  # 14x14 -> 28x28
        self.upsample2 = _make_upsample_layer(256, 128)  # 28x28 -> 56x56
        self.upsample3 = _make_upsample_layer(128, 64)  # 56x56 -> 112x112
        self.upsample4 = _make_upsample_layer(64, 32)  # 112x112 -> 224x224

        # 最终的预测层
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        return self.final_conv(x)


class CausalMTLModel(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config
        self.data_config = data_config
        self.zp_depth_gain = nn.Parameter(torch.tensor(0.0))

        # ============================================================
        # [MODIFIED] Encoder Initialization with Adapter Support
        # ============================================================
        encoder_name = model_config['encoder_name']

        if 'resnet' in encoder_name:
            # ResNet Path
            self.encoder = ResNetEncoder(
                name=encoder_name,
                pretrained=model_config['pretrained'],
                dilated=True  # LibMTL 默认使用 resnet_dilated
            )
            # ResNet 输出是 [256, 512, 1024, 2048]
            # 我们需要把它们全部投影到一个统一的维度 (e.g. 768) 以模拟 ViT
            target_dim = 256
            self.resnet_adapters = nn.ModuleList([
                nn.Conv2d(in_c, target_dim, kernel_size=1)
                for in_c in self.encoder.feature_dims
            ])
            encoder_feature_dim = target_dim
            print(f"✅ Using ResNet50 Encoder with adapters -> {target_dim} dim")
        else:
            # ViT Path (原逻辑)
            self.encoder = ViTEncoder(
                name=encoder_name,
                pretrained=model_config['pretrained'],
                img_size=data_config['img_size'][0]
            )
            encoder_feature_dim = self.encoder.feature_dim
            self.resnet_adapters = None

        # Latent dims
        self.latent_dim_s = model_config['latent_dim_s']
        self.latent_dim_p = model_config['latent_dim_p']

        # --- 修改：投影层输入维度变大 (Concatenation of 4 layers) ---
        combined_feature_dim = encoder_feature_dim * 4

        # Projectors (输入是 combined_feature_dim)
        self.projector_s = nn.Conv2d(combined_feature_dim, self.latent_dim_s, kernel_size=1)
        self.projector_p_seg = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)
        self.projector_p_depth = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)
        self.projector_p_normal = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)

        # 统一投影到 PROJ_CHANNELS 供任务解码
        PROJ_CHANNELS = 256
        self.proj_f = nn.Conv2d(combined_feature_dim, PROJ_CHANNELS, kernel_size=1)

        self.proj_z_s = nn.Conv2d(self.latent_dim_s, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_seg = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_depth = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_normal = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)

        self.zp_depth_refiner = ZpDepthRefiner(c_in=PROJ_CHANNELS, alpha=0.1)
        self.zp_normal_refiner = nn.Sequential(
            nn.Conv2d(PROJ_CHANNELS, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

        # 任务分支
        decoder_main_dim = PROJ_CHANNELS * 2
        num_seg_classes = model_config.get('num_seg_classes', 40)
        self.zp_seg_refiner = ZpSegRefiner(c_in=PROJ_CHANNELS, out_classes=num_seg_classes, alpha=0.2)

        self.predictor_seg = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim, z_p_channels=PROJ_CHANNELS,
            out_channels=num_seg_classes, scale_cap=1.0
        )
        self.predictor_depth = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim, z_p_channels=PROJ_CHANNELS,
            out_channels=1, scale_cap=1.0, use_sigmoid=True
        )
        self.predictor_normal = GatedSegDepthDecoder(decoder_main_dim, PROJ_CHANNELS, 3, scale_cap=1.0)

        self.decoder_zp_depth = SegDepthDecoder(self.latent_dim_p, 1)
        self.decoder_zp_normal = SegDepthDecoder(self.latent_dim_p, 3)

        # 原有的可视化重构（保留为 AUX）
        from .building_blocks import ConvDecoder as VisualizationDecoder
        self.decoder_geom = ResNetDecoderWithDeepSupervision(self.latent_dim_s, 1, tuple(data_config['img_size']))
        self.decoder_app = ResNetDecoderWithDeepSupervision(self.latent_dim_p, 3, tuple(data_config['img_size']))
        self.final_app_activation = nn.Sigmoid()

        # ======== 新增：分解式重构三件套 ========
        self.albedo_head = AlbedoHead(self.latent_dim_p, hidden=128)
        self.normal_head = NormalHead(self.latent_dim_s, hidden=128)
        self.light_head = LightHead(in_ch=encoder_feature_dim)  # 使用 h

        # 上采样到输入大小所需的目标尺寸
        self._target_size = tuple(data_config['img_size'])  # (H,W)

    def extract_features(self, x):
        """
        统一的特征提取接口：
        1. 运行 Backbone
        2. (如果是 ResNet) 运行 Adapter 进行对齐
        3. 拼接特征
        返回: (combined_feat, h, features_list)
        """
        raw_features = self.encoder(x)

        if self.resnet_adapters is not None:
            # ResNet Adapter Logic
            features = []
            target_h, target_w = raw_features[2].shape[-2:]  # 以 Layer3 (1/16) 为基准

            for i, feat in enumerate(raw_features):
                feat = self.resnet_adapters[i](feat)
                if feat.shape[-2:] != (target_h, target_w):
                    feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                features.append(feat)
        else:
            # ViT Logic
            features = raw_features

        # 拼接
        combined_feat = torch.cat(features, dim=1)

        # 提取全局特征 h
        last_feat = features[-1]
        h = last_feat.mean(dim=[2, 3])

        return combined_feat, h, features

    def forward(
            self,
            x,
            stage: int = 2,
            override_zs_map: torch.Tensor | None = None,
            override_zp_seg_map: torch.Tensor | None = None,
            override_zp_depth_map: torch.Tensor | None = None,
    ):
        # [MODIFIED] 使用统一接口提取特征 (兼容 ViT 和 ResNet)
        combined_feat, h, features = self.extract_features(x)

        # 1. 潜变量投影 (Latent Projections)
        z_s_map = self.projector_s(combined_feat) if override_zs_map is None else override_zs_map
        z_p_seg_map = self.projector_p_seg(combined_feat) if override_zp_seg_map is None else override_zp_seg_map
        z_p_depth_map = self.projector_p_depth(
            combined_feat) if override_zp_depth_map is None else override_zp_depth_map

        # [NEW] Normal Latent
        z_p_normal_map = self.projector_p_normal(combined_feat)

        # Global Pooling
        z_s = z_s_map.mean(dim=[2, 3])
        z_p_seg = z_p_seg_map.mean(dim=[2, 3])
        z_p_depth = z_p_depth_map.mean(dim=[2, 3])
        z_p_normal = z_p_normal_map.mean(dim=[2, 3])


        # 2. 统一投影给任务解码器 (Unified Decoder Projections)
        f_proj = self.proj_f(combined_feat)
        zs_proj = self.proj_z_s(z_s_map)

        zp_seg_proj = self.proj_z_p_seg(z_p_seg_map)
        zp_depth_proj = self.proj_z_p_depth(z_p_depth_map)
        # [NEW] Normal Projection
        zp_normal_proj = self.proj_z_p_normal(z_p_normal_map)

        # 阶段控制开关
        use_film = (stage >= 2)
        use_zp_residual = (stage >= 2)

        # ---------- 3. 下游任务解码 (Tasks) ----------

        # === Task 1: Segmentation ===
        seg_main = torch.cat([f_proj, zs_proj], dim=1)
        pred_seg_main = self.predictor_seg(seg_main, zp_seg_proj, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            seg_res = self.zp_seg_refiner(zp_seg_proj, out_size=pred_seg_main.shape[-2:])
            pred_seg = pred_seg_main + seg_res
        else:
            pred_seg = pred_seg_main

        # === Task 2: Depth ===
        depth_main = torch.cat([f_proj, zs_proj], dim=1)
        pred_depth_main = self.predictor_depth(depth_main, zp_depth_proj, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            zp_residual = self.zp_depth_refiner(zp_depth_proj)
            zp_residual = F.interpolate(zp_residual, size=pred_depth_main.shape[-2:], mode='bilinear',
                                        align_corners=False)
            pred_depth = pred_depth_main + zp_residual
        else:
            pred_depth = pred_depth_main

        # === Task 3: Normal [NEW] ===
        normal_main = torch.cat([f_proj, zs_proj], dim=1)
        pred_normal_main = self.predictor_normal(normal_main, zp_normal_proj, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            # Normal 的 Refiner 需要输出 3 通道，且通常权重(alpha)较小
            # 我们在 __init__ 里定义的 zp_normal_refiner 已经包含了这些逻辑
            normal_res = self.zp_normal_refiner(zp_normal_proj) * 0.1
            normal_res = F.interpolate(normal_res, size=pred_normal_main.shape[-2:], mode='bilinear',
                                       align_corners=False)
            pred_normal = pred_normal_main + normal_res
        else:
            pred_normal = pred_normal_main

        # 法线归一化 (关键步骤)
        pred_normal = F.normalize(pred_normal, p=2, dim=1)


        # ---------- 4. 辅助与重构 (Aux / Recon) ----------

        # 辅助深度解码 (仅 z_p)
        pred_depth_from_zp = self.decoder_zp_depth(z_p_depth_map) if use_zp_residual else torch.zeros_like(pred_depth)

        # 原有重构 (像素级)
        recon_geom_final, recon_geom_aux = self.decoder_geom(z_s_map)
        recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_p_seg_map)
        recon_app_final = self.final_app_activation(recon_app_final_logits)
        recon_app_aux = self.final_app_activation(recon_app_aux_logits)

        # ---------- 5. 分解式重构 (Decomposition) ----------
        A = self.albedo_head(z_p_seg_map)  # 由外观路 z_p
        N = self.normal_head(z_s_map)  # 由结构路 z_s (这是结构法线，用于重构)
        L = self.light_head(h)  # 全局光照
        S = shading_from_normals(N, L)  # 明暗

        # 上采样到输入大小 (如果 Feature Map 小于 Input)
        target_size = self._target_size
        if A.shape[-2:] != target_size:
            A = F.interpolate(A, size=target_size, mode='bilinear', align_corners=False)
            N = F.interpolate(N, size=target_size, mode='bilinear', align_corners=False)
            S = F.interpolate(S, size=target_size, mode='bilinear', align_corners=False)

        I_hat = torch.clamp(A * S, 0.0, 1.0)  # 重构图像

        outputs = {
            # 潜变量
            'z_s': z_s, 'z_p_seg': z_p_seg, 'z_p_depth': z_p_depth,
            'z_p_normal': z_p_normal,  # [NEW]
            'z_s_map': z_s_map, 'z_p_seg_map': z_p_seg_map, 'z_p_depth_map': z_p_depth_map,

            # 任务输出
            'pred_seg': pred_seg,
            'pred_depth': pred_depth,
            'pred_normal': pred_normal,  # [NEW] 显式命名

            # [CRITICAL] 评估器 Evaluator 会寻找 'normals' 键
            # 这里我们把‘预测法线’(pred_normal) 赋给它，用于计算 Task Metric
            'normals': pred_normal,

            # 分解中间量
            'decomposition_normal': N,  # z_s 产生的结构法线
            'albedo': A, 'shading': S, 'sh_coeff': L, 'recon_decomp': I_hat,

            # 辅助输出
            'pred_depth_from_zp': pred_depth_from_zp,
            'recon_geom': recon_geom_final, 'recon_app': recon_app_final,
            'recon_geom_aux': recon_geom_aux, 'recon_app_aux': recon_app_aux,

            # 杂项
            'stage': stage,
        }
        return outputs