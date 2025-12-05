import torch
import torch.nn as nn
from .building_blocks import ViTEncoder, MLP, ResNetDecoderWithDeepSupervision
import torch.nn.functional as F
from .heads.albedo_head import AlbedoHead
from .heads.normal_head import NormalHead
from .heads.light_head  import LightHead
from .layers.shading    import shading_from_normals

class GatedFiLM(nn.Module):
    """用 z_p 对主特征做通道级缩放/平移；scale_cap 限制 z_p 的影响力。"""
    def __init__(self, c_main: int, c_zp: int, scale_cap: float = 1.0):
        super().__init__()
        self.scale_cap = scale_cap
        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # [B,c_zp,H,W] -> [B,c_zp,1,1]
            nn.Conv2d(c_zp, c_main*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_main*2, c_main*2, 1),
        )

    def forward(self, h_main, z_p):
        gamma_beta = self.film(z_p)               # [B,2*c_main,1,1]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # 各 [B,c_main,1,1]
        gamma = torch.tanh(gamma) * self.scale_cap
        beta  = torch.tanh(beta)  * self.scale_cap
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
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(True),
            nn.Conv2d(64,  out_classes, 3, padding=1)
        )

    def forward(self, zp_proj, out_size=None):
        x = self.net(zp_proj)                 # [B,out_classes,14,14]
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
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(True),
            nn.Conv2d(64,  1,   3, padding=1)
        )
    def forward(self, zp_proj):
        return self.alpha * self.net(zp_proj)

class GatedSegDepthDecoder(nn.Module):
    def __init__(self, main_in_channels: int, z_p_channels: int, out_channels: int, scale_cap: float = 1.0):
        super().__init__()

        self.output_channels = out_channels
        self.pre_smooth = nn.Sequential(
            nn.Conv2d(main_in_channels, main_in_channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(main_in_channels),
            nn.ReLU(inplace=True)
        )
        def _up(in_c, out_c, do_resize=True):
            layers = []
            if do_resize:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                #layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            return nn.Sequential(*layers)

        # 14→28→56→112（前三个块放大），第4个块不放大，只做卷积
        self.up1 = _up(main_in_channels, 256, do_resize=True)
        self.up2 = _up(256, 128, do_resize=True)
        self.up3 = _up(128, 64,  do_resize=True)
        self.up4 = _up(64,  64,  do_resize=False)   # 保持 112

        # FiLM 的通道要与 up4 输出一致（64）
        self.g1 = GatedFiLM(256, z_p_channels, scale_cap)
        self.g2 = GatedFiLM(128, z_p_channels, scale_cap)
        self.g3 = GatedFiLM(64,  z_p_channels, scale_cap)
        self.g4 = GatedFiLM(64,  z_p_channels, scale_cap)

        # PixelShuffle: 112→224，通道 64→16
        self.ps = nn.PixelShuffle(2)

        # PS 后通道是 64/4=16
        self.final_conv = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, main_feat, z_p_feat, use_film: bool = True, detach_zp: bool = False):
        # 绝不对 z_p_feat 做 detach，确保有梯度回传
        zpf = z_p_feat
        x = self.pre_smooth(main_feat)

        x = self.up1(x); x = self.g1(x, zpf) if use_film else x
        x = self.up2(x);         x = self.g2(x, zpf) if use_film else x
        x = self.up3(x);         x = self.g3(x, zpf) if use_film else x
        x = self.up4(x);         x = self.g4(x, zpf) if use_film else x
        x = self.ps(x)
        return self.final_conv(x)



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

        # Encoder (更新后返回多层特征列表)
        self.encoder = ViTEncoder(
            name=model_config['encoder_name'],
            pretrained=model_config['pretrained'],
            img_size=data_config['img_size'][0]
        )
        encoder_feature_dim = self.encoder.feature_dim

        # Latent dims
        self.latent_dim_s = model_config['latent_dim_s']
        self.latent_dim_p = model_config['latent_dim_p']

        # --- 修改：投影层输入维度变大 (Concatenation of 4 layers) ---
        combined_feature_dim = encoder_feature_dim * 4

        # Projectors (输入是 combined_feature_dim)
        self.projector_s = nn.Conv2d(combined_feature_dim, self.latent_dim_s, kernel_size=1)
        self.projector_p_seg = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)
        self.projector_p_depth = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)

        # Scene-level projector (依然使用 Global Avg Pooling 后的特征，通常取最后一层)
        # 这里为了简单，我们让 Scene Projector 依然只看最后一层的高层语义
        self.projector_p_scene = MLP(encoder_feature_dim, self.latent_dim_p)

        # 统一投影到 PROJ_CHANNELS 供任务解码
        PROJ_CHANNELS = 256
        self.proj_f = nn.Conv2d(combined_feature_dim, PROJ_CHANNELS, kernel_size=1)

        self.proj_z_s = nn.Conv2d(self.latent_dim_s, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_seg = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_depth = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.zp_depth_refiner = ZpDepthRefiner(c_in=PROJ_CHANNELS, alpha=0.1)

        # 任务分支
        decoder_main_dim = PROJ_CHANNELS * 2
        num_seg_classes = 40
        num_scene_classes = model_config['num_scene_classes']
        self.zp_seg_refiner = ZpSegRefiner(c_in=PROJ_CHANNELS, out_classes=num_seg_classes, alpha=0.2)

        # 使用更新了 GroupNorm 和 Bilinear Upsample 的门控解码器
        self.predictor_seg = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim, z_p_channels=PROJ_CHANNELS,
            out_channels=num_seg_classes, scale_cap=1.0
        )
        self.predictor_depth = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim, z_p_channels=PROJ_CHANNELS,
            out_channels=1, scale_cap=1.0
        )

        predictor_scene_input_dim = encoder_feature_dim + self.latent_dim_s + self.latent_dim_p
        self.predictor_scene = MLP(predictor_scene_input_dim, num_scene_classes)
        self.decoder_zp_depth = SegDepthDecoder(self.latent_dim_p, 1)

        # 原有的可视化重构（保留为 AUX）
        # 这里为了简单，重构解码器依然只接收 z_s (128维) 或 z_p (256维)
        # 所以不需要修改它们的输入维度
        from .building_blocks import ConvDecoder as VisualizationDecoder
        self.decoder_geom = ResNetDecoderWithDeepSupervision(self.latent_dim_s, 1, tuple(data_config['img_size']))
        self.decoder_app = ResNetDecoderWithDeepSupervision(self.latent_dim_p, 3, tuple(data_config['img_size']))
        self.final_app_activation = nn.Sigmoid()

        # ======== 新增：分解式重构三件套 ========
        self.albedo_head = AlbedoHead(self.latent_dim_p, hidden=128)
        self.normal_head = NormalHead(self.latent_dim_s, hidden=128)
        self.light_head = LightHead(in_ch=encoder_feature_dim)  # 使用 h (768)

        # 上采样到输入大小所需的目标尺寸
        self._target_size = tuple(data_config['img_size'])  # (H,W)

    def forward(
            self,
            x,
            stage: int = 2,
            override_zs_map: torch.Tensor | None = None,
            override_zp_seg_map: torch.Tensor | None = None,
            override_zp_depth_map: torch.Tensor | None = None,
    ):
        # 编码: 返回特征列表 [feat3, feat6, feat9, feat12]
        # features 里的每个 tensor 都是 [B, 768, 14, 14]
        features = self.encoder(x)

        # 1. 拼接多层特征用于密集预测任务
        combined_feat = torch.cat(features, dim=1)  # [B, 768*4, 14, 14]

        # 2. 提取最后一层特征用于全局任务 (Scene Classification, Lighting)
        last_feat = features[-1]
        h = last_feat.mean(dim=[2, 3])  # [B, 768]

        # 潜变量投影 (输入变为 combined_feat)
        z_s_map = self.projector_s(combined_feat) if override_zs_map is None else override_zs_map
        z_p_seg_map = self.projector_p_seg(combined_feat) if override_zp_seg_map is None else override_zp_seg_map
        z_p_depth_map = self.projector_p_depth(
            combined_feat) if override_zp_depth_map is None else override_zp_depth_map

        z_s = z_s_map.mean(dim=[2, 3])
        z_p_seg = z_p_seg_map.mean(dim=[2, 3])
        z_p_depth = z_p_depth_map.mean(dim=[2, 3])

        # Scene Projector 依然使用 h
        z_p_scene = self.projector_p_scene(h)

        # 统一投影给任务解码器
        f_proj = self.proj_f(combined_feat)
        zs_proj = self.proj_z_s(z_s_map)
        zp_seg_proj = self.proj_z_p_seg(z_p_seg_map)
        zp_depth_proj = self.proj_z_p_depth(z_p_depth_map)

        use_film = (stage >= 2)
        use_zp_residual = (stage >= 2)

        # ---------- 分割 ----------
        seg_main = torch.cat([f_proj, zs_proj], dim=1)
        pred_seg_main = self.predictor_seg(seg_main, zp_seg_proj, use_film=use_film, detach_zp=False)
        seg_residual = self.zp_seg_refiner(zp_seg_proj, out_size=pred_seg_main.shape[-2:]) \
            if use_zp_residual else torch.zeros_like(pred_seg_main)
        pred_seg = pred_seg_main + seg_residual

        # ---------- 深度 ----------
        depth_main = torch.cat([f_proj, zs_proj], dim=1)
        pred_depth_main = self.predictor_depth(depth_main, zp_depth_proj, use_film=use_film, detach_zp=False)
        if use_zp_residual:
            zp_residual = self.zp_depth_refiner(zp_depth_proj)
            zp_residual = F.interpolate(zp_residual, size=pred_depth_main.shape[-2:], mode='bilinear',
                                        align_corners=False)
        else:
            zp_residual = torch.zeros_like(pred_depth_main)
        pred_depth = pred_depth_main + zp_residual

        # ---------- 场景 ----------
        scene_predictor_input = torch.cat([h, z_s, z_p_scene], dim=1)
        pred_scene = self.predictor_scene(scene_predictor_input)
        pred_depth_from_zp = self.decoder_zp_depth(z_p_depth_map) if use_zp_residual else torch.zeros_like(pred_depth)

        # ---------- 原有重构（AUX 可视化） ----------
        recon_geom_final, recon_geom_aux = self.decoder_geom(z_s_map)
        recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_p_seg_map)
        recon_app_final = self.final_app_activation(recon_app_final_logits)
        recon_app_aux = self.final_app_activation(recon_app_aux_logits)

        # ---------- 新增：分解式重构 ----------
        A = self.albedo_head(z_p_seg_map)  # 由外观路
        N = self.normal_head(z_s_map)  # 由结构路
        L = self.light_head(h)  # 全局光照（低频 SH）
        S = shading_from_normals(N, L)  # 明暗（近灰）

        # 上采样到输入大小（如果当前特征分辨率 < 输入分辨率）
        target_size = self._target_size
        if A.shape[-2:] != target_size:
            A = F.interpolate(A, size=target_size, mode='bilinear', align_corners=False)
            N = F.interpolate(N, size=target_size, mode='bilinear', align_corners=False)
            S = F.interpolate(S, size=target_size, mode='bilinear', align_corners=False)

        I_hat = torch.clamp(A * S, 0.0, 1.0)  # 分解式重构

        outputs = {
            # 潜变量（原样保留）
            'z_s': z_s, 'z_p_seg': z_p_seg, 'z_p_depth': z_p_depth, 'z_p_scene': z_p_scene,
            'z_s_map': z_s_map, 'z_p_seg_map': z_p_seg_map, 'z_p_depth_map': z_p_depth_map,
            # 任务输出
            'pred_seg': pred_seg, 'pred_depth': pred_depth, 'pred_scene': pred_scene,
            'pred_depth_from_zp': pred_depth_from_zp,
            # 原有像素重构（AUX）
            'recon_geom': recon_geom_final, 'recon_app': recon_app_final,
            'recon_geom_aux': recon_geom_aux, 'recon_app_aux': recon_app_aux,
            # 新增：分解式重构三件套
            'albedo': A, 'normals': N, 'shading': S, 'sh_coeff': L, 'recon_decomp': I_hat,
            # 杂项
            'stage': stage,
        }
        return outputs