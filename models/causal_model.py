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
    def __init__(self, main_in_channels: int, z_p_channels: int, out_channels: int, 
                 low_level_in_channels: int = 256, # ResNet Layer1 输出
                 scale_cap: float = 1.0,
                 use_sigmoid=False):
        super().__init__()

        self.output_channels = out_channels
        
        # [1] 高层特征处理 (保持不变)
        aspp_out = 256
        self.aspp = ASPP(in_channels=main_in_channels, out_channels=aspp_out, atrous_rates=[12, 24, 36])

        # ======================================================================
        # [Strategy 3 核心修改]：双路 Attention
        # ======================================================================
        
        # 第一路 Attention：处理高层语义 (1/16 尺度)
        # 作用：决定“这是什么物体” (Semantic Context)
        self.high_level_attention = CrossAttentionFusion(
            main_channels=aspp_out, 
            style_channels=z_p_channels  # 如果用了 Style Bank，这里要是 256*3=768
        )

        # Skip Connection 投影层 (保持不变)
        self.low_level_proj_channels = 48
        self.project_low_level = nn.Sequential(
            nn.Conv2d(low_level_in_channels, self.low_level_proj_channels, 1, bias=False),
            nn.GroupNorm(4, self.low_level_proj_channels),
            nn.ReLU(inplace=True)
        )

        # 第二路 Attention：处理低层细节 (1/4 尺度)
        # 作用：决定“边界在哪里”、“纹理是什么” (Texture/Edge Details)
        # 注意：这里的主特征是投影后的 low_level (48通道)
        self.low_level_attention = CrossAttentionFusion(
            main_channels=self.low_level_proj_channels, # 48
            style_channels=z_p_channels,                # 仍然查询同一个风格库
            inter_channels=32                           # 内部投影维度小一点，省显存
        )

        # ======================================================================

        # [4] 最终预测头
        head_in_channels = aspp_out + self.low_level_proj_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in_channels, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, main_feat, z_p_feat, low_level_feat=None, use_film: bool = True, detach_zp: bool = False):
        """
        z_p_feat: 可以是单个 z_p，也可以是拼接后的 z_p_bank
        """
        # 1. 提取高层特征
        x = self.aspp(main_feat)

        if use_film:
            if detach_zp: z_p_feat = z_p_feat.detach()
            
            # [Step A] 高层融合：把风格注入到语义中
            x = self.high_level_attention(x, z_p_feat)

        # 2. 处理低层特征 (Skip Connection)
        if low_level_feat is not None:
            low = self.project_low_level(low_level_feat) # [B, 48, H/4, W/4]
            
            # [Step B] 低层融合：关键点！把风格注入到边缘细节中
            # 这能让分割边界更贴合纹理变化，让法线细节更丰富
            if use_film:
                # 注意：z_p_feat 需要插值到和 low 一样的尺寸吗？
                # CrossAttentionFusion 内部会自动处理尺寸不匹配，所以直接传即可
                # 但为了效率，建议在这里不对 z_p 做操作，让 Attention 内部去 view/permute
                low = self.low_level_attention(low, z_p_feat)

            # 上采样高层特征并拼接
            x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, low], dim=1)
        
        # 3. 输出
        out = self.head(x)
        scale_factor = 4 if (low_level_feat is not None) else 8
        out = F.interpolate(out, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out
    
class CrossAttentionFusion(nn.Module):
    """
    基于 Cross-Attention 的特征融合模块。
    用于替代 GatedFiLM，实现结构 (Structure) 对 风格 (Style) 的按需查询。
    
    Args:
        main_channels (int): 主特征 (Structure/Query) 的通道数
        style_channels (int): 风格特征 (Style/Key/Value) 的通道数
        inter_channels (int, optional): 内部投影通道数。默认减半以节省显存。
    """
    def __init__(self, main_channels, style_channels, inter_channels=None):
        super().__init__()
        self.inter_channels = inter_channels if inter_channels else main_channels // 2

        # Query: 来自结构特征 (z_s / main_feat)
        # 作用：询问 "这个位置的形状需要什么纹理？"
        self.query_conv = nn.Conv2d(main_channels, self.inter_channels, kernel_size=1)

        # Key: 来自风格特征 (z_p)
        # 作用：提供 "我这里有木纹/玻璃/金属..." 的索引
        self.key_conv = nn.Conv2d(style_channels, self.inter_channels, kernel_size=1)

        # Value: 来自风格特征 (z_p)
        # 作用：提供实际的纹理特征内容
        self.value_conv = nn.Conv2d(style_channels, main_channels, kernel_size=1)

        # Learnable Scale: 初始为0，保证初始状态下只利用结构特征，避免训练震荡
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_main, z_p):
        """
        x_main: [B, C_main, H, W]  <- 结构特征 (Query)
        z_p:    [B, C_style, H, W] <- 风格特征 (Key/Value)
                (注意：z_p 即使是全图尺寸，如果是从 4x4 插值来的，Attention 也能自动处理)
        """
        batch_size, C, H, W = x_main.size()
        
        # 1. 生成 Query [B, HW, C_inter]
        # permute 调整为 [Batch, Sequence_Length, Dim]
        proj_query = self.query_conv(x_main).view(batch_size, self.inter_channels, -1)
        proj_query = proj_query.permute(0, 2, 1) 

        # 2. 生成 Key [B, C_inter, HW]
        # 注意：这里假设 z_p 的 spatial size 和 x_main 可能不同，但如果是插值后的则相同。
        # view(-1) 会拉平 H*W 维度
        proj_key = self.key_conv(z_p).view(batch_size, self.inter_channels, -1)

        # 3. 计算 Attention Map (Similarity) -> [B, HW_main, HW_style]
        # 这一步计算了每个结构像素与每个风格像素的相关性
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # 4. 生成 Value [B, C_main, HW_style]
        proj_value = self.value_conv(z_p).view(batch_size, C, -1)

        # 5. 加权融合 (Weighted Sum) -> [B, C_main, HW_main]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        # 6. 还原形状
        out = out.view(batch_size, C, H, W)

        # 7. 残差连接 (Residual Connection)
        # z_s 仍然是主导，z_p 作为补充细节加在上面
        out = self.gamma * out + x_main
        
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
        #self.upsample4 = _make_upsample_layer(64, 32)  # 112x112 -> 224x224

        # 最终的预测层
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        #x = self.upsample4(x)
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
            target_dim = 1024
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
        SHARED_DECODER_DIM = 256
        aspp_in_channels = PROJ_CHANNELS * 2
        STYLE_BANK_CHANNELS = PROJ_CHANNELS * 3
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

        # [修改] 实例化解码器时，将 SHARED_DECODER_DIM 传进去
        self.predictor_seg = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim,  # <--- 动态传递
            z_p_channels=STYLE_BANK_CHANNELS,
            out_channels=num_seg_classes,
            scale_cap=1.0
        )
        self.predictor_depth = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim,  # <--- 动态传递
            z_p_channels=STYLE_BANK_CHANNELS,
            out_channels=1,
            scale_cap=1.0,
            use_sigmoid=False
        )
        self.predictor_normal = GatedSegDepthDecoder(
            main_in_channels=decoder_main_dim,  # <--- 动态传递
            z_p_channels=STYLE_BANK_CHANNELS,
            out_channels=3,
            scale_cap=1.0
        )

        self.decoder_zp_depth = SegDepthDecoder(self.latent_dim_p, 1)
        self.decoder_zp_normal = SegDepthDecoder(self.latent_dim_p, 3)

        # 原有的可视化重构（保留为 AUX）
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

        return combined_feat, h, features, raw_features

    def forward(
            self,
            x,
            stage: int = 2,
            override_zs_map: torch.Tensor | None = None,
            override_zp_seg_map: torch.Tensor | None = None,
            override_zp_depth_map: torch.Tensor | None = None,
    ):
        # [MODIFIED] 使用统一接口提取特征
        # combined_feat, h, features = self.extract_features(x)
        #
        # # 1. 潜变量投影 (Latent Projections)
        # z_s_map = self.projector_s(combined_feat) if override_zs_map is None else override_zs_map
        # z_p_seg_map = self.projector_p_seg(combined_feat) if override_zp_seg_map is None else override_zp_seg_map
        # z_p_depth_map = self.projector_p_depth(
        #     combined_feat) if override_zp_depth_map is None else override_zp_depth_map
        #
        # # [NEW] Normal Latent
        # z_p_normal_map = self.projector_p_normal(combined_feat)
        #
        # # Global Pooling
        # z_s = z_s_map.mean(dim=[2, 3])
        # z_p_seg = z_p_seg_map.mean(dim=[2, 3])
        # z_p_depth = z_p_depth_map.mean(dim=[2, 3])
        # z_p_normal = z_p_normal_map.mean(dim=[2, 3])

        combined_feat, h, features, raw_features = self.extract_features(x)

        low_level_feat = raw_features[0]

        # 获取特征图的原始空间尺寸 [H, W] (例如 32x64)
        H_map, W_map = combined_feat.shape[2], combined_feat.shape[3]

        # 设置网格大小 (2x2 或 4x4)
        # grid_size = 2
        grid_size = 4

        # 1. 潜变量投影 (Latent Projections)

        # === A. 结构分支 z_s (保持不变，保留全分辨率空间结构) ===
        if override_zs_map is None:
            z_s_map = self.projector_s(combined_feat)
        else:
            z_s_map = override_zs_map

        # === B. 风格分支 z_p (Adaptive GAP + Interpolate) ===

        # 1. Adaptive Pooling: [B, C, H, W] -> [B, C, K, K]
        #    这将强制把空间信息压缩成 KxK 的粗糙网格
        feat_grid = F.adaptive_avg_pool2d(combined_feat, (grid_size, grid_size))

        # 2. 投影到 latent_dim_p: [B, C, K, K] -> [B, dim_p, K, K]
        #    Projector 是 1x1 卷积，它可以处理任意 spatial 尺寸
        z_p_seg_grid = self.projector_p_seg(feat_grid)
        z_p_depth_grid = self.projector_p_depth(feat_grid)

        # 兼容性检查：如果有 normal projector
        if hasattr(self, 'projector_p_normal'):
            z_p_normal_grid = self.projector_p_normal(feat_grid)
        else:
            z_p_normal_grid = None

        # 3. 上采样回 [B, dim, H, W] 以兼容解码器
        #    使用 'nearest' 插值，严格保留 KxK 的块状结构，不进行平滑过渡，
        #    防止模型通过插值梯度偷学到位置信息。
        if override_zp_seg_map is None:
            z_p_seg_map = F.interpolate(z_p_seg_grid, size=(H_map, W_map), mode='nearest')
        else:
            z_p_seg_map = override_zp_seg_map

        if override_zp_depth_map is None:
            z_p_depth_map = F.interpolate(z_p_depth_grid, size=(H_map, W_map), mode='nearest')
        else:
            z_p_depth_map = override_zp_depth_map

        if z_p_normal_grid is not None:
            z_p_normal_map = F.interpolate(z_p_normal_grid, size=(H_map, W_map), mode='nearest')
        else:
            z_p_normal_map = None

        # =======================================================
        # 2. Global Pooling Variables (用于 Loss 计算，如 CKA)
        # =======================================================

        # z_s 仍计算全局均值
        z_s = z_s_map.mean(dim=[2, 3])

        # z_p 现在是 [B, dim_p, K, K]。
        # 为了计算 CKA (独立性损失)，我们需要一个向量。
        # 这里有两个选择：
        #   Option A: 再次求均值 -> [B, dim_p]。这代表"整张图的平均风格"。
        #             (推荐，因为 CKA 通常衡量全局独立性，且 dim 较小计算快)
        #   Option B: Flatten -> [B, dim_p * K * K]。这代表"网格风格的拼接"。

        # 建议使用 Mean，保持与 z_s (也是 Mean 来的) 的维度一致性
        z_p_seg = z_p_seg_grid.mean(dim=[2, 3])
        z_p_depth = z_p_depth_grid.mean(dim=[2, 3])
        z_p_normal = z_p_normal_grid.mean(dim=[2, 3]) if z_p_normal_grid is not None else None

        # 2. 统一投影给任务解码器
        f_proj = self.proj_f(combined_feat)
        zs_proj = self.proj_z_s(z_s_map)

        zp_seg_proj = self.proj_z_p_seg(z_p_seg_map)
        zp_depth_proj = self.proj_z_p_depth(z_p_depth_map)
        zp_normal_proj = self.proj_z_p_normal(z_p_normal_map)

        # 阶段控制开关
        use_film = (stage >= 2)
        use_zp_residual = (stage >= 2)
        zp_bank = torch.cat([zp_seg_proj, zp_depth_proj, zp_normal_proj], dim=1)

        # ---------- 3. 下游任务解码 (Tasks) ----------

        # [REVERTED] 恢复原始逻辑：拼接 f(256) + zs(256) = 512
        # 这个 512 维的特征会进入每个解码器内部的 ASPP
        task_input_feat = torch.cat([f_proj, zs_proj], dim=1)

        # 辅助函数：用于 checkpoint 的包装
        def run_decoder(decoder_module, main, zp, low_level_feat, film, detach):
            return decoder_module(main, zp, low_level_feat, film, detach)

        # === Task 1: Segmentation ===
        # [OPTIMIZATION] 使用 Checkpoint 节省显存 (仅在训练时)
        if self.training and task_input_feat.requires_grad:
            # 注意：这里传入的是 512 维的 task_input_feat
            pred_seg_main = checkpoint(run_decoder, self.predictor_seg, task_input_feat, zp_bank, low_level_feat, use_film, False,
                                       use_reentrant=False)
        else:
            pred_seg_main = self.predictor_seg(task_input_feat, zp_bank, low_level_feat=low_level_feat, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            seg_res = self.zp_seg_refiner(zp_seg_proj, out_size=pred_seg_main.shape[-2:])
            pred_seg = pred_seg_main + seg_res
        else:
            pred_seg = pred_seg_main

        # === Task 2: Depth ===
        if self.training and task_input_feat.requires_grad:
            pred_depth_main = checkpoint(run_decoder, self.predictor_depth, task_input_feat, zp_bank, low_level_feat, use_film,
                                         False, use_reentrant=False)
        else:
            pred_depth_main = self.predictor_depth(task_input_feat, zp_bank, low_level_feat=low_level_feat, use_film=use_film, detach_zp=False)

        if use_zp_residual:
            zp_residual = self.zp_depth_refiner(zp_depth_proj)
            zp_residual = F.interpolate(zp_residual, size=pred_depth_main.shape[-2:], mode='bilinear',
                                        align_corners=False)
            pred_depth = pred_depth_main + zp_residual
        else:
            pred_depth = pred_depth_main

        # === Task 3: Normal [NEW] ===
        if self.training and task_input_feat.requires_grad:
            pred_normal_main = checkpoint(run_decoder, self.predictor_normal, task_input_feat, zp_bank, low_level_feat, use_film,
                                          False, use_reentrant=False)
        else:
            pred_normal_main = self.predictor_normal(task_input_feat, zp_bank, low_level_feat=low_level_feat, use_film=use_film,
                                                     detach_zp=False)

        if use_zp_residual:
            # 兼容性处理：如果您的代码里没有 zp_normal_refiner，请注释掉这几行
            if hasattr(self, 'zp_normal_refiner'):
                normal_res = self.zp_normal_refiner(zp_normal_proj) * 0.1
                normal_res = F.interpolate(normal_res, size=pred_normal_main.shape[-2:], mode='bilinear',
                                           align_corners=False)
                pred_normal = pred_normal_main + normal_res
            else:
                pred_normal = pred_normal_main
        else:
            pred_normal = pred_normal_main

        # 法线归一化
        pred_normal = F.normalize(pred_normal, p=2, dim=1)

        # [REMOVED] Scene Classification 移除，恢复为您原始代码逻辑

        # ---------- 4. 辅助与重构 (Aux / Recon) ----------

        # 辅助深度解码 (仅 z_p)
        pred_depth_from_zp = self.decoder_zp_depth(z_p_depth_map) if use_zp_residual else torch.zeros_like(pred_depth)

        # 原有重构 (像素级) - 同样应用 Checkpoint 以解决显存瓶颈
        # 注意：这里保留了重构分支，因为这是 Causal 解耦的核心
        if self.training and z_s_map.requires_grad:
            recon_geom_final, recon_geom_aux = checkpoint(self.decoder_geom, z_s_map, use_reentrant=False)
            recon_app_final_logits, recon_app_aux_logits = checkpoint(self.decoder_app, z_p_seg_map,
                                                                      use_reentrant=False)
        else:
            recon_geom_final, recon_geom_aux = self.decoder_geom(z_s_map)
            recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_p_seg_map)

        recon_app_final = self.final_app_activation(recon_app_final_logits)
        recon_app_aux = self.final_app_activation(recon_app_aux_logits)

        # ---------- 5. 分解式重构 (Decomposition) ----------
        # 如果您的代码里有这部分就保留，没有就删除
        A = self.albedo_head(z_p_seg_map)
        N = self.normal_head(z_s_map)
        L = self.light_head(h)
        S = shading_from_normals(N, L)

        # 上采样
        target_size = self._target_size
        if A.shape[-2:] != target_size:
            A = F.interpolate(A, size=target_size, mode='bilinear', align_corners=False)
            N = F.interpolate(N, size=target_size, mode='bilinear', align_corners=False)
            S = F.interpolate(S, size=target_size, mode='bilinear', align_corners=False)

        I_hat = torch.clamp(A * S, 0.0, 1.0)

        outputs = {
            # 潜变量
            'z_s': z_s, 'z_p_seg': z_p_seg, 'z_p_depth': z_p_depth,
            'z_p_normal': z_p_normal,
            'z_s_map': z_s_map, 'z_p_seg_map': z_p_seg_map, 'z_p_depth_map': z_p_depth_map,

            # 任务输出
            'pred_seg': pred_seg,
            'pred_depth': pred_depth,
            'pred_normal': pred_normal,
            'normals': pred_normal,  # 评估器需要的键名

            # 分解中间量
            'decomposition_normal': N,
            'albedo': A, 'shading': S, 'sh_coeff': L, 'recon_decomp': I_hat,

            # 辅助输出
            'pred_depth_from_zp': pred_depth_from_zp,
            'recon_geom': recon_geom_final, 'recon_app': recon_app_final,
            'recon_geom_aux': recon_geom_aux, 'recon_app_aux': recon_app_aux,

            # 杂项
            'stage': stage,
        }
        return outputs