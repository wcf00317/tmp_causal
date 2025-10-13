import torch
import torch.nn as nn
from .building_blocks import ViTEncoder, MLP, ResNetDecoderWithDeepSupervision


class SegDepthDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.output_channels = output_channels

        # 内部上采样块，使用老师推荐的 interpolate + conv 结构
        def _make_upsample_layer(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
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

        self.encoder = ViTEncoder(
            name=model_config['encoder_name'],
            pretrained=model_config['pretrained'],
            img_size=data_config['img_size'][0]
        )
        encoder_feature_dim = self.encoder.feature_dim

        self.latent_dim_s = model_config['latent_dim_s']
        self.latent_dim_p = model_config['latent_dim_p']

        self.projector_s = nn.Conv2d(encoder_feature_dim, self.latent_dim_s, kernel_size=1)
        self.projector_p_seg = nn.Conv2d(encoder_feature_dim, self.latent_dim_p, kernel_size=1)
        self.projector_p_depth = nn.Conv2d(encoder_feature_dim, self.latent_dim_p, kernel_size=1)

        # 用于场景分类的全局特征的投影器仍然是MLP
        self.projector_p_scene = MLP(encoder_feature_dim, self.latent_dim_p)

        PROJ_CHANNELS = 256  # 定义一个统一的投影维度
        self.proj_f = nn.Conv2d(encoder_feature_dim, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_s = nn.Conv2d(self.latent_dim_s, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_seg = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_p_depth = nn.Conv2d(self.latent_dim_p, PROJ_CHANNELS, kernel_size=1)

        # 解码器的输入通道数现在是投影后拼接的维度
        decoder_input_dim = PROJ_CHANNELS * 3  # feature_map + z_s + z_p

        num_seg_classes = 40
        num_scene_classes = model_config['num_scene_classes']

        self.predictor_seg = SegDepthDecoder(decoder_input_dim, num_seg_classes)
        self.predictor_depth = SegDepthDecoder(decoder_input_dim, 1)

        # 场景分类预测器保持不变
        predictor_scene_input_dim = encoder_feature_dim + self.latent_dim_s + self.latent_dim_p
        self.predictor_scene = MLP(predictor_scene_input_dim, num_scene_classes)
        self.decoder_zp_depth = SegDepthDecoder(self.latent_dim_p, 1)

        from .building_blocks import ConvDecoder as VisualizationDecoder
        # self.decoder_geom = VisualizationDecoder(self.latent_dim_s, 1, data_config['img_size'])
        # self.decoder_app = VisualizationDecoder(self.latent_dim_p, 2, data_config['img_size'])
        self.decoder_geom = ResNetDecoderWithDeepSupervision(self.latent_dim_s, 1, tuple(data_config['img_size']))
        #self.decoder_app = ResNetDecoderWithDeepSupervision(self.latent_dim_p, 2, tuple(data_config['img_size']))
        self.decoder_app = ResNetDecoderWithDeepSupervision(self.latent_dim_p, 3, tuple(data_config['img_size']))
        self.final_app_activation = nn.Sigmoid()

    def forward(self, x):
        feature_map = self.encoder(x)

        z_s_map = self.projector_s(feature_map)  # Shape: [B, 64, 14, 14]
        z_p_seg_map = self.projector_p_seg(feature_map)  # Shape: [B, 128, 14, 14]
        z_p_depth_map = self.projector_p_depth(feature_map)  # Shape: [B, 128, 14, 14]

        # 3. 只有在需要全局信息时，才进行全局平均池化
        h = feature_map.mean(dim=[2, 3])  # Shape: [B, 768]
        z_s = z_s_map.mean(dim=[2, 3])  # Shape: [B, 64]
        z_p_seg = z_p_seg_map.mean(dim=[2, 3])  # Shape: [B, 128]
        z_p_depth = z_p_depth_map.mean(dim=[2, 3])  # Shape: [B, 128]
        z_p_scene = self.projector_p_scene(h)  # Shape: [B, 128]

        # 1. 将不同的特征图投影到统一维度
        f_proj = self.proj_f(feature_map)
        zs_proj = self.proj_z_s(z_s_map)
        zp_seg_proj = self.proj_z_p_seg(z_p_seg_map)
        zp_depth_proj = self.proj_z_p_depth(z_p_depth_map)

        # 2. 拼接投影后的特征图，送入解码器
        pred_seg = self.predictor_seg(torch.cat([f_proj, zs_proj, zp_seg_proj], dim=1))
        pred_depth = self.predictor_depth(torch.cat([f_proj, zs_proj, zp_depth_proj], dim=1))

        # 5. 场景分类使用全局向量
        scene_predictor_input = torch.cat([h, z_s, z_p_scene], dim=1)
        pred_scene = self.predictor_scene(scene_predictor_input)

        pred_depth_from_zp = self.decoder_zp_depth(z_p_depth_map)

        # 6. 用于可视化的解码器使用全局向量
        # recon_geom = self.decoder_geom(z_s)
        # recon_app = self.decoder_app(z_p_seg)
        recon_geom_final, recon_geom_aux = self.decoder_geom(z_s_map)
        recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_p_seg_map)

        recon_app_final = self.final_app_activation(recon_app_final_logits)
        recon_app_aux = self.final_app_activation(recon_app_aux_logits)

        outputs = {
            'z_s': z_s,
            'z_p_seg': z_p_seg,
            'z_p_depth': z_p_depth,
            'z_p_scene': z_p_scene,  # 新增
            'pred_seg': pred_seg,
            'pred_depth': pred_depth,
            'pred_scene': pred_scene,  # 新增
            'pred_depth_from_zp': pred_depth_from_zp,
            # 'recon_geom': recon_geom,
            # 'recon_app': recon_app,
            'recon_geom': recon_geom_final,      # 主重构
            'recon_app': recon_app_final,  # 主重构
            'recon_geom_aux': recon_geom_aux,  # 辅助重构
            'recon_app_aux': recon_app_aux,  # 辅助重构

            'z_s_map': z_s_map,
            'z_p_seg_map': z_p_seg_map
        }

        return outputs