import torch
import torch.nn as nn
from .building_blocks import ViTEncoder, MLP, SimpleDecoder


class SegDepthDecoder(nn.Module):
    def __init__(self, encoder_feature_dim, latent_dim_s, latent_dim_p, output_channels):
        super().__init__()
        self.output_channels = output_channels

        # 上采样块
        self.deconv1 = self._make_deconv_layer(encoder_feature_dim + latent_dim_s + latent_dim_p, 256)
        self.deconv2 = self._make_deconv_layer(256, 128)
        self.deconv3 = self._make_deconv_layer(128, 64)
        self.deconv4 = self._make_deconv_layer(64, 32)

        # 最终的预测层
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)

    def _make_deconv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, feature_map, z_s_map, z_p_map):
        # 1. 在解码器的最开始，就将所有信息拼接起来
        # 这是关键一步：原始的、包含丰富信息的feature_map被直接送入
        x = torch.cat([feature_map, z_s_map, z_p_map], dim=1)

        # 2. 逐层上采样
        x = self.deconv1(x)  # 14x14 -> 28x28
        x = self.deconv2(x)  # 28x28 -> 56x56
        x = self.deconv3(x)  # 56x56 -> 112x112
        x = self.deconv4(x)  # 112x112 -> 224x224

        # 3. 生成最终预测
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

        num_seg_classes = 40
        num_scene_classes = model_config['num_scene_classes']

        self.predictor_seg = SegDepthDecoder(encoder_feature_dim, self.latent_dim_s, self.latent_dim_p, num_seg_classes)
        self.predictor_depth = SegDepthDecoder(encoder_feature_dim, self.latent_dim_s, self.latent_dim_p, 1)

        # 场景分类预测器保持不变
        predictor_scene_input_dim = encoder_feature_dim + self.latent_dim_s + self.latent_dim_p
        self.predictor_scene = MLP(predictor_scene_input_dim, num_scene_classes)


        from .building_blocks import ConvDecoder as VisualizationDecoder
        self.decoder_geom = VisualizationDecoder(self.latent_dim_s, 1, data_config['img_size'])
        self.decoder_app = VisualizationDecoder(self.latent_dim_p, 2, data_config['img_size'])

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

        if self.training and self.config['z_s_bottleneck_noise'] > 0:
            noise = torch.randn_like(z_s) * self.config['z_s_bottleneck_noise']
            z_s = z_s + noise

        # 4. 将保留了空间信息的特征图送入解码器
        pred_seg = self.predictor_seg(feature_map, z_s_map, z_p_seg_map)
        pred_depth = self.predictor_depth(feature_map, z_s_map, z_p_depth_map)

        # 5. 场景分类使用全局向量
        scene_predictor_input = torch.cat([h, z_s, z_p_scene], dim=1)
        pred_scene = self.predictor_scene(scene_predictor_input)

        # 6. 用于可视化的解码器使用全局向量
        recon_geom = self.decoder_geom(z_s)
        recon_app = self.decoder_app(z_p_seg)


        outputs = {
            'z_s': z_s,
            'z_p_seg': z_p_seg,
            'z_p_depth': z_p_depth,
            'z_p_scene': z_p_scene,  # 新增
            'pred_seg': pred_seg,
            'pred_depth': pred_depth,
            'pred_scene': pred_scene,  # 新增
            'recon_geom': recon_geom,
            'recon_app': recon_app,
        }

        return outputs