import torch
import torch.nn as nn
from .building_blocks import ViTEncoder, MLP, FuserAndDecoder


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

        # Decoupler: 将 normal 替换为 scene
        self.projector_s = MLP(encoder_feature_dim, self.latent_dim_s)
        self.projector_p_seg = MLP(encoder_feature_dim, self.latent_dim_p)
        self.projector_p_depth = MLP(encoder_feature_dim, self.latent_dim_p)
        self.projector_p_scene = MLP(encoder_feature_dim, self.latent_dim_p)  # 新增

        # Predictors: 将 normal 替换为 scene
        num_seg_classes = 40
        num_scene_classes = model_config['num_scene_classes']
        self.predictor_seg = FuserAndDecoder(encoder_feature_dim, self.latent_dim_s, self.latent_dim_p, num_seg_classes)
        self.predictor_depth = FuserAndDecoder(encoder_feature_dim, self.latent_dim_s, self.latent_dim_p, 1)
        # 场景分类预测器是一个MLP，而不是卷积解码器
        self.predictor_scene = MLP(self.latent_dim_s + self.latent_dim_p, num_scene_classes)  # 新增

        from .building_blocks import ConvDecoder as VisualizationDecoder
        self.decoder_geom = VisualizationDecoder(self.latent_dim_s, 1, data_config['img_size'])
        self.decoder_app = VisualizationDecoder(self.latent_dim_p, 2, data_config['img_size'])

    def forward(self, x):
        feature_map = self.encoder(x)
        h = feature_map.mean(dim=[2, 3])

        z_s = self.projector_s(h)
        z_p_seg = self.projector_p_seg(h)
        z_p_depth = self.projector_p_depth(h)
        z_p_scene = self.projector_p_scene(h)  # 新增

        if self.training and self.config['z_s_bottleneck_noise'] > 0:
            noise = torch.randn_like(z_s) * self.config['z_s_bottleneck_noise']
            z_s = z_s + noise

        pred_seg = self.predictor_seg(feature_map, z_s, z_p_seg)
        pred_depth = self.predictor_depth(feature_map, z_s, z_p_depth)
        pred_scene = self.predictor_scene(torch.cat([z_s, z_p_scene], dim=1))  # 新增

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