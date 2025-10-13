import torch
import torch.nn as nn
from .hsic import HSIC
import torch.nn.functional as F
from metrics.lpips import LPIPSMetric

class CompositeLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.weights = loss_weights

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss = nn.L1Loss()
        self.scene_loss = nn.CrossEntropyLoss()

        self.independence_loss = HSIC(normalize=True)
        self.recon_geom_loss = nn.L1Loss()
        #self.recon_app_loss = nn.MSELoss()
        self.recon_app_loss_lpips = LPIPSMetric(net='vgg')  # LPIPS部分
        self.recon_app_loss_l1 = nn.L1Loss()  # 新增L1部分

    def forward(self, outputs, targets):
        loss_dict = {}

        # --- 1. Task Losses ---
        l_seg = self.seg_loss(outputs['pred_seg'], targets['segmentation'])
        l_depth = self.depth_loss(outputs['pred_depth'], targets['depth'])
        l_scene = self.scene_loss(outputs['pred_scene'], targets['scene_type'])

        l_task = (self.weights.get('lambda_seg', 1.0) * l_seg +
                  self.weights.get('lambda_depth', 1.0) * l_depth +
                  self.weights.get('lambda_scene', 1.0) * l_scene)

        loss_dict.update({'task_loss': l_task, 'seg_loss': l_seg, 'depth_loss': l_depth, 'scene_loss': l_scene})

        # --- 2. Causal Independence Loss ---
        z_s = outputs['z_s']
        z_p_seg = outputs['z_p_seg']
        z_p_depth = outputs['z_p_depth']
        z_p_scene = outputs['z_p_scene']

        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)

        l_ind = (self.independence_loss(z_s_centered, z_p_seg - z_p_seg.mean(0, keepdim=True)) +
                 self.independence_loss(z_s_centered, z_p_depth - z_p_depth.mean(0, keepdim=True)) +
                 self.independence_loss(z_s_centered, z_p_scene - z_p_scene.mean(0, keepdim=True)))
        loss_dict['independence_loss'] = l_ind

        # --- 3. Reconstruction Loss (Main and Auxiliary) ---
        # Main reconstruction loss
        l_recon_g = self.recon_geom_loss(outputs['recon_geom'], targets['depth'])
        #l_recon_a = self.recon_app_loss(outputs['recon_app'], targets['appearance_target'])
        l_recon_a_lpips = self.recon_app_loss_lpips(outputs['recon_app'], targets['appearance_target'])
        l_recon_a_l1 = self.recon_app_loss_l1(outputs['recon_app'], targets['appearance_target'])

        # 混合损失
        l_recon_a = l_recon_a_lpips + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1
        loss_dict.update({'recon_geom_loss': l_recon_g, 'recon_app_loss': l_recon_a})

        # --- START: CORRECTED AUXILIARY LOSS CALCULATION ---
        recon_geom_aux = outputs['recon_geom_aux']
        recon_app_aux = outputs['recon_app_aux']

        aux_size_g = recon_geom_aux.shape[2:]
        aux_size_a = recon_app_aux.shape[2:]

        target_depth_aux = F.interpolate(targets['depth'], size=aux_size_g, mode='bilinear', align_corners=False)
        target_app_aux = F.interpolate(targets['appearance_target'], size=aux_size_a, mode='bilinear',
                                       align_corners=False)

        l_recon_g_aux = self.recon_geom_loss(recon_geom_aux, target_depth_aux)

        # 让辅助外观损失也使用混合模式
        l_recon_a_lpips_aux = self.recon_app_loss_lpips(recon_app_aux, target_app_aux)
        l_recon_a_l1_aux = self.recon_app_loss_l1(recon_app_aux, target_app_aux)
        l_recon_a_aux = l_recon_a_lpips_aux + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1_aux

        loss_dict.update({'recon_geom_aux_loss': l_recon_g_aux, 'recon_app_aux_loss': l_recon_a_aux})

        # --- 4. Total Loss ---
        total_loss = (l_task +
                      self.weights.get('lambda_independence', 0) * l_ind +
                      self.weights.get('alpha_recon_geom', 0) * l_recon_g +
                      self.weights.get('beta_recon_app', 0) * l_recon_a +
                      self.weights.get('alpha_recon_geom_aux', 0) * l_recon_g_aux +
                      self.weights.get('beta_recon_app_aux', 0) * l_recon_a_aux)
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict