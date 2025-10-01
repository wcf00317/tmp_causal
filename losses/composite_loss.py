import torch
import torch.nn as nn
from .hsic import HSIC
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.weights = loss_weights

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss = nn.L1Loss()
        self.scene_loss = nn.CrossEntropyLoss()  # 新增

        self.independence_loss = HSIC(normalize=True)
        self.recon_geom_loss = nn.L1Loss()
        self.recon_app_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        loss_dict = {}

        # --- 1. 任务损失 (替换 normal 为 scene) ---
        l_seg = self.seg_loss(outputs['pred_seg'], targets['segmentation'])
        l_depth = self.depth_loss(outputs['pred_depth'], targets['depth'])
        l_scene = self.scene_loss(outputs['pred_scene'], targets['scene_type'])  # 新增

        l_task = (self.weights.get('lambda_seg', 1.0) * l_seg +
                  self.weights.get('lambda_depth', 1.0) * l_depth +
                  self.weights.get('lambda_scene', 1.0) * l_scene)  # 新增

        loss_dict.update({'task_loss': l_task, 'seg_loss': l_seg, 'depth_loss': l_depth, 'scene_loss': l_scene})

        # --- 2. 因果独立性损失 (替换 z_p_normal 为 z_p_scene) ---
        z_s = outputs['z_s']
        z_p_seg = outputs['z_p_seg']
        z_p_depth = outputs['z_p_depth']
        z_p_scene = outputs['z_p_scene']  # 新增

        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)

        l_ind = (self.independence_loss(z_s_centered, z_p_seg - z_p_seg.mean(0, keepdim=True)) +
                 self.independence_loss(z_s_centered, z_p_depth - z_p_depth.mean(0, keepdim=True)) +
                 self.independence_loss(z_s_centered, z_p_scene - z_p_scene.mean(0, keepdim=True)))  # 新增
        loss_dict['independence_loss'] = l_ind

        # --- 3. 重构损失 (保持不变) ---
        l_recon_g = self.recon_geom_loss(outputs['recon_geom'], targets['depth'])
        l_recon_a = self.recon_app_loss(outputs['recon_app'], targets['appearance_target'])
        loss_dict.update({'recon_geom_loss': l_recon_g, 'recon_app_loss': l_recon_a})

        recon_geom_aux_list = outputs['recon_geom_aux']
        recon_app_aux_list = outputs['recon_app_aux']

        l_recon_g_aux_total, l_recon_a_aux_total = 0.0, 0.0
        num_aux = len(recon_geom_aux_list)

        for recon_geom_aux, recon_app_aux in zip(recon_geom_aux_list, recon_app_aux_list):
            aux_size = recon_geom_aux.shape[2:]
            target_depth_aux = F.interpolate(targets['depth'], size=aux_size, mode='bilinear', align_corners=False)
            target_app_aux = F.interpolate(targets['appearance_target'], size=aux_size, mode='bilinear',
                                           align_corners=False)

            l_recon_g_aux_total += self.recon_geom_loss(recon_geom_aux, target_depth_aux)
            l_recon_a_aux_total += self.recon_app_loss(recon_app_aux, target_app_aux)

        # 取平均 (也可以直接相加，看你实验需求)
        l_recon_g_aux = l_recon_g_aux_total / num_aux
        l_recon_a_aux = l_recon_a_aux_total / num_aux

        loss_dict.update({'recon_geom_aux_loss': l_recon_g_aux, 'recon_app_aux_loss': l_recon_a_aux})

        # --- 4. 总损失 ---
        total_loss = (l_task +
                      self.weights.get('lambda_independence', 0) * l_ind +
                      self.weights.get('alpha_recon_geom', 0) * l_recon_g +
                      self.weights.get('beta_recon_app', 0) * l_recon_a +
                      self.weights.get('alpha_recon_geom_aux', 0) * l_recon_g_aux +
                      self.weights.get('beta_recon_app_aux', 0) * l_recon_a_aux)
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict