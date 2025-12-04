import torch
import torch.nn as nn
import torch.nn.functional as F


class MTLLoss(nn.Module):
    """
    General Multi-Task Loss with support for:
    1. Fixed Weighting (Raw MTL)
    2. Uncertainty Weighting (Kendall et al.) - Enabled via config
    """

    def __init__(self, loss_weights, use_uncertainty=False):
        super().__init__()
        self.weights = loss_weights
        self.use_uncertainty = use_uncertainty

        # Standard Loss Functions
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss_fn = nn.L1Loss()
        self.scene_loss_fn = nn.CrossEntropyLoss()

        # Uncertainty Parameters (σ)
        # 初始化为 log(σ^2) = 0
        if self.use_uncertainty:
            self.log_vars = nn.ParameterDict({
                'seg': nn.Parameter(torch.zeros(1)),
                'depth': nn.Parameter(torch.zeros(1)),
                'scene': nn.Parameter(torch.zeros(1))
            })

    def forward(self, outputs, targets):
        loss_dict = {}

        # 1. Compute Raw Losses
        l_seg = self.seg_loss_fn(outputs['pred_seg'], targets['segmentation'])
        l_depth = self.depth_loss_fn(outputs['pred_depth'], targets['depth'])
        l_scene = self.scene_loss_fn(outputs['pred_scene'], targets['scene_type'])

        loss_dict['seg_loss_raw'] = l_seg.item()
        loss_dict['depth_loss_raw'] = l_depth.item()
        loss_dict['scene_loss_raw'] = l_scene.item()

        # 2. Weighting Strategy
        if self.use_uncertainty:
            # Loss = 1/(2σ^2) * L + log(σ)
            # 这里实现 Kendall 的公式
            precision_seg = 0.5 * torch.exp(-self.log_vars['seg'])
            loss_seg = precision_seg * l_seg + 0.5 * self.log_vars['seg']

            precision_depth = 0.5 * torch.exp(-self.log_vars['depth'])
            loss_depth = precision_depth * l_depth + 0.5 * self.log_vars['depth']

            precision_scene = 0.5 * torch.exp(-self.log_vars['scene'])
            loss_scene = precision_scene * l_scene + 0.5 * self.log_vars['scene']

            # 记录方差以便观察
            loss_dict['sigma_seg'] = self.log_vars['seg'].exp().item()
            loss_dict['sigma_depth'] = self.log_vars['depth'].exp().item()

        else:
            # Fixed Weighting (Raw MTL)
            w_seg = self.weights.get('lambda_seg', 1.0)
            w_depth = self.weights.get('lambda_depth', 1.0)
            w_scene = self.weights.get('lambda_scene', 1.0)

            loss_seg = w_seg * l_seg
            loss_depth = w_depth * l_depth
            loss_scene = w_scene * l_scene

        total_loss = loss_seg + loss_depth + loss_scene
        loss_dict['total_loss'] = total_loss
        loss_dict['seg_loss'] = loss_seg
        loss_dict['depth_loss'] = loss_depth
        loss_dict['scene_loss'] = loss_scene

        # 返回 total_loss 用于 backward，loss_dict 用于日志
        return total_loss, loss_dict