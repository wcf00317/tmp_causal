import torch
import torch.nn as nn


class SingleTaskLoss(nn.Module):
    def __init__(self, active_task, loss_weights):
        super().__init__()
        self.active_task = active_task
        self.weights = loss_weights

        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss_fn = nn.L1Loss()
        self.scene_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss_dict = {}
        total_loss = 0.0

        if self.active_task == 'seg':
            l_seg = self.seg_loss_fn(outputs['pred_seg'], targets['segmentation'])
            total_loss += self.weights.get('lambda_seg', 1.0) * l_seg
            loss_dict['seg_loss'] = l_seg

        elif self.active_task == 'depth':
            l_depth = self.depth_loss_fn(outputs['pred_depth'], targets['depth'])
            total_loss += self.weights.get('lambda_depth', 1.0) * l_depth
            loss_dict['depth_loss'] = l_depth

        elif self.active_task == 'scene':
            l_scene = self.scene_loss_fn(outputs['pred_scene'], targets['scene_type'])
            total_loss += self.weights.get('lambda_scene', 1.0) * l_scene
            loss_dict['scene_loss'] = l_scene

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict