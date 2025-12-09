import math
import torch
import torch.nn as nn
#from .hsic import HSIC
from .linear_cka import LinearCKA
import torch.nn.functional as F
from metrics.lpips import LPIPSMetric

def edge_weight_from_gt(gt, alpha=2.0, q=0.7):
    """计算边界权重 w in [1, 1+alpha]"""
    if gt.dim()==3: gt=gt.unsqueeze(1)
    # 使用 .to(gt.device, dtype=gt.dtype) 确保张量在同一设备和类型
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=gt.device, dtype=gt.dtype).view(1,1,3,3)
    ky = kx.transpose(2,3)
    gx = F.conv2d(gt, kx, padding=1); gy = F.conv2d(gt, ky, padding=1)
    mag = (gx.abs()+gy.abs())
    thr = torch.quantile(mag.view(mag.size(0), -1), q, dim=1, keepdim=True).view(-1,1,1,1)
    mask = (mag >= thr).float()
    return 1.0 + alpha*mask

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4).clamp(0,1)

def total_variation_l1(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean()
    dw = (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()
    return dh + dw

def cross_covariance_abs(A: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """A: Bx3xHxW, N: Bx3xHxW"""
    B = A.size(0)
    a = A.view(B, 3, -1); a = a - a.mean(dim=2, keepdim=True)
    n = N.view(B, 3, -1); n = n - n.mean(dim=2, keepdim=True)
    cov = torch.bmm(a, n.transpose(1,2)) / (a.size(-1) - 1)  # Bx3x3
    return cov.abs().mean()

def rgb_to_lab_safe(x: torch.Tensor) -> torch.Tensor:
    """如果没装 kornia，就回退成直接返回 x（不会报错，只是少了真正的 Lab 约束）"""
    try:
        import kornia as K
        return K.color.rgb_to_lab(x)
    except Exception:
        return x


def _sobel(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(2,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy

def seg_edge_consistency_loss(seg_logits, geom_from_zs, weight=0.1, tau=0.1):
    """
    让分割的边缘（p_max 的梯度）和几何的边缘对齐。
    seg_logits: [B,C,H,W]
    geom_from_zs: [B,1,H,W]
    """
    # 对齐分辨率
    if seg_logits.shape[-2:] != geom_from_zs.shape[-2:]:
        geom_from_zs = F.interpolate(geom_from_zs, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)

    # 分割边缘：对 softmax 后的最大类概率求梯度
    p = torch.softmax(seg_logits, dim=1)
    p_max = p.max(dim=1, keepdim=True).values
    gx_p, gy_p = _sobel(p_max)

    # 几何边缘（不反传给几何，避免干扰 z_s 重构）
    g = (geom_from_zs - geom_from_zs.mean()) / (geom_from_zs.std() + 1e-6)
    gx_g, gy_g = _sobel(g)

    # 归一化 & 只在强几何边缘处计算（掩码）
    def _norm(a):
        return a / (a.abs().mean() + 1e-6)
    gx_p, gy_p = _norm(gx_p), _norm(gy_p)
    gx_g, gy_g = _norm(gx_g), _norm(gy_g)

    edge_mag = (gx_g.abs() + gy_g.abs())
    # 每张图动态取高梯度分位（比如 top 30%）
    q = torch.quantile(edge_mag.view(edge_mag.size(0), -1), 0.70, dim=1, keepdim=True)  # 0.6~0.8 可调
    q = q.view(-1, 1, 1, 1)
    mask = (edge_mag > q).float()

    l = ( (gx_p - gx_g).abs() + (gy_p - gy_g).abs() ) * mask
    return weight * l.mean()

def edge_consistency_loss(geom_from_zs, depth_gt, weight=0.1):
    """
    让 z_s 的几何重构在边缘上和 GT 深度一致。输入都是 [B,1,H,W]。
    """
    # 尺寸对齐
    if geom_from_zs.shape[-2:] != depth_gt.shape[-2:]:
        depth_gt = F.interpolate(depth_gt, size=geom_from_zs.shape[-2:], mode="bilinear", align_corners=False)

    # 归一化，避免尺度影响梯度
    g = (geom_from_zs - geom_from_zs.mean()) / (geom_from_zs.std() + 1e-6)
    d = (depth_gt     - depth_gt.mean())     / (depth_gt.std()     + 1e-6)

    gx_g, gy_g = _sobel(g)
    gx_d, gy_d = _sobel(d)
    l = (gx_g - gx_d).abs().mean() + (gy_g - gy_d).abs().mean()
    return weight * l


class EdgeConsistencyLoss(nn.Module):
    def __init__(self, levels: int = 3, eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.eps = eps
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3).contiguous()
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _grad(self, x: torch.Tensor):
        gx = F.conv2d(x, self.sobel_x.to(x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.dtype), padding=1)
        return gx, gy

    def _charbonnier(self, x: torch.Tensor):
        return torch.sqrt(x * x + self.eps * self.eps)

    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        if pred_depth.dim() == 3: pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.dim()  == 3:  gt_depth  = gt_depth.unsqueeze(1)
        if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
            gt_depth = F.interpolate(gt_depth, size=pred_depth.shape[-2:], mode="nearest")  # ★ 保边

        # log-depth 更关注相对变化
        pd = torch.log1p(torch.clamp(pred_depth, min=0))
        gd = torch.log1p(torch.clamp(gt_depth , min=0))

        loss = 0.0
        pd_ms, gd_ms = pd, gd
        for _ in range(self.levels):
            gx_p, gy_p = self._grad(pd_ms)
            gx_g, gy_g = self._grad(gd_ms)
            # Charbonnier 代替 L1，边缘梯度更稳定
            loss = loss + self._charbonnier(gx_p - gx_g).mean() + self._charbonnier(gy_p - gy_g).mean()
            # 下一尺度（平均池化）
            if _ < self.levels - 1:
                pd_ms = F.avg_pool2d(pd_ms, 2, 2)
                gd_ms = F.avg_pool2d(gd_ms, 2, 2)
        return loss


class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # pred: [B, 3, H, W] (Output from NormalHead, usually normalized)
        # gt:   [B, 3, H, W] (From Dataset)

        # 1. 确保预测值归一化
        pred = F.normalize(pred, p=2, dim=1)

        # 2. 生成掩码 (GT非零区域)
        # LibMTL 逻辑: binary_mask = (torch.sum(gt, dim=1) != 0)
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1)

        # 3. 计算 Cosine Loss: 1 - cos(theta)
        # element-wise dot product
        dot_prod = (pred * gt).sum(dim=1, keepdim=True)

        # 仅在有效区域计算
        num_valid = torch.sum(binary_mask)
        if num_valid > 0:
            loss = 1 - torch.sum(dot_prod * binary_mask) / num_valid
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss

# =========================
#   自适应版本（当前生效）
# =========================
class AdaptiveCompositeLoss(nn.Module):
    """
    在 Kendall 不确定性加权中，将 regularizer 项从
        0.5 * log_var
    改为
        0.5 * (log_var + C)
    其中 C = -log_var_min（这里为 4.0），保证 regularizer 非负。
    该改动不改变 d/d(log_var) 的梯度（仍为 0.5），
    因而不影响优化动态，但能避免“总损为负”的日志现象。
    """
    def __init__(self, loss_weights):
        super().__init__()
        self.weights = loss_weights

        # ==== 基础项 ====
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.depth_loss = nn.L1Loss()
        self.scene_loss = nn.CrossEntropyLoss()
        self.normal_loss = NormalLoss()  # [NEW] 新增法线损失函数

        self.independence_loss = LinearCKA(eps=1e-6)

        self.recon_geom_loss     = nn.L1Loss()
        self.recon_app_loss_lpips= LPIPSMetric(net='vgg')
        self.recon_app_loss_l1   = nn.L1Loss()

        self.edge_consistency_loss = EdgeConsistencyLoss()

        # ==== 用 YAML 权重做 log_var 的先验初始化（log_var=log(1/λ)） ====
        def _to_logvar(inv_weight_default_one):
            lam = float(inv_weight_default_one)
            lam = max(lam, 1e-6)
            return math.log(1.0 / lam)

        init = init = {
            'seg'        : _to_logvar(self.weights.get('lambda_seg',       1.0)),
            'depth'      : _to_logvar(self.weights.get('lambda_depth',     1.0)),
            'scene'      : _to_logvar(self.weights.get('lambda_scene',     1.0)),
            'normal'     : _to_logvar(self.weights.get('lambda_normal',    1.0)), # [NEW] 新增法线权重参数
            'ind'        : _to_logvar(self.weights.get('lambda_independence', 1.0)),
            'recon_geom' : _to_logvar(self.weights.get('alpha_recon_geom', 1.0)),
            'recon_app'  : _to_logvar(self.weights.get('beta_recon_app',   1.0)),
        }
        self.log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.tensor([v], dtype=torch.float32))
            for k, v in init.items()
        })

        # 轻度范围，避免 early collapse / explosion
        self._logvar_min = -4.0
        self._logvar_max =  4.0

        # ★ 新增：正则项常数偏移 C = -log_var_min (=4.0)，确保 0.5*(log_var + C) ≥ 0
        self._regularizer_shift = -self._logvar_min

    def _uw(self, name: str, loss_scalar: torch.Tensor) -> torch.Tensor:
        """
        Kendall 不确定性加权（修正版）：
            0.5 * exp(-log_var) * L + 0.5 * (log_var + C)
        其中 C = self._regularizer_shift。
        - 保证 regularizer 非负（总损不至于出现负数）
        - 不改变关于 log_var 的梯度（仍为 0.5）
        """
        lv = self.log_vars[name]
        lv_clamped = torch.clamp(lv, self._logvar_min, self._logvar_max)

        weighted_loss = 0.5 * torch.exp(-lv_clamped) * loss_scalar
        regularizer   = 0.5 * (lv_clamped + self._regularizer_shift)
        return weighted_loss + regularizer

    def forward(self, outputs, targets):
        loss_dict = {}

        # ===== 1) 主任务 =====
        l_seg = self.seg_loss(outputs['pred_seg'], targets['segmentation'])
        loss_seg = self._uw('seg', l_seg)

        # Depth
        l_depth = self.depth_loss(outputs['pred_depth'], targets['depth'])
        loss_depth = self._uw('depth', l_depth)

        # Scene (Optional: 只有当权重 > 0 时才计算，兼容 NYUv2 无场景标签)
        if self.weights.get('lambda_scene', 0.0) > 0:
            l_scene = self.scene_loss(outputs['pred_scene'], targets['scene_type'])
            loss_scene = self._uw('scene', l_scene)
        else:
            l_scene = torch.tensor(0.0, device=l_seg.device)
            loss_scene = torch.tensor(0.0, device=l_seg.device)

        # Normal [NEW] (只有当输出和目标都存在时计算)
        if 'normals' in outputs and 'normal' in targets and self.weights.get('lambda_normal', 0.0) > 0:
            l_normal = self.normal_loss(outputs['normals'], targets['normal'])
            loss_normal = self._uw('normal', l_normal)
        else:
            l_normal = torch.tensor(0.0, device=l_seg.device)
            loss_normal = torch.tensor(0.0, device=l_seg.device)

        # 汇总任务 Loss
        l_task = loss_seg + loss_depth + loss_scene + loss_normal

        loss_dict.update({
            'seg_loss': l_seg,
            'depth_loss': l_depth,
            'scene_loss': l_scene,
            'normal_loss': l_normal,  # [NEW]
            'task_loss': l_task,
        })

        # ===== 2) 解耦独立性 =====
        z_s = outputs['z_s']
        z_p_seg = outputs['z_p_seg']
        z_p_depth = outputs['z_p_depth']
        #z_p_scene = outputs['z_p_scene']

        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)

        cka_seg = self.independence_loss(z_s_centered, z_p_seg - z_p_seg.mean(0, keepdim=True))
        cka_depth = self.independence_loss(z_s_centered, z_p_depth - z_p_depth.mean(0, keepdim=True))
        if 'z_p_normal' in outputs:
            z_p_normal = outputs['z_p_normal']
            cka_normal = self.independence_loss(z_s_centered, z_p_normal - z_p_normal.mean(0, keepdim=True))
        else:
            cka_normal = torch.tensor(0.0, device=z_s.device)

        l_ind = cka_seg + cka_depth + cka_normal
        loss_dict['cka_seg'] = cka_seg
        loss_dict['cka_depth'] = cka_depth
        loss_dict['cka_normal'] = cka_normal
        loss_dict['independence_loss'] = l_ind

        stage = int(outputs.get('stage', 2))
        if stage == 1:
            loss_ind = torch.zeros_like(l_ind)
        else:
            loss_ind = self.weights.get('lambda_independence', 0.05) * l_ind



        # ===== 3) 重建（主） =====
        l_recon_g = self.recon_geom_loss(outputs['recon_geom'], targets['depth'])
        l_recon_a_lpips = self.recon_app_loss_lpips(outputs['recon_app'], targets['appearance_target'])
        l_recon_a_l1 = self.recon_app_loss_l1(outputs['recon_app'], targets['appearance_target'])
        l_recon_a = l_recon_a_lpips + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1

        loss_recon_g = self._uw('recon_geom', l_recon_g)
        loss_recon_a = self._uw('recon_app', l_recon_a)

        loss_dict.update({
            'recon_geom_loss': l_recon_g,
            'recon_app_loss': l_recon_a,
        })

        _dev = outputs['pred_depth'].device if 'pred_depth' in outputs else next(
            v.device for v in outputs.values() if torch.is_tensor(v))

        # ===== 3.1) 分解式重构（可选） =====
        A = outputs.get('albedo', None)
        S = outputs.get('shading', None)
        Nn = outputs.get('normals', None)
        I_hat = outputs.get('recon_decomp', None)

        l_img = torch.tensor(0.0, device=_dev)
        l_alb_tv = torch.tensor(0.0, device=_dev)
        l_alb_chroma = torch.tensor(0.0, device=_dev)
        l_sh_gray = torch.tensor(0.0, device=_dev)
        l_norm = torch.tensor(0.0, device=_dev)
        l_xcov = torch.tensor(0.0, device=_dev)

        if (A is not None) and (S is not None) and (I_hat is not None):
            target_img = targets.get('appearance_target', targets.get('image', None))
            if target_img is not None:
                l_img = F.l1_loss(srgb_to_linear(I_hat), srgb_to_linear(target_img))
            l_alb_tv = total_variation_l1(A)
            Lab = rgb_to_lab_safe(A)
            l_alb_chroma = total_variation_l1(Lab[:, 1:]) if Lab.shape[1] >= 3 else total_variation_l1(A)
            l_sh_gray = ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2).mean()

        if Nn is not None:
            l_norm = (Nn.norm(dim=1) - 1).abs().mean() + total_variation_l1(Nn)

        if (A is not None) and (Nn is not None):
            l_xcov = cross_covariance_abs(A, Nn)

        loss_dict.update({
            'decomp_img': l_img,
            'decomp_alb_tv': l_alb_tv,
            'decomp_alb_chroma': l_alb_chroma,
            'decomp_sh_gray': l_sh_gray,
            'decomp_norm': l_norm,
            'decomp_xcov': l_xcov,
        })

        # ===== 4) 重建（辅助） =====
        recon_geom_aux = outputs['recon_geom_aux']
        recon_app_aux = outputs['recon_app_aux']
        aux_size_g = recon_geom_aux.shape[2:]
        aux_size_a = recon_app_aux.shape[2:]

        # ★ 关键修复：depth GT 用 nearest，避免软化边界
        target_depth_aux = F.interpolate(targets['depth'], size=aux_size_g, mode='nearest')
        target_app_aux = F.interpolate(targets['appearance_target'], size=aux_size_a, mode='bilinear',
                                       align_corners=False)

        l_recon_g_aux = self.recon_geom_loss(recon_geom_aux, target_depth_aux)
        l_recon_a_lpips_aux = self.recon_app_loss_lpips(recon_app_aux, target_app_aux)
        l_recon_a_l1_aux = self.recon_app_loss_l1(recon_app_aux, target_app_aux)
        l_recon_a_aux = l_recon_a_lpips_aux + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1_aux

        loss_dict.update({
            'recon_geom_aux_loss': l_recon_g_aux,
            'recon_app_aux_loss': l_recon_a_aux,
        })

        # ===== 5) 一致性项 =====
        # 5.1 原有：几何辅助头 vs GT
        l_depth_from_zp = self.depth_loss(outputs['pred_depth_from_zp'], targets['depth'])
        l_edge_geom = self.edge_consistency_loss(outputs['recon_geom'], targets['depth'])
        edge_w = self.weights.get('alpha_recon_geom_edges', 0.1)
        seg_edge_w = self.weights.get('beta_seg_edge_from_geom', 0.1)

        # 5.2 新增：主头 pred_depth vs GT 的边缘一致性（关键改动）
        l_edge_pred = self.edge_consistency_loss(outputs['pred_depth'], targets['depth'])
        edge_pred_w = self.weights.get('lambda_edge_consistency_pred',
                                       self.weights.get('lambda_edge_consistency', 0.0))

        loss_dict.update({
            'depth_from_zp_loss': l_depth_from_zp,
            'edge_consistency_loss_geom': l_edge_geom,
            'edge_consistency_loss_pred': l_edge_pred,
        })

        # ===== 6) 汇总 =====
        edge_scale = 0.0 if stage == 1 else 1.0  # 与原逻辑保持一致
        total_loss = (
                l_task + loss_ind + loss_recon_g + loss_recon_a +
                self.weights.get('alpha_recon_geom_aux', 0.0) * l_recon_g_aux +
                self.weights.get('beta_recon_app_aux', 0.0) * l_recon_a_aux +
                self.weights.get('lambda_depth_zp', 0.0) * l_depth_from_zp +
                # 几何辅助头的边缘一致性（原有逻辑，保留）
                self.weights.get('lambda_edge_consistency', 0.0) * edge_scale * l_edge_geom +
                # 你原来那条“函数版 edge_consistency_loss（带 weight=edge_w）”仍保留
                edge_consistency_loss(outputs['recon_geom'], targets['depth'], weight=edge_w) +
                # ★ 新增：主头 pred_depth 的边缘一致性（关键）
                edge_scale * edge_pred_w * l_edge_pred +
                # seg 与几何的边缘一致性（只影响分割头）
                edge_scale * seg_edge_consistency_loss(outputs['pred_seg'], outputs['recon_geom'], weight=seg_edge_w) +
                # 分解式项（可选）
                self.weights.get('lambda_img', 0.0) * l_img +
                self.weights.get('lambda_alb_tv', 0.0) * l_alb_tv +
                self.weights.get('lambda_alb_chroma', 0.0) * l_alb_chroma +
                self.weights.get('lambda_sh_gray', 0.0) * l_sh_gray +
                self.weights.get('lambda_norm', 0.0) * l_norm +
                self.weights.get('lambda_xcov', 0.0) * l_xcov
        )

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict