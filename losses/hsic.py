import torch
import torch.nn as nn

def _pairwise_sq_dists(X):
    """
    计算成对的平方欧氏距离矩阵。
    这种方法比 torch.cdist(X, X).pow(2) 在数值上更稳定。
    公式: (a-b)^2 = a^2 - 2ab + b^2
    """
    # x_norm 形状: (n, 1)
    x_norm = (X**2).sum(dim=1, keepdim=True)
    # dists 广播计算: (n, 1) + (1, n) - 2 * (n, n) -> (n, n)
    dists = x_norm + x_norm.t() - 2.0 * (X @ X.t())
    # 保证距离值为非负
    return torch.clamp(dists, min=0.0)

class HSIC(nn.Module):
    """
    HSIC的鲁棒实现版本。
    - 可选的、基于中位数启发法的自适应sigma。
    - 可选的归一化。
    - 确保返回的张量设备和类型与输入一致。
    """

    def __init__(self, normalize=True, eps=1e-8):
        """
        Args:
            normalize (bool): 是否返回归一化后的HSIC值 (建议为True)。
            eps (float): 用于避免除零错误的小常数。
        """
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def _rbf_kernel(self, X, sigma):
        dist_sq = _pairwise_sq_dists(X)
        # 在sigma平方上加一个epsilon防止其为0
        return torch.exp(-dist_sq / (2.0 * (sigma ** 2) + self.eps))

    def _median_heuristic(self, X):
        """
        使用中位数启发法自适应地选择RBF核的sigma参数。
        """
        with torch.no_grad():
            dists_sq = _pairwise_sq_dists(X)
            # 只选择矩阵上三角的非零元素来计算中位数，更精确
            vals = dists_sq[torch.triu(torch.ones_like(dists_sq), diagonal=1) == 1]
            median = torch.median(vals)
            # sigma = sqrt(median / 2)
            sigma = torch.sqrt(median / 2.0 + self.eps)
        # 如果所有点都重合导致中位数为0，则回退到1.0
        return sigma if sigma > 0 else torch.tensor(1.0, device=X.device)

    def _center_kernel_matrix(self, K):
        n = K.shape[0]
        # H = I - (1/n) * 1_n * 1_n^T
        ones = torch.ones((n, n), device=K.device, dtype=K.dtype)
        H = torch.eye(n, device=K.device, dtype=K.dtype) - ones / n
        return H @ K @ H

    def forward(self, X, Y):
        n = X.shape[0]
        if n < 4: # 在过小的batch上计算HSIC没有意义且不稳定
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        # 1. 使用中位数启发法确定sigma
        sigma_x = self._median_heuristic(X)
        sigma_y = self._median_heuristic(Y)

        # 2. 计算核矩阵并中心化
        Kx = self._rbf_kernel(X, sigma_x)
        Ky = self._rbf_kernel(Y, sigma_y)
        Kxc = self._center_kernel_matrix(Kx)
        Kyc = self._center_kernel_matrix(Ky)

        # 3. 计算HSIC biased estimator
        hsic_val = torch.trace(Kxc @ Kyc) / ((n - 1) ** 2 + self.eps)

        if self.normalize:
            # 4. 归一化HSIC值
            # norm_x is sqrt(HSIC(X, X))
            norm_x = torch.sqrt(torch.trace(Kxc @ Kxc) / ((n - 1) ** 2) + self.eps)
            norm_y = torch.sqrt(torch.trace(Kyc @ Kyc) / ((n - 1) ** 2) + self.eps)
            hsic_val = hsic_val / (norm_x * norm_y + self.eps)

        return hsic_val