# models/heads/normal_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class NormalHead(nn.Module):
    def __init__(self, in_ch, hidden=128, output="normal"):
        super().__init__()
        self.output = output  # "normal" 或 "depth"
        self.body = nn.Sequential(
            ConvBNReLU(in_ch, hidden),
            ConvBNReLU(hidden, hidden),
        )
        if output == "normal":
            self.head = nn.Conv2d(hidden, 3, kernel_size=1)
        else:
            self.head = nn.Conv2d(hidden, 1, kernel_size=1)  # depth
    def forward(self, z_s_map):
        h = self.body(z_s_map)
        y = self.head(h)
        if self.output == "normal":
            n = F.normalize(y, dim=1, eps=1e-6)
            return n  # Bx3xHxW, 单位法线
        else:
            # 简单由深度近似法线（Sobel等），你也可换成熟悉的几何算子
            d = y
            n = depth_to_normal(d)  # 你已有的话直接用；没有我后面给实现
            return n
