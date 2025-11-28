# models/heads/albedo_head.py
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class AlbedoHead(nn.Module):
    def __init__(self, in_ch, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, hidden),
            ConvBNReLU(hidden, hidden),
            nn.Conv2d(hidden, 3, kernel_size=1),
            nn.Sigmoid(),  # [0,1]
        )
    def forward(self, z_p_map):
        return self.net(z_p_map)  # Bx3xHxW
