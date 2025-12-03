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

        # 定义一个简单的上采样块: Upsample -> Conv -> BN -> ReLU
        def _up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.net = nn.Sequential(
            # 输入: [B, in_ch, 24, 24] (如果是384输入)
            _up_block(in_ch, hidden),  # -> 48x48
            _up_block(hidden, hidden),  # -> 96x96
            _up_block(hidden, 64),  # -> 192x192
            _up_block(64, 32),  # -> 384x384

            # 输出层
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # [0,1]
        )

    def forward(self, z_p_map):
        return self.net(z_p_map)
