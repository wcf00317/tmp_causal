# ====================================================================================
# models/building_blocks.py -- FINAL REVISED VERSION
# ====================================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer


class ViTEncoder(nn.Module):
    """
    A wrapper for the Vision Transformer to extract multi-scale patch token features.

    功能升级：
    1. [工程] 严格根据 img_size (224/384) 自动匹配最佳预训练权重 (DEFAULT/SWAG)。
    2. [架构] 返回多层特征列表 (indices: 2, 5, 8, 11) 以支持高分辨率解码。
    """

    def __init__(self, name="vit_b_16", pretrained=True, img_size=224, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        if name == "vit_b_16":
            weights = None
            if pretrained:
                if img_size == 384:
                    # 384x384 输入，使用 SWAG 高分权重 (最佳推荐)
                    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                    print(f"✅ [ViTEncoder] Using 384x384 pretrained weights (SWAG_E2E_V1).")
                elif img_size == 224:
                    # 兼容默认情况
                    weights = models.ViT_B_16_Weights.DEFAULT
                    print(f"✅ [ViTEncoder] Using 224x224 pretrained weights (DEFAULT).")
                else:
                    raise ValueError(f"For pretrained ViT, img_size must be 224 or 384. Got {img_size}.")

            # 初始化模型，严格绑定 image_size
            self.vit: VisionTransformer = models.vit_b_16(weights=weights, image_size=img_size)
            self.feature_dim = 768
        else:
            raise ValueError(f"Encoder '{name}' not supported.")

        # Remove the final classification head
        self.vit.heads = nn.Identity()

    def forward(self, x):
        # 1. 预处理 (torchvision 内部会检查输入尺寸是否匹配 img_size)
        x = self.vit._process_input(x)

        # 2. 拼接 Class Token
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # 3. 手动执行 Encoder Layers 以提取中间层特征
        # 注意：torchvision 的 encoder 包含 pos_embedding + dropout + layers + ln

        # 3.1 加位置编码 + Dropout
        x = x + self.vit.encoder.pos_embedding
        if hasattr(self.vit.encoder, 'dropout'):
            x = self.vit.encoder.dropout(x)

        features = []
        # 定义要提取的层索引 (0-11)，这里取 [2, 5, 8, 11] 即第 3, 6, 9, 12 层
        # 这种取法覆盖了低、中、高层语义
        out_indices = [2, 5, 8, 11]

        # 3.2 逐层前向传播
        for i, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)

            if i in out_indices:
                # 提取 Patch Tokens (去掉 Class Token)
                patch_tokens = x[:, 1:, :]
                b, _, c = patch_tokens.shape

                # 整理为 2D 特征图: [B, 768, H/16, W/16]
                feat = patch_tokens.permute(0, 2, 1).view(b, c, self.grid_size, self.grid_size)
                features.append(feat)

        # 返回特征列表，最后一层 features[-1] 即为最高层特征
        return features
class MLP(nn.Module):
    # ... (Code from above, no changes needed here)
    """
    A simple Multi-Layer Perceptron for projection heads.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.out_features = output_dim
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    """
    一个标准的卷积解码器，加入了BatchNorm2d来稳定梯度流。
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.output_channels = output_channels

        self.decoder_net = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False),  # 使用BN时，卷积层可以不用偏置
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Block 2: 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Block 3: 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Block 4: 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Output Layer: 112x112 -> 224x224
            # 最后一层通常不加BN和ReLU，直接输出logits
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, *feature_maps):
        combined_features = torch.cat(feature_maps, dim=1)
        return self.decoder_net(combined_features)


class ConvDecoder(nn.Module):
    """
    一个增强版的卷积解码器，使用InstanceNorm2d来保证在小批次下的训练稳定性。
    """
    def __init__(self, latent_dim, output_channels, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        start_size = target_size[0] // 16

        self.upsample_in = nn.Sequential(
            nn.Linear(latent_dim, 512 * start_size * start_size),
            nn.ReLU(True)
        )
        self.start_size = start_size

        self.net = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256), # <-- 使用 InstanceNorm2d 替换 BatchNorm2d
            nn.ReLU(True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128), # <-- 使用 InstanceNorm2d 替换 BatchNorm2d
            nn.ReLU(True),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),  # <-- 使用 InstanceNorm2d 替换 BatchNorm2d
            nn.ReLU(True),
            # 112x112 -> 224x224 (最后一层不加归一化和ReLU)
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.upsample_in(x)
        x = x.view(-1, 512, self.start_size, self.start_size)
        x = self.net(x)
        return x


class SelfAttention(nn.Module):
    """
    一个高效的、用于卷积特征图的自注意力模块。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 定义Q, K, V的卷积投影层
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 伽马参数，用于残差连接的加权，初始化为0
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # 1. 计算 Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C'
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C' x (W*H)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x (W*H)

        # 2. 计算注意力图 (Attention Map)
        energy = torch.bmm(proj_query, proj_key)  # B x (W*H) x (W*H)
        attention = self.softmax(energy)  # B x (W*H) x (W*H)

        # 3. 将注意力应用于 Value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # 4. 残差连接
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    """
    一个标准的残差上采样块，包含快捷连接。
    它首先上采样，然后通过两个卷积层，最后将输入添加到输出上。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 快捷连接路径：需要上采样并调整通道数以匹配主路径输出
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        # 主路径
        self.main_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        main = self.main_path(x)
        return self.relu(shortcut + main)


class ResNetDecoderWithDeepSupervision(nn.Module):
    """
    使用残差块构建的解码器，并集成了深度监督功能。
    """
    def __init__(self, input_channels, output_channels, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        start_size = target_size[0] // 16  # 224 / 16 = 14

        # 初始层：将扁平的latent vector转换为14x14的特征图
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )
        self.start_size = start_size

        # 上采样模块 (ResNet blocks)
        self.res_block1 = ResidualBlock(512, 256)  # 14x14 -> 28x28
        self.res_block2 = ResidualBlock(256, 128)  # 28x28 -> 56x56

        self.attention = SelfAttention(128)

        self.res_block3 = ResidualBlock(128, 64)   # 56x56 -> 112x112

        # --- 深度监督分支 ---
        # 从 56x56 的特征图 (self.res_block2的输出) 创建一个辅助预测
        self.aux_head = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)

        # 主输出路径
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 112x112 -> 224x224
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 1. 初始上采样
        # x = self.upsample_in(x)
        # x = x.view(-1, 512, self.start_size, self.start_size)
        x = self.initial_conv(x)

        # 2. 通过残差块
        x = self.res_block1(x)
        x_56 = self.res_block2(x) # <-- 在这里截取中间特征 (56x56)
        x_56_attn = self.attention(x_56)
        # 3. 计算辅助输出
        out_aux = self.aux_head(x_56_attn)

        # 继续主路径
        x_final = self.res_block3(x_56_attn)
        out_final = self.final_upsample(x_final)

        # 5. 返回主输出和辅助输出
        return out_final, out_aux