# ====================================================================================
# models/building_blocks.py -- FINAL REVISED VERSION
# ====================================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer


# ViTEncoder remains the same as above, correctly extracting the 14x14 grid
class ViTEncoder(nn.Module):
    # ... (Code from above, no changes needed here)
    """
    A wrapper for the Vision Transformer to extract patch token features.
    This is the proper way to use ViT for dense prediction tasks.
    """

    def __init__(self, name="vit_b_16", pretrained=True, img_size=224, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # Load the pretrained ViT model
        if name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.vit: VisionTransformer = models.vit_b_16(weights=weights, image_size=img_size)
            self.feature_dim = 768  # Output feature dimension for ViT-B
        else:
            raise ValueError(f"Encoder '{name}' not supported.")

        # Remove the final classification head
        self.vit.heads = nn.Identity()

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder(x)
        patch_tokens = x[:, 1:, :]
        b, _, c = patch_tokens.shape
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.view(b, c, self.grid_size, self.grid_size)
        return patch_tokens


# MLP remains the same
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
    A simple Convolutional Decoder to reconstruct maps from a non-spatial latent vector.
    This is specifically used for the visualization part of the pre-experiment.
    """

    def __init__(self, latent_dim, output_channels, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # This starting grid size should align with the ViT patch grid (e.g., 14x14)
        start_size = target_size[0] // 16
        self.upsample_in = nn.Sequential(
            nn.Linear(latent_dim, 256 * start_size * start_size),
            nn.ReLU(True)
        )
        self.start_size = start_size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # -> 224x224
        )

    def forward(self, x):
        x = self.upsample_in(x)
        x = x.view(-1, 256, self.start_size, self.start_size)  # Reshape to (B, 256, 14, 14)
        x = self.net(x)
        return x