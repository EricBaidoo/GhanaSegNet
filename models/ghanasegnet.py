import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=256):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class GhanaSegNet(nn.Module):
    def __init__(self, num_classes=6):
        super(GhanaSegNet, self).__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-lite0')
        
        self.enc0 = nn.Sequential(self.encoder._conv_stem, self.encoder._bn0, self.encoder._swish)
        self.enc1 = self.encoder._blocks[0:2]
        self.enc2 = self.encoder._blocks[2:4]
        self.enc3 = self.encoder._blocks[4:10]
        self.enc4 = self.encoder._blocks[10:]

        self.conv1 = nn.Conv2d(320, 256, kernel_size=1)
        self.transformer = TransformerBlock(dim=256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(128 + 112, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(64 + 40, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(32 + 24, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(16 + 16, 16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)                          # stem
        x1 = self.enc1(x0)                         # low
        x2 = self.enc2(x1)                         # mid
        x3 = self.enc3(x2)                         # high
        x4 = self.enc4(x3)                         # deeper

        x4 = self.conv1(x4)
        x4 = self.transformer(x4)

        d4 = self.up4(x4)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
