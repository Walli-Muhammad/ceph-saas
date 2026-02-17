"""
U-Net Model for Cephalometric Landmark Detection

Standard U-Net architecture for heatmap regression:
- Input: 1 channel (grayscale X-ray), 512x512
- Output: 29 channels (one heatmap per landmark), 512x512
- Final activation: Sigmoid for probability outputs [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv -> BatchNorm -> ReLU) x 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled feature map from decoder
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (if input size is not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Cephalometric Landmark Detection
    
    Architecture:
    - Encoder: 4 downsampling blocks (64, 128, 256, 512 channels)
    - Bottleneck: 1024 channels
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 29 channels with Sigmoid activation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 29,
        bilinear: bool = True
    ):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (29 for landmarks)
            bilinear: Use bilinear upsampling instead of transposed conv
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Sigmoid activation for probability outputs
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (B, 1, 512, 512)
        
        Returns:
            Output tensor of shape (B, 29, 512, 512) with values in [0, 1]
        """
        # Encoder
        x1 = self.inc(x)      # (B, 64, 512, 512)
        x2 = self.down1(x1)   # (B, 128, 256, 256)
        x3 = self.down2(x2)   # (B, 256, 128, 128)
        x4 = self.down3(x3)   # (B, 512, 64, 64)
        x5 = self.down4(x4)   # (B, 512, 32, 32) or (B, 1024, 32, 32)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # (B, 256, 64, 64)
        x = self.up2(x, x3)   # (B, 128, 128, 128)
        x = self.up3(x, x2)   # (B, 64, 256, 256)
        x = self.up4(x, x1)   # (B, 64, 512, 512)
        
        # Output layer
        logits = self.outc(x)  # (B, 29, 512, 512)
        
        # Apply sigmoid for probability outputs
        output = self.sigmoid(logits)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """
    Test the U-Net model with a dummy input.
    """
    print("=" * 80)
    print("U-NET MODEL TEST")
    print("=" * 80)
    
    # Create model
    model = UNet(in_channels=1, out_channels=29, bilinear=True)
    print(f"Model created with {model.count_parameters():,} trainable parameters")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 512, 512)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Validate output
    assert output.shape == (batch_size, 29, 512, 512), \
        f"Expected shape (2, 29, 512, 512), got {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, \
        f"Output should be in [0, 1], got [{output.min()}, {output.max()}]"
    
    print("\n✓ Model test passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_model()
