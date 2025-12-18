# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Generator Model: Mini 1D U-Net for FPGA Deployment
# =============================================================================
"""
MINI ARCHITECTURE FOR RTL
=========================

This is a compact U-Net generator designed for FPGA implementation.
The architecture matches the RTL implementation in rtl/generator_mini.v

Architecture:
    Input [2×16] → Enc1 [4×8] → Bottleneck [8×4] → Dec1 [4×8] → Output [2×16]
                   ↓                                    ↑
                   └──────── Skip Connection ───────────┘

Parameters:
    - Frame length: 16 samples
    - Channels: 2 → 4 → 8 → 4 → 2
    - Kernel size: 3
    - Stride: 2 for encoder, 1 for decoder

Fixed-Point (for FPGA):
    - Weights: Q1.7 (8-bit signed)
    - Activations: Q8.8 (16-bit signed)
    - Accumulator: Q16.16 (32-bit)

Total Parameters: ~744
Total MACs/Frame: ~5,000
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv1D → LeakyReLU
    
    Mathematical operation:
        y = LeakyReLU(Conv1D(x))
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        leaky_slope: float = 0.2
    ):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.activation = nn.LeakyReLU(negative_slope=leaky_slope)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))
    
    def get_params_count(self) -> int:
        """Calculate number of parameters: K * C_in * C_out + C_out (bias)"""
        return self.kernel_size * self.in_channels * self.out_channels + self.out_channels
    
    def get_macs(self, output_length: int) -> int:
        """Calculate MACs: K * C_in * C_out * L_out"""
        return self.kernel_size * self.in_channels * self.out_channels * output_length


class MiniGenerator(nn.Module):
    """
    Mini U-Net Generator for FPGA deployment.
    
    This matches the RTL implementation with:
    - 16-sample frame length
    - 3-level U-Net (Enc1 → Bottleneck → Dec1)
    - Channel progression: 2 → 4 → 8 → 4 → 2
    - Additive skip connections
    
    Layer Specifications:
    ---------------------
    | Layer      | In Ch | Out Ch | Stride | L_out | Params | MACs   |
    |------------|-------|--------|--------|-------|--------|--------|
    | Enc1       | 2     | 4      | 2      | 8     | 28     | 192    |
    | Bottleneck | 4     | 8      | 2      | 4     | 104    | 384    |
    | Dec1       | 8     | 4      | 1      | 8     | 100    | 768    |
    | OutConv    | 4     | 2      | 1      | 16    | 26     | 384    |
    |------------|-------|--------|--------|-------|--------|--------|
    | TOTAL      |       |        |        |       | 258    | 1,728  |
    
    Note: Skip connection adds Enc1 output to Dec1 output (after upsample)
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        frame_length: int = 16,
        leaky_slope: float = 0.2
    ):
        super(MiniGenerator, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.frame_length = frame_length
        
        # Encoder 1: 2 → 4 channels, stride=2, output length = 8
        self.enc1 = ConvBlock(
            in_channels=input_channels,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=1,
            leaky_slope=leaky_slope
        )
        
        # Bottleneck: 4 → 8 channels, stride=2, output length = 4
        self.bottleneck = ConvBlock(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            leaky_slope=leaky_slope
        )
        
        # Upsample 1: Nearest neighbor ×2 (4 → 8 samples)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Decoder 1: 8 → 4 channels, stride=1, output length = 8
        self.dec1 = ConvBlock(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            leaky_slope=leaky_slope
        )
        
        # Upsample 2: Nearest neighbor ×2 (8 → 16 samples)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Output convolution: 4 → 2 channels
        self.out_conv = nn.Conv1d(
            in_channels=4,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        # Final activation (tanh for normalized output)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mini U-Net.
        
        Args:
            x: Input tensor of shape [B, 2, 16]
            
        Returns:
            Output tensor of shape [B, 2, 16]
        """
        # Encoder
        enc1_out = self.enc1(x)          # [B, 4, 8]
        
        # Bottleneck
        bneck_out = self.bottleneck(enc1_out)  # [B, 8, 4]
        
        # Decoder with skip connection
        up1 = self.upsample1(bneck_out)   # [B, 8, 8]
        dec1_out = self.dec1(up1)         # [B, 4, 8]
        
        # Skip connection (additive)
        skip_out = dec1_out + enc1_out    # [B, 4, 8]
        
        # Final upsampling and output
        up2 = self.upsample2(skip_out)    # [B, 4, 16]
        out = self.out_conv(up2)          # [B, 2, 16]
        out = self.tanh(out)              # [B, 2, 16]
        
        return out
    
    def get_layer_info(self) -> List[dict]:
        """Get information about each layer for documentation."""
        return [
            {"name": "enc1", "in_ch": 2, "out_ch": 4, "stride": 2, "length": 8},
            {"name": "bottleneck", "in_ch": 4, "out_ch": 8, "stride": 2, "length": 4},
            {"name": "upsample1", "scale": 2, "length": 8},
            {"name": "dec1", "in_ch": 8, "out_ch": 4, "stride": 1, "length": 8},
            {"name": "skip_add", "channels": 4, "length": 8},
            {"name": "upsample2", "scale": 2, "length": 16},
            {"name": "out_conv", "in_ch": 4, "out_ch": 2, "stride": 1, "length": 16},
            {"name": "tanh", "length": 16},
        ]
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_macs(self) -> int:
        """Estimate multiply-accumulate operations per forward pass."""
        # enc1: 3 * 2 * 4 * 8 = 192
        # bottleneck: 3 * 4 * 8 * 4 = 384
        # dec1: 3 * 8 * 4 * 8 = 768
        # out_conv: 3 * 4 * 2 * 16 = 384
        return 192 + 384 + 768 + 384


# Alias for backward compatibility
UNetGenerator = MiniGenerator


def create_generator(config: dict = None) -> MiniGenerator:
    """Factory function to create generator from config."""
    if config is None:
        config = {}
    
    return MiniGenerator(
        input_channels=config.get('input_channels', 2),
        output_channels=config.get('output_channels', 2),
        frame_length=config.get('frame_length', 16),
        leaky_slope=config.get('leaky_slope', 0.2)
    )


if __name__ == "__main__":
    # Test the model
    model = MiniGenerator()
    print(f"Mini Generator Architecture")
    print(f"=" * 50)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Estimated MACs: {model.estimate_macs():,}")
    print()
    
    # Test forward pass
    x = torch.randn(1, 2, 16)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
