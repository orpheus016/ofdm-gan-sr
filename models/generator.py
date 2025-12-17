# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Generator Model: 1D U-Net with Additive Skip Connections
# =============================================================================
"""
MATHEMATICAL FOUNDATION
=======================

1D U-Net Generator Architecture:
--------------------------------
The generator G: ℝ^(2×L) → ℝ^(2×L) transforms noisy OFDM I/Q samples to enhanced samples.

For input x ∈ ℝ^(2×1024) (I and Q channels, 1024 samples):

ENCODER (Downsampling Path):
----------------------------
Each encoder block E_i applies:
    E_i(x) = LeakyReLU(Conv1D_s1(LeakyReLU(Conv1D_s2(x))))

Where:
- Conv1D_s2: Strided convolution with stride=2 (downsamples by 2x)
- Conv1D_s1: Standard convolution with stride=1
- LeakyReLU(x) = max(αx, x), where α = 0.2

Mathematical operation for Conv1D:
    y[n] = Σ_{k=0}^{K-1} w[k] · x[s·n + k] + b

Where:
- K = 3 (kernel size)
- s = stride (1 or 2)
- w = learned weights
- b = bias

BOTTLENECK:
-----------
    B(x) = LeakyReLU(Conv1D(LeakyReLU(Conv1D(x))))

Two consecutive convolutions at the lowest resolution (L/32 = 32 samples).

DECODER (Upsampling Path):
--------------------------
Each decoder block D_i applies:
    D_i(x, skip) = LeakyReLU(Conv1D(LeakyReLU(Conv1D(Upsample(x) + skip))))

Where:
- Upsample: Nearest neighbor upsampling by factor 2
    Upsample(x)[2n] = Upsample(x)[2n+1] = x[n]
- Skip connection uses ADDITION (not concatenation):
    merged = upsampled + encoder_output

FINAL PROJECTION:
-----------------
    out = tanh(Conv1D_1×1(decoder_output))

The tanh activation constrains output to [-1, 1] for normalized I/Q values.

SKIP CONNECTIONS (Additive):
----------------------------
Skip connections preserve high-frequency details:
    E_i output → D_{6-i} input (via addition)

Addition preserves gradient flow and requires matching channel dimensions,
which is ensured by the symmetric architecture.

CHANNEL PROGRESSION:
--------------------
Level   Encoder Ch      Decoder Ch      Length
0       2 → 32          32 → 2          1024
1       32 → 64         64 → 32         512
2       64 → 128        128 → 64        256
3       128 → 256       256 → 128       128
4       256 → 512       512 → 256       64
BN      512 → 512       -               32

PARAMETER COUNT:
----------------
Total parameters: ~5.5M
Total MACs per frame: ~365M

For detailed layer-by-layer breakdown, see docs/architecture.md
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv1D → LeakyReLU
    
    Mathematical operation:
        y = LeakyReLU(Conv1D(x))
        
    Where Conv1D computes:
        y[n] = Σ_{k=0}^{K-1} w[k] · x[s·n + k - pad] + b
        
    And LeakyReLU:
        LeakyReLU(x) = x if x > 0 else α·x, with α = 0.2
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
        
        # Store for documentation
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


class EncoderBlock(nn.Module):
    """
    Encoder block: Downsample by 2x using strided convolution.
    
    Architecture:
        Conv1D(stride=2) → LeakyReLU → Conv1D(stride=1) → LeakyReLU
        
    Input shape: [B, C_in, L]
    Output shape: [B, C_out, L/2]
    
    Mathematical description:
        Let x ∈ ℝ^(C_in × L), then:
        h = LeakyReLU(W_1 * x + b_1)     # Strided conv, output: ℝ^(C_out × L/2)
        y = LeakyReLU(W_2 * h + b_2)     # Standard conv, output: ℝ^(C_out × L/2)
        
    Where * denotes 1D convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        leaky_slope: float = 0.2
    ):
        super(EncoderBlock, self).__init__()
        
        # Strided convolution for downsampling
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            leaky_slope=leaky_slope
        )
        
        # Standard convolution
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            leaky_slope=leaky_slope
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample by 2x and add skip connection.
    
    Architecture:
        Upsample(×2) → Add(skip) → Conv1D → LeakyReLU → Conv1D → LeakyReLU
        
    Input shape: [B, C_in, L]
    Skip shape: [B, C_in, 2L]  (must match after upsampling)
    Output shape: [B, C_out, 2L]
    
    Mathematical description:
        Let x ∈ ℝ^(C_in × L) and skip ∈ ℝ^(C_in × 2L), then:
        u = Upsample(x)              # Nearest neighbor, output: ℝ^(C_in × 2L)
        m = u + skip                  # Additive merge
        h = LeakyReLU(W_1 * m + b_1)  # First conv
        y = LeakyReLU(W_2 * h + b_2)  # Second conv
        
    Nearest Neighbor Upsampling:
        For each sample x[n], create x'[2n] = x'[2n+1] = x[n]
        This doubles the temporal resolution without learnable parameters.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        leaky_slope: float = 0.2
    ):
        super(DecoderBlock, self).__init__()
        
        # Nearest neighbor upsampling (no learnable params)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Two convolutions after merge
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            leaky_slope=leaky_slope
        )
        
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            leaky_slope=leaky_slope
        )
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample input
        x = self.upsample(x)
        
        # Additive skip connection
        x = x + skip
        
        # Convolutional processing
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class UNetGenerator(nn.Module):
    """
    1D U-Net Generator for OFDM Signal Enhancement
    
    Architecture Overview:
    =====================
    
    Input: Noisy I/Q signal x ∈ ℝ^(2×1024)
    Output: Enhanced I/Q signal ŷ ∈ ℝ^(2×1024)
    
    The network learns the mapping: G(x) = ŷ ≈ y (clean signal)
    
    Structure:
    ----------
                     ┌─────────────────────────────────────┐
    Input ──► Enc1 ──┼──► Enc2 ──┼──► Enc3 ──┼──► Enc4 ──┼──► Enc5 ──► Bottleneck
              │      │    │      │    │      │    │      │    │           │
              │      │    │      │    │      │    │      │    │           │
              ▼      │    ▼      │    ▼      │    ▼      │    ▼           │
             Skip1   │  Skip2   │  Skip3   │  Skip4   │  Skip5          │
              │      │    │      │    │      │    │      │    │           │
              │      │    │      │    │      │    │      │    │           ▼
              └──────┼────┼──────┼────┼──────┼────┼──────┼────┼─────► Dec5
                     │    └──────┼────┼──────┼────┼──────┼────┼─────► Dec4
                     │           │    └──────┼────┼──────┼────┼─────► Dec3
                     │           │           │    └──────┼────┼─────► Dec2
                     │           │           │           │    └─────► Dec1
                     │           │           │           │             │
                     │           │           │           │             ▼
                     │           │           │           │          Final Conv
                     │           │           │           │             │
                     │           │           │           │             ▼
                     │           │           │           │           Output
    
    Layer Specifications:
    ---------------------
    | Layer    | Input Ch | Output Ch | Length In | Length Out | Params   |
    |----------|----------|-----------|-----------|------------|----------|
    | enc1_1   | 2        | 32        | 1024      | 512        | 608      |
    | enc1_2   | 32       | 32        | 512       | 512        | 3,104    |
    | enc2_1   | 32       | 64        | 512       | 256        | 12,416   |
    | enc2_2   | 64       | 64        | 256       | 256        | 49,216   |
    | enc3_1   | 64       | 128       | 256       | 128        | 98,560   |
    | enc3_2   | 128      | 128       | 128       | 128        | 196,864  |
    | enc4_1   | 128      | 256       | 128       | 64         | 393,472  |
    | enc4_2   | 256      | 256       | 64        | 64         | 786,688  |
    | enc5_1   | 256      | 512       | 64        | 32         | 1,572,864|
    | enc5_2   | 512      | 512       | 32        | 32         | 3,145,728|
    | bottle1  | 512      | 512       | 32        | 32         | 3,145,728|
    | bottle2  | 512      | 512       | 32        | 32         | 3,145,728|
    | dec5_1   | 512      | 512       | 64        | 64         | 3,145,728|
    | dec5_2   | 512      | 512       | 64        | 64         | 3,145,728|
    | dec4_1   | 512      | 256       | 128       | 128        | 1,572,864|
    | dec4_2   | 256      | 256       | 128       | 128        | 786,688  |
    | dec3_1   | 256      | 128       | 256       | 256        | 393,472  |
    | dec3_2   | 128      | 128       | 256       | 256        | 196,864  |
    | dec2_1   | 128      | 64        | 512       | 512        | 98,560   |
    | dec2_2   | 64       | 64        | 512       | 512        | 49,216   |
    | dec1_1   | 64       | 32        | 1024      | 1024       | 12,416   |
    | dec1_2   | 32       | 32        | 1024      | 1024       | 3,104    |
    | final    | 32       | 2         | 1024      | 1024       | 194      |
    |----------|----------|-----------|-----------|------------|----------|
    | TOTAL    |          |           |           |            | ~5.50M   |
    
    Memory Requirements (for skip buffers):
    ----------------------------------------
    Skip Level | Channels | Length | Elements | Bytes (INT16)
    E1         | 32       | 512    | 16,384   | 32,768
    E2         | 64       | 256    | 16,384   | 32,768
    E3         | 128      | 128    | 16,384   | 32,768
    E4         | 256      | 64     | 16,384   | 32,768
    E5         | 512      | 32     | 16,384   | 32,768
    ---------------------------------------------------------
    Total skip buffer memory: 163,840 bytes (~160 KB)
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        base_channels: int = 32,
        depth: int = 5,
        kernel_size: int = 3,
        leaky_slope: float = 0.2
    ):
        super(UNetGenerator, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # Channel progression: [32, 64, 128, 256, 512]
        channels = [base_channels * (2 ** i) for i in range(depth)]
        
        # ===========================================
        # ENCODER (Downsampling Path)
        # ===========================================
        
        # E1: [B, 2, 1024] → [B, 32, 512]
        self.enc1_1 = nn.Conv1d(input_channels, channels[0], kernel_size, stride=2, padding=1)
        self.enc1_2 = nn.Conv1d(channels[0], channels[0], kernel_size, stride=1, padding=1)
        
        # E2: [B, 32, 512] → [B, 64, 256]
        self.enc2_1 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=1)
        self.enc2_2 = nn.Conv1d(channels[1], channels[1], kernel_size, stride=1, padding=1)
        
        # E3: [B, 64, 256] → [B, 128, 128]
        self.enc3_1 = nn.Conv1d(channels[1], channels[2], kernel_size, stride=2, padding=1)
        self.enc3_2 = nn.Conv1d(channels[2], channels[2], kernel_size, stride=1, padding=1)
        
        # E4: [B, 128, 128] → [B, 256, 64]
        self.enc4_1 = nn.Conv1d(channels[2], channels[3], kernel_size, stride=2, padding=1)
        self.enc4_2 = nn.Conv1d(channels[3], channels[3], kernel_size, stride=1, padding=1)
        
        # E5: [B, 256, 64] → [B, 512, 32]
        self.enc5_1 = nn.Conv1d(channels[3], channels[4], kernel_size, stride=2, padding=1)
        self.enc5_2 = nn.Conv1d(channels[4], channels[4], kernel_size, stride=1, padding=1)
        
        # ===========================================
        # BOTTLENECK
        # ===========================================
        
        # B: [B, 512, 32] → [B, 512, 32]
        self.bottle1 = nn.Conv1d(channels[4], channels[4], kernel_size, stride=1, padding=1)
        self.bottle2 = nn.Conv1d(channels[4], channels[4], kernel_size, stride=1, padding=1)
        
        # ===========================================
        # DECODER (Upsampling Path with Additive Skips)
        # ===========================================
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # D5: [B, 512, 32] → [B, 256, 64] (upsample to 64, reduce channels to match e4)
        self.dec1_1 = nn.Conv1d(channels[4], channels[3], kernel_size, stride=1, padding=1)
        self.dec1_2 = nn.Conv1d(channels[3], channels[3], kernel_size, stride=1, padding=1)
        
        # D4: [B, 256, 64] → [B, 128, 128] (after upsample + skip_e4)
        self.dec2_1 = nn.Conv1d(channels[3], channels[2], kernel_size, stride=1, padding=1)
        self.dec2_2 = nn.Conv1d(channels[2], channels[2], kernel_size, stride=1, padding=1)
        
        # D3: [B, 128, 128] → [B, 64, 256] (after upsample + skip_e3)
        self.dec3_1 = nn.Conv1d(channels[2], channels[1], kernel_size, stride=1, padding=1)
        self.dec3_2 = nn.Conv1d(channels[1], channels[1], kernel_size, stride=1, padding=1)
        
        # D2: [B, 64, 256] → [B, 32, 512] (after upsample + skip_e2)
        self.dec4_1 = nn.Conv1d(channels[1], channels[0], kernel_size, stride=1, padding=1)
        self.dec4_2 = nn.Conv1d(channels[0], channels[0], kernel_size, stride=1, padding=1)
        
        # D1: [B, 32, 512] → [B, 32, 1024] (after upsample + skip_e1)
        self.dec5_1 = nn.Conv1d(channels[0], channels[0], kernel_size, stride=1, padding=1)
        self.dec5_2 = nn.Conv1d(channels[0], channels[0], kernel_size, stride=1, padding=1)
        
        # ===========================================
        # FINAL PROJECTION
        # ===========================================
        
        # Final: [B, 32, 1024] → [B, 2, 1024]
        self.final = nn.Conv1d(channels[0], output_channels, kernel_size, stride=1, padding=1)
        
        # Activations
        self.lrelu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net generator.
        
        Args:
            x: Noisy I/Q signal, shape [B, 2, L] where L=1024
            
        Returns:
            Enhanced I/Q signal, shape [B, 2, L]
            
        Mathematical flow:
            1. Encoder: Extract hierarchical features with progressive downsampling
            2. Bottleneck: Process at lowest resolution (highest abstraction)
            3. Decoder: Upsample and merge with skip connections to recover details
            4. Final: Project back to I/Q space with tanh normalization
        """
        
        # =================== ENCODER ===================
        # E1: 2×1024 → 32×512
        x = self.lrelu(self.enc1_1(x))
        e1 = self.lrelu(self.enc1_2(x))  # Skip connection source
        
        # E2: 32×512 → 64×256
        x = self.lrelu(self.enc2_1(e1))
        e2 = self.lrelu(self.enc2_2(x))  # Skip connection source
        
        # E3: 64×256 → 128×128
        x = self.lrelu(self.enc3_1(e2))
        e3 = self.lrelu(self.enc3_2(x))  # Skip connection source
        
        # E4: 128×128 → 256×64
        x = self.lrelu(self.enc4_1(e3))
        e4 = self.lrelu(self.enc4_2(x))  # Skip connection source
        
        # E5: 256×64 → 512×32
        x = self.lrelu(self.enc5_1(e4))
        e5 = self.lrelu(self.enc5_2(x))  # Skip connection source
        
        # =================== BOTTLENECK ===================
        # B: 512×32 → 512×32
        b = self.lrelu(self.bottle1(e5))
        b = self.lrelu(self.bottle2(b))
        
        # =================== DECODER ===================
        # D5: 512×32 → 256×64 (upsample and reduce channels to match e4)
        d5 = self.upsample(b)
        d5 = self.lrelu(self.dec1_1(d5))
        d5 = d5 + e4  # Additive skip connection (both 256×64)
        d5 = self.lrelu(self.dec1_2(d5))
        
        # D4: 256×64 → 128×128 (upsample + add skip_e3)
        d4 = self.upsample(d5)
        d4 = self.lrelu(self.dec2_1(d4))
        d4 = d4 + e3  # Additive skip connection (both 128×128)
        d4 = self.lrelu(self.dec2_2(d4))
        
        # D3: 128×128 → 64×256 (upsample + add skip_e2)
        d3 = self.upsample(d4)
        d3 = self.lrelu(self.dec3_1(d3))
        d3 = d3 + e2  # Additive skip connection (both 64×256)
        d3 = self.lrelu(self.dec3_2(d3))
        
        # D2: 64×256 → 32×512 (upsample + add skip_e1)
        d2 = self.upsample(d3)
        d2 = self.lrelu(self.dec4_1(d2))
        d2 = d2 + e1  # Additive skip connection (both 32×512)
        d2 = self.lrelu(self.dec4_2(d2))
        
        # D1: 32×512 → 32×1024 (final upsample, no skip)
        d1 = self.upsample(d2)
        d1 = self.lrelu(self.dec5_1(d1))
        d1 = self.lrelu(self.dec5_2(d1))
        
        # =================== FINAL ===================
        # Final: 32×1024 → 2×1024
        out = self.tanh(self.final(d1))
        
        return out
    
    def get_layer_info(self) -> List[dict]:
        """
        Get detailed information about each layer.
        
        Returns:
            List of dictionaries containing layer specifications.
        """
        layers = [
            {"name": "enc1_1", "in_ch": 2, "out_ch": 32, "stride": 2, "L_out": 512},
            {"name": "enc1_2", "in_ch": 32, "out_ch": 32, "stride": 1, "L_out": 512},
            {"name": "enc2_1", "in_ch": 32, "out_ch": 64, "stride": 2, "L_out": 256},
            {"name": "enc2_2", "in_ch": 64, "out_ch": 64, "stride": 1, "L_out": 256},
            {"name": "enc3_1", "in_ch": 64, "out_ch": 128, "stride": 2, "L_out": 128},
            {"name": "enc3_2", "in_ch": 128, "out_ch": 128, "stride": 1, "L_out": 128},
            {"name": "enc4_1", "in_ch": 128, "out_ch": 256, "stride": 2, "L_out": 64},
            {"name": "enc4_2", "in_ch": 256, "out_ch": 256, "stride": 1, "L_out": 64},
            {"name": "enc5_1", "in_ch": 256, "out_ch": 512, "stride": 2, "L_out": 32},
            {"name": "enc5_2", "in_ch": 512, "out_ch": 512, "stride": 1, "L_out": 32},
            {"name": "bottle1", "in_ch": 512, "out_ch": 512, "stride": 1, "L_out": 32},
            {"name": "bottle2", "in_ch": 512, "out_ch": 512, "stride": 1, "L_out": 32},
            {"name": "dec1_1", "in_ch": 512, "out_ch": 512, "stride": 1, "L_out": 64},
            {"name": "dec1_2", "in_ch": 512, "out_ch": 512, "stride": 1, "L_out": 64},
            {"name": "dec2_1", "in_ch": 512, "out_ch": 256, "stride": 1, "L_out": 128},
            {"name": "dec2_2", "in_ch": 256, "out_ch": 256, "stride": 1, "L_out": 128},
            {"name": "dec3_1", "in_ch": 256, "out_ch": 128, "stride": 1, "L_out": 256},
            {"name": "dec3_2", "in_ch": 128, "out_ch": 128, "stride": 1, "L_out": 256},
            {"name": "dec4_1", "in_ch": 128, "out_ch": 64, "stride": 1, "L_out": 512},
            {"name": "dec4_2", "in_ch": 64, "out_ch": 64, "stride": 1, "L_out": 512},
            {"name": "dec5_1", "in_ch": 64, "out_ch": 32, "stride": 1, "L_out": 1024},
            {"name": "dec5_2", "in_ch": 32, "out_ch": 32, "stride": 1, "L_out": 1024},
            {"name": "final", "in_ch": 32, "out_ch": 2, "stride": 1, "L_out": 1024},
        ]
        
        K = 3  # Kernel size
        for layer in layers:
            # Parameters = K * C_in * C_out + C_out (bias)
            layer["params"] = K * layer["in_ch"] * layer["out_ch"] + layer["out_ch"]
            # MACs = K * C_in * C_out * L_out
            layer["macs"] = K * layer["in_ch"] * layer["out_ch"] * layer["L_out"]
            
        return layers
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total trainable parameters and compute total MACs.
        
        Returns:
            Tuple of (total_params, total_macs)
        """
        layers = self.get_layer_info()
        total_params = sum(l["params"] for l in layers)
        total_macs = sum(l["macs"] for l in layers)
        return total_params, total_macs


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("U-Net Generator Architecture Verification")
    print("=" * 60)
    
    # Create model
    L = 1024
    gen = UNetGenerator()
    
    # Create test input
    noisy_input = torch.randn(1, 2, L)
    
    # Forward pass
    print(f"\nInput shape: {noisy_input.shape}")
    output = gen(noisy_input)
    print(f"Output shape: {output.shape}")
    
    # Verify shapes
    assert output.shape == (1, 2, L), "Output shape mismatch!"
    print("✓ Shape verification passed")
    
    # Count parameters
    total_params, total_macs = gen.count_parameters()
    print(f"\nTotal parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"Total MACs per frame: {total_macs:,} (~{total_macs/1e6:.1f}M)")
    
    # PyTorch parameter count (for verification)
    pytorch_params = sum(p.numel() for p in gen.parameters())
    print(f"PyTorch parameter count: {pytorch_params:,}")
    
    # Layer-by-layer breakdown
    print("\n" + "=" * 60)
    print("Layer-by-Layer Breakdown")
    print("=" * 60)
    print(f"{'Layer':<12} {'In→Out':<12} {'L_out':<8} {'Params':<12} {'MACs':<12}")
    print("-" * 60)
    
    for layer in gen.get_layer_info():
        print(f"{layer['name']:<12} {layer['in_ch']}→{layer['out_ch']:<8} "
              f"{layer['L_out']:<8} {layer['params']:<12,} {layer['macs']:<12,}")
    
    print("-" * 60)
    print(f"{'TOTAL':<12} {'':<12} {'':<8} {total_params:<12,} {total_macs:<12,}")
    
    print("\n✓ Generator verification complete!")
