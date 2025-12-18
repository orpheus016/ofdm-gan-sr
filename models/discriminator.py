# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Discriminator Model: Mini 1D CNN Critic for FPGA Deployment
# =============================================================================
"""
MINI DISCRIMINATOR FOR RTL
==========================

This is a compact discriminator designed for FPGA implementation.
The architecture matches the RTL implementation in rtl/discriminator_mini.v

Architecture:
    Input [4×16] → Conv1 [8×8] → Conv2 [16×4] → SumPool [16] → Dense → Score [1]

Input: Concatenation of candidate signal (2ch) and condition signal (2ch)

Parameters:
    - Frame length: 16 samples
    - Channels: 4 → 8 → 16 → 1
    - Kernel size: 3
    - Stride: 2 for convolutions

CWGAN-GP Specifics:
    - No batch normalization (as per WGAN-GP paper)
    - LeakyReLU activations
    - Unbounded output (Wasserstein critic)

Fixed-Point (for FPGA):
    - Weights: Q1.7 (8-bit signed)
    - Activations: Q8.8 (16-bit signed)
    - Accumulator: Q16.16 (32-bit)

Total Parameters: ~600
Total MACs/Frame: ~4,000
"""

import torch
import torch.nn as nn
from typing import Tuple


class MiniDiscriminator(nn.Module):
    """
    Mini Discriminator (Critic) for CWGAN-GP.
    
    This matches the RTL implementation with:
    - 16-sample frame length
    - 4-channel conditional input (candidate + condition)
    - Channel progression: 4 → 8 → 16 → 1
    - Global sum pooling
    
    Layer Specifications:
    ---------------------
    | Layer   | In Ch | Out Ch | Stride | L_out | Params | MACs   |
    |---------|-------|--------|--------|-------|--------|--------|
    | Conv1   | 4     | 8      | 2      | 8     | 104    | 768    |
    | Conv2   | 8     | 16     | 2      | 4     | 400    | 1,536  |
    | SumPool | 16    | 16     | -      | 1     | 0      | 64     |
    | Dense   | 16    | 1      | -      | 1     | 17     | 16     |
    |---------|-------|--------|--------|-------|--------|--------|
    | TOTAL   |       |        |        |       | 521    | 2,384  |
    """
    
    def __init__(
        self,
        input_channels: int = 4,  # Candidate(2) + Condition(2)
        frame_length: int = 16,
        leaky_slope: float = 0.2
    ):
        super(MiniDiscriminator, self).__init__()
        
        self.input_channels = input_channels
        self.frame_length = frame_length
        
        # Conv1: 4 → 8 channels, stride=2, output length = 8
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        )
        
        # Conv2: 8 → 16 channels, stride=2, output length = 4
        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        )
        
        # LeakyReLU activation (no batch norm for WGAN-GP)
        self.lrelu = nn.LeakyReLU(negative_slope=leaky_slope)
        
        # Dense layer: 16 → 1 (after sum pooling)
        self.dense = nn.Linear(16, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        candidate: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the conditional discriminator.
        
        Args:
            candidate: Real or generated I/Q signal, shape [B, 2, 16]
            condition: Noisy I/Q signal (condition), shape [B, 2, 16]
            
        Returns:
            Validity score, shape [B, 1]
            
        Mathematical operation:
            1. Concatenate: combined = [candidate; condition] ∈ ℝ^(4×16)
            2. Conv1: features = LeakyReLU(Conv(combined)) ∈ ℝ^(8×8)
            3. Conv2: features = LeakyReLU(Conv(features)) ∈ ℝ^(16×4)
            4. Sum pool: pooled = Σ_n features[:,n] ∈ ℝ^16
            5. Dense: score = W·pooled + b ∈ ℝ^1
        """
        # Concatenate candidate and condition along channel dimension
        # Shape: [B, 2, 16] + [B, 2, 16] → [B, 4, 16]
        combined = torch.cat([candidate, condition], dim=1)
        
        # Conv1: [B, 4, 16] → [B, 8, 8]
        out = self.lrelu(self.conv1(combined))
        
        # Conv2: [B, 8, 8] → [B, 16, 4]
        out = self.lrelu(self.conv2(out))
        
        # Global sum pooling over temporal dimension
        # Shape: [B, 16, 4] → [B, 16]
        out = torch.sum(out, dim=2)
        
        # Final projection to scalar
        # Shape: [B, 16] → [B, 1]
        score = self.dense(out)
        
        return score
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_macs(self) -> int:
        """Estimate multiply-accumulate operations per forward pass."""
        # conv1: 3 * 4 * 8 * 8 = 768
        # conv2: 3 * 8 * 16 * 4 = 1536
        # sum_pool: 16 * 4 = 64
        # dense: 16 * 1 = 16
        return 768 + 1536 + 64 + 16


# Aliases for backward compatibility
Discriminator = MiniDiscriminator
ConditionalDiscriminator = MiniDiscriminator


def compute_gradient_penalty(
    discriminator: MiniDiscriminator,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    condition: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    The gradient penalty enforces the 1-Lipschitz constraint by penalizing
    gradients that deviate from norm 1.
    
    Args:
        discriminator: The discriminator model
        real_samples: Real I/Q signals, shape [B, 2, L]
        fake_samples: Generated I/Q signals, shape [B, 2, L]
        condition: Condition (noisy input), shape [B, 2, L]
        device: Device to use
        
    Returns:
        Gradient penalty scalar
        
    Mathematical operation:
        1. Sample ε ~ U(0,1) for interpolation
        2. Create interpolated samples: x̂ = ε·x_real + (1-ε)·x_fake
        3. Compute discriminator output: D(x̂, c)
        4. Compute gradients: ∇_{x̂} D(x̂, c)
        5. Penalty: GP = E[(||∇_{x̂}D(x̂,c)||_2 - 1)²]
    """
    if device is None:
        device = real_samples.device
        
    batch_size = real_samples.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated, condition)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients for norm computation
    gradients = gradients.view(batch_size, -1)
    
    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)
    
    # Gradient penalty: (||∇D|| - 1)²
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def create_discriminator(config: dict = None) -> MiniDiscriminator:
    """Factory function to create discriminator from config."""
    if config is None:
        config = {}
    
    return MiniDiscriminator(
        input_channels=config.get('input_channels', 4),
        frame_length=config.get('frame_length', 16),
        leaky_slope=config.get('leaky_slope', 0.2)
    )


if __name__ == "__main__":
    # Test the model
    model = MiniDiscriminator()
    print(f"Mini Discriminator Architecture")
    print(f"=" * 50)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Estimated MACs: {model.estimate_macs():,}")
    print()
    
    # Test forward pass
    candidate = torch.randn(1, 2, 16)
    condition = torch.randn(1, 2, 16)
    score = model(candidate, condition)
    print(f"Candidate shape: {candidate.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Output shape:    {score.shape}")
    print(f"Output value:    {score.item():.4f}")
    
    # Test gradient penalty
    real = torch.randn(4, 2, 16, requires_grad=True)
    fake = torch.randn(4, 2, 16, requires_grad=True)
    cond = torch.randn(4, 2, 16)
    gp = compute_gradient_penalty(model, real, fake, cond)
    print(f"\nGradient penalty: {gp.item():.4f}")
