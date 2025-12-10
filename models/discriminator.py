# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Discriminator Model: Conditional 1D CNN Critic
# =============================================================================
"""
MATHEMATICAL FOUNDATION
=======================

Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP):
-------------------------------------------------------------

The discriminator (critic) D learns to distinguish between:
- Real pairs: (clean_signal, noisy_signal)  
- Fake pairs: (generated_signal, noisy_signal)

WASSERSTEIN DISTANCE:
---------------------
Unlike standard GAN discriminators that output probabilities,
the WGAN critic outputs unbounded real values (scores).

The Wasserstein-1 distance (Earth Mover's Distance) is:
    W(P_r, P_g) = sup_{||f||_L ≤ 1} E_{x~P_r}[f(x)] - E_{x~P_g}[f(x)]

Where ||f||_L ≤ 1 means f is 1-Lipschitz.

CRITIC OBJECTIVE:
-----------------
The critic maximizes:
    L_D = E[D(real, condition)] - E[D(G(condition), condition)] - λ·GP

Where:
- D(x, c) is the critic score for signal x given condition c
- G(c) is the generator output for noisy input c
- λ = 10 is the gradient penalty coefficient
- GP is the gradient penalty term

GRADIENT PENALTY:
-----------------
To enforce the 1-Lipschitz constraint, we penalize gradients:
    GP = E_{x̂}[(||∇_{x̂}D(x̂, c)||_2 - 1)²]

Where x̂ is an interpolation between real and fake:
    x̂ = ε·x_real + (1-ε)·x_fake, with ε ~ U(0,1)

This encourages ||∇D|| ≈ 1 everywhere, making D approximately 1-Lipschitz.

CONDITIONAL ARCHITECTURE:
-------------------------
The discriminator takes concatenated input:
    input = concat(candidate, condition) along channel dimension
    
Where:
- candidate ∈ ℝ^(2×L): Either real or generated I/Q signal
- condition ∈ ℝ^(2×L): The noisy I/Q input
- input ∈ ℝ^(4×L): Concatenated 4-channel signal

ARCHITECTURE:
-------------
    [4, 1024] → C1(s=2) → [32, 512]
             → C2(s=2) → [64, 256]
             → C3(s=2) → [128, 128]
             → C4(s=2) → [256, 64]
             → C5(s=2) → [512, 32]
             → C6(s=2) → [512, 16]
             → Global Sum Pool → [512]
             → Dense → [1]

CONVOLUTION OPERATION:
----------------------
For each convolutional layer:
    y[n] = LeakyReLU(Σ_{k=0}^{K-1} w[k] · x[s·n + k] + b)

With K=3, s=2 for all layers.

GLOBAL SUM POOLING:
-------------------
    z = Σ_{n=0}^{L-1} x[n]

Reduces temporal dimension to scalar per channel.
Sum pooling (not average) preserves magnitude information.

OUTPUT INTERPRETATION:
----------------------
Higher scores → critic believes input is real
Lower scores → critic believes input is fake

The generator tries to maximize D(G(z), c).
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class Discriminator(nn.Module):
    """
    Conditional 1D CNN Discriminator (Critic) for CWGAN-GP
    
    Architecture Overview:
    =====================
    
    Input: Concatenation of candidate (real/fake) and condition (noisy)
    Shape: [B, 4, L] where L=1024
    Output: Validity score [B, 1]
    
    The critic processes conditional pairs to score their realness.
    Higher score = more likely to be real.
    
    Layer Specifications:
    ---------------------
    | Layer | In Ch | Out Ch | Stride | L_out | Params    | MACs      |
    |-------|-------|--------|--------|-------|-----------|-----------|
    | C1    | 4     | 32     | 2      | 512   | 416       | 115,200   |
    | C2    | 32    | 64     | 2      | 256   | 6,208     | 786,432   |
    | C3    | 64    | 128    | 2      | 128   | 24,704    | 1,572,864 |
    | C4    | 128   | 256    | 2      | 64    | 98,560    | 3,145,728 |
    | C5    | 256   | 512    | 2      | 32    | 393,728   | 6,291,456 |
    | C6    | 512   | 512    | 2      | 16    | 786,944   | 6,291,456 |
    | Dense | 512   | 1      | -      | 1     | 513       | 512       |
    |-------|-------|--------|--------|-------|-----------|-----------|
    | TOTAL |       |        |        |       | ~1.31M    | ~18.2M    |
    
    Note: The discriminator input is 4 channels (not 2) because we concatenate
    the candidate signal (2 ch) with the condition signal (2 ch).
    """
    
    def __init__(
        self,
        input_channels: int = 4,  # Candidate(2) + Condition(2)
        base_channels: int = 32,
        num_layers: int = 6,
        kernel_size: int = 3,
        leaky_slope: float = 0.2
    ):
        super(Discriminator, self).__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Build convolutional layers
        layers = []
        in_ch = input_channels
        
        for i in range(num_layers):
            # Channel progression: 4→32→64→128→256→512→512
            if i == 0:
                out_ch = base_channels
            elif i < 5:
                out_ch = base_channels * (2 ** i)
            else:
                out_ch = 512  # Cap at 512
                
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=1))
            layers.append(nn.LeakyReLU(leaky_slope))
            in_ch = out_ch
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Final dense layer
        # After 6 strided convs with s=2, length: 1024→512→256→128→64→32→16
        self.final_channels = 512
        self.dense = nn.Linear(self.final_channels, 1)
        
    def forward(
        self, 
        candidate: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the conditional discriminator.
        
        Args:
            candidate: Real or generated I/Q signal, shape [B, 2, L]
            condition: Noisy I/Q signal (condition), shape [B, 2, L]
            
        Returns:
            Validity score, shape [B, 1]
            
        Mathematical operation:
            1. Concatenate: combined = [candidate; condition] ∈ ℝ^(4×L)
            2. Conv stack: features = Conv_stack(combined) ∈ ℝ^(512×16)
            3. Sum pool: pooled = Σ_n features[:,n] ∈ ℝ^512
            4. Dense: score = W·pooled + b ∈ ℝ^1
        """
        # Concatenate candidate and condition along channel dimension
        # Shape: [B, 2, L] + [B, 2, L] → [B, 4, L]
        combined = torch.cat([candidate, condition], dim=1)
        
        # Apply convolutional layers
        # Shape: [B, 4, L] → [B, 512, L/64]
        out = self.conv_layers(combined)
        
        # Global sum pooling over temporal dimension
        # Shape: [B, 512, L/64] → [B, 512]
        out = torch.sum(out, dim=2)
        
        # Final projection to scalar
        # Shape: [B, 512] → [B, 1]
        validity = self.dense(out)
        
        return validity
    
    def get_layer_info(self, input_length: int = 1024) -> List[dict]:
        """
        Get detailed information about each layer.
        
        Args:
            input_length: Length of input signal
            
        Returns:
            List of dictionaries containing layer specifications.
        """
        K = self.kernel_size
        layers = []
        
        # Channel progression
        channels = [self.input_channels, 32, 64, 128, 256, 512, 512]
        lengths = [input_length]
        for _ in range(self.num_layers):
            lengths.append(lengths[-1] // 2)
            
        for i in range(self.num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            L_out = lengths[i + 1]
            
            layers.append({
                "name": f"C{i+1}",
                "in_ch": in_ch,
                "out_ch": out_ch,
                "stride": 2,
                "L_out": L_out,
                "params": K * in_ch * out_ch + out_ch,
                "macs": K * in_ch * out_ch * L_out
            })
            
        # Dense layer
        layers.append({
            "name": "Dense",
            "in_ch": 512,
            "out_ch": 1,
            "stride": "-",
            "L_out": 1,
            "params": 512 + 1,  # weight + bias
            "macs": 512
        })
        
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


def compute_gradient_penalty(
    discriminator: Discriminator,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    condition: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    The gradient penalty enforces the 1-Lipschitz constraint on the critic.
    
    Mathematical formulation:
    -------------------------
    GP = E[(||∇_{x̂}D(x̂, c)||_2 - 1)²]
    
    Where:
        x̂ = ε·x_real + (1-ε)·x_fake
        ε ~ Uniform(0, 1)
    
    This penalizes the critic when its gradient norm deviates from 1.
    
    Args:
        discriminator: The discriminator model
        real_samples: Real I/Q signals, shape [B, 2, L]
        fake_samples: Generated I/Q signals, shape [B, 2, L]
        condition: Noisy I/Q signals (condition), shape [B, 2, L]
        device: Computation device
        
    Returns:
        Gradient penalty scalar value
        
    Implementation details:
    -----------------------
    1. Sample random ε from uniform distribution
    2. Create interpolated samples: x̂ = ε·real + (1-ε)·fake
    3. Forward pass through discriminator with gradient tracking
    4. Compute gradients w.r.t. interpolated samples
    5. Calculate L2 norm of gradients
    6. Return penalty = mean((||grad||_2 - 1)²)
    """
    batch_size = real_samples.size(0)
    
    # Sample random interpolation coefficients
    # Shape: [B, 1, 1] for broadcasting
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    
    # Create interpolated samples
    # x̂ = ε·real + (1-ε)·fake
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated = interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated, condition)
    
    # Create gradient outputs (ones)
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients: [B, 2, L] → [B, 2*L]
    gradients = gradients.view(batch_size, -1)
    
    # Compute L2 norm: ||∇D||_2
    gradient_norm = gradients.norm(2, dim=1)
    
    # Gradient penalty: E[(||∇D||_2 - 1)²]
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Discriminator Architecture Verification")
    print("=" * 60)
    
    # Create model
    L = 1024
    disc = Discriminator()
    
    # Create test inputs
    candidate = torch.randn(1, 2, L)  # Real or fake signal
    condition = torch.randn(1, 2, L)  # Noisy signal (condition)
    
    # Forward pass
    print(f"\nCandidate shape: {candidate.shape}")
    print(f"Condition shape: {condition.shape}")
    
    score = disc(candidate, condition)
    print(f"Output score shape: {score.shape}")
    
    # Verify shapes
    assert score.shape == (1, 1), "Output shape mismatch!"
    print("✓ Shape verification passed")
    
    # Count parameters
    total_params, total_macs = disc.count_parameters()
    print(f"\nTotal parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"Total MACs per frame: {total_macs:,} (~{total_macs/1e6:.1f}M)")
    
    # PyTorch parameter count (for verification)
    pytorch_params = sum(p.numel() for p in disc.parameters())
    print(f"PyTorch parameter count: {pytorch_params:,}")
    
    # Layer-by-layer breakdown
    print("\n" + "=" * 60)
    print("Layer-by-Layer Breakdown")
    print("=" * 60)
    print(f"{'Layer':<8} {'In→Out':<12} {'Stride':<8} {'L_out':<8} {'Params':<12} {'MACs':<12}")
    print("-" * 60)
    
    for layer in disc.get_layer_info():
        print(f"{layer['name']:<8} {layer['in_ch']}→{layer['out_ch']:<8} "
              f"{layer['stride']:<8} {layer['L_out']:<8} {layer['params']:<12,} {layer['macs']:<12,}")
    
    print("-" * 60)
    print(f"{'TOTAL':<8} {'':<12} {'':<8} {'':<8} {total_params:<12,} {total_macs:<12,}")
    
    # Test gradient penalty
    print("\n" + "=" * 60)
    print("Gradient Penalty Test")
    print("=" * 60)
    
    device = torch.device("cpu")
    real = torch.randn(4, 2, L, requires_grad=True)
    fake = torch.randn(4, 2, L, requires_grad=True)
    cond = torch.randn(4, 2, L)
    
    gp = compute_gradient_penalty(disc, real, fake, cond, device)
    print(f"Gradient penalty value: {gp.item():.4f}")
    print("✓ Gradient penalty computation successful")
    
    print("\n✓ Discriminator verification complete!")
