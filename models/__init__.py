# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Models Package Initialization (Mini Architecture for FPGA)
# =============================================================================

from .generator import MiniGenerator, UNetGenerator, ConvBlock
from .discriminator import MiniDiscriminator, Discriminator, compute_gradient_penalty

__all__ = [
    'MiniGenerator',
    'UNetGenerator',      # Alias for MiniGenerator
    'MiniDiscriminator',
    'Discriminator',      # Alias for MiniDiscriminator
    'ConvBlock',
    'compute_gradient_penalty'
]
