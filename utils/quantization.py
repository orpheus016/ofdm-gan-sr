# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Quantization Utilities for FPGA Deployment
# =============================================================================
"""
QUANTIZATION FOR FPGA DEPLOYMENT
================================

This module provides utilities for:
1. Quantization-Aware Training (QAT)
2. Post-Training Quantization (PTQ)
3. Weight export for FPGA

QUANTIZATION MATH:
------------------

Fixed-Point Representation:
    x_int = round(x_float * scale)
    x_float ≈ x_int / scale

Where scale = 2^(n-1) - 1 for signed n-bit integers.

Symmetric Quantization:
    scale = max(|x|) / (2^(n-1) - 1)
    x_quant = round(x / scale)
    x_dequant = x_quant * scale

Per-Channel Quantization:
    Each output channel has its own scale factor.
    scale_c = max(|W_c|) / (2^(n-1) - 1)

FPGA Data Types:
- Weights: INT8 (signed 8-bit)
- Activations: INT16 (signed 16-bit)
- Accumulators: INT32 (signed 32-bit)

INT8 Weight Range: [-128, 127]
INT16 Activation Range: [-32768, 32767]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Any
import struct
import json
from pathlib import Path
import hashlib


class QuantizationConfig:
    """Configuration for quantization parameters."""
    
    def __init__(
        self,
        weight_bits: int = 8,
        activation_bits: int = 16,
        accumulator_bits: int = 32,
        per_channel: bool = True
    ):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.accumulator_bits = accumulator_bits
        self.per_channel = per_channel
        
        # Calculate ranges
        self.weight_max = 2 ** (weight_bits - 1) - 1  # 127 for INT8
        self.weight_min = -(2 ** (weight_bits - 1))   # -128 for INT8
        self.activation_max = 2 ** (activation_bits - 1) - 1  # 32767 for INT16
        self.activation_min = -(2 ** (activation_bits - 1))   # -32768 for INT16


def compute_scale(
    tensor: torch.Tensor,
    n_bits: int,
    per_channel: bool = False,
    channel_dim: int = 0
) -> torch.Tensor:
    """
    Compute quantization scale factor.
    
    Mathematical formula:
        scale = max(|x|) / (2^(n-1) - 1)
    
    For per-channel:
        scale_c = max(|x_c|) / (2^(n-1) - 1)
    
    Args:
        tensor: Tensor to compute scale for
        n_bits: Number of bits for quantization
        per_channel: Whether to use per-channel scaling
        channel_dim: Dimension for channels
        
    Returns:
        Scale factor(s)
    """
    max_val = 2 ** (n_bits - 1) - 1
    
    if per_channel:
        # Compute max absolute value per channel
        dims = list(range(tensor.dim()))
        dims.remove(channel_dim)
        abs_max = tensor.abs().amax(dim=dims, keepdim=True)
    else:
        abs_max = tensor.abs().max()
        
    # Avoid division by zero
    abs_max = torch.clamp(abs_max, min=1e-8)
    
    scale = abs_max / max_val
    
    return scale


def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    n_bits: int
) -> torch.Tensor:
    """
    Quantize tensor to integer representation.
    
    Mathematical operation:
        x_int = round(clamp(x / scale, min_val, max_val))
    
    Args:
        tensor: Float tensor to quantize
        scale: Scale factor(s)
        n_bits: Number of bits
        
    Returns:
        Quantized tensor (still float dtype for gradient flow)
    """
    max_val = 2 ** (n_bits - 1) - 1
    min_val = -(2 ** (n_bits - 1))
    
    scaled = tensor / scale
    quantized = torch.round(scaled)
    quantized = torch.clamp(quantized, min_val, max_val)
    
    return quantized


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize tensor back to float.
    
    Mathematical operation:
        x_float = x_int * scale
    
    Args:
        quantized: Quantized tensor
        scale: Scale factor(s)
        
    Returns:
        Dequantized float tensor
    """
    return quantized * scale


class FakeQuantize(nn.Module):
    """
    Fake quantization for QAT.
    
    During training:
        1. Quantize weights/activations
        2. Dequantize immediately
        3. Use straight-through estimator for gradients
    
    This simulates quantization effects while maintaining gradients.
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        per_channel: bool = True,
        channel_dim: int = 0
    ):
        super().__init__()
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        
        # Running statistics for activation quantization
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('running_max', torch.tensor(0.0))
        self.momentum = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update running statistics
            with torch.no_grad():
                max_val = x.abs().max()
                self.running_max = (1 - self.momentum) * self.running_max + self.momentum * max_val
                self.scale = compute_scale(x, self.n_bits, self.per_channel, self.channel_dim)
        
        # Fake quantization
        quantized = quantize_tensor(x, self.scale, self.n_bits)
        dequantized = dequantize_tensor(quantized, self.scale)
        
        # Straight-through estimator: use dequantized for forward, original for backward
        return x + (dequantized - x).detach()


class QuantizedConv1d(nn.Module):
    """
    Quantization-aware 1D convolution.
    
    During forward pass:
        1. Quantize weights to INT8
        2. Quantize input activations to INT16
        3. Perform convolution (simulated INT32 accumulation)
        4. Quantize output to INT16
    """
    
    def __init__(
        self,
        conv: nn.Conv1d,
        config: QuantizationConfig
    ):
        super().__init__()
        self.conv = conv
        self.config = config
        
        self.weight_quantizer = FakeQuantize(
            n_bits=config.weight_bits,
            per_channel=config.per_channel,
            channel_dim=0
        )
        
        self.activation_quantizer = FakeQuantize(
            n_bits=config.activation_bits,
            per_channel=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input
        x_quant = self.activation_quantizer(x)
        
        # Quantize weights
        w_quant = self.weight_quantizer(self.conv.weight)
        
        # Convolution with quantized values
        out = nn.functional.conv1d(
            x_quant, w_quant,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
        
        return out


def export_weights_fpga(
    model: nn.Module,
    output_dir: str,
    config: QuantizationConfig = None
) -> Dict[str, Any]:
    """
    Export quantized weights for FPGA deployment.
    
    Exports:
    1. Quantized weights as binary files (INT8)
    2. Scale factors as JSON
    3. Layer metadata (shapes, strides, etc.)
    4. CRC32 checksums for verification
    
    Args:
        model: Trained PyTorch model
        output_dir: Output directory for weight files
        config: Quantization configuration
        
    Returns:
        Export metadata dictionary
    """
    if config is None:
        config = QuantizationConfig()
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'config': {
            'weight_bits': config.weight_bits,
            'activation_bits': config.activation_bits,
            'per_channel': config.per_channel
        },
        'layers': {}
    }
    
    # Export each layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            layer_info = _export_conv_layer(
                name, module, output_path, config
            )
            metadata['layers'][name] = layer_info
            
        elif isinstance(module, nn.Linear):
            layer_info = _export_linear_layer(
                name, module, output_path, config
            )
            metadata['layers'][name] = layer_info
            
    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Exported {len(metadata['layers'])} layers to {output_dir}")
    
    return metadata


def _export_conv_layer(
    name: str,
    layer: nn.Conv1d,
    output_path: Path,
    config: QuantizationConfig
) -> Dict[str, Any]:
    """Export a single Conv1d layer."""
    
    weight = layer.weight.detach()
    
    # Compute per-channel scale
    scale = compute_scale(weight, config.weight_bits, config.per_channel, channel_dim=0)
    
    # Quantize weights
    weight_quant = quantize_tensor(weight, scale, config.weight_bits)
    weight_int8 = weight_quant.to(torch.int8)
    
    # Flatten for export (out_ch, in_ch, kernel) -> 1D array
    # Ensure tensor is on CPU before converting to NumPy
    weight_flat = weight_int8.detach().cpu().numpy().flatten()
    
    # Save weight binary
    weight_file = f"{name.replace('.', '_')}_weights.bin"
    weight_path = output_path / weight_file
    weight_flat.tofile(weight_path)
    
    # Compute CRC32
    crc = compute_crc32(weight_flat.tobytes())
    
    # Save scale factors
    scale_flat = scale.squeeze().detach().cpu().numpy()
    scale_file = f"{name.replace('.', '_')}_scale.bin"
    scale_path = output_path / scale_file
    scale_flat.astype(np.float32).tofile(scale_path)
    
    # Export bias if present
    bias_info = None
    if layer.bias is not None:
        bias = layer.bias.detach().cpu().numpy()
        bias_file = f"{name.replace('.', '_')}_bias.bin"
        bias_path = output_path / bias_file
        bias.astype(np.float32).tofile(bias_path)
        bias_info = {
            'file': bias_file,
            'shape': list(layer.bias.shape)
        }
        
    return {
        'type': 'Conv1d',
        'weight_file': weight_file,
        'scale_file': scale_file,
        'bias': bias_info,
        'weight_shape': list(weight.shape),
        'kernel_size': layer.kernel_size[0],
        'stride': layer.stride[0],
        'padding': layer.padding[0],
        'in_channels': layer.in_channels,
        'out_channels': layer.out_channels,
        'crc32': crc
    }


def _export_linear_layer(
    name: str,
    layer: nn.Linear,
    output_path: Path,
    config: QuantizationConfig
) -> Dict[str, Any]:
    """Export a single Linear layer."""
    
    weight = layer.weight.detach()
    
    # Compute scale
    scale = compute_scale(weight, config.weight_bits, config.per_channel, channel_dim=0)
    
    # Quantize
    weight_quant = quantize_tensor(weight, scale, config.weight_bits)
    weight_int8 = weight_quant.to(torch.int8)
    
    # Save
    weight_flat = weight_int8.detach().cpu().numpy().flatten()
    weight_file = f"{name.replace('.', '_')}_weights.bin"
    weight_path = output_path / weight_file
    weight_flat.tofile(weight_path)
    
    crc = compute_crc32(weight_flat.tobytes())
    
    # Scale
    scale_flat = scale.squeeze().detach().cpu().numpy()
    scale_file = f"{name.replace('.', '_')}_scale.bin"
    scale_path = output_path / scale_file
    scale_flat.astype(np.float32).tofile(scale_path)
    
    # Bias
    bias_info = None
    if layer.bias is not None:
        bias = layer.bias.detach().cpu().numpy()
        bias_file = f"{name.replace('.', '_')}_bias.bin"
        bias_path = output_path / bias_file
        bias.astype(np.float32).tofile(bias_path)
        bias_info = {
            'file': bias_file,
            'shape': list(layer.bias.shape)
        }
        
    return {
        'type': 'Linear',
        'weight_file': weight_file,
        'scale_file': scale_file,
        'bias': bias_info,
        'weight_shape': list(weight.shape),
        'in_features': layer.in_features,
        'out_features': layer.out_features,
        'crc32': crc
    }


def compute_crc32(data: bytes) -> str:
    """Compute CRC32 checksum of data."""
    import binascii
    crc = binascii.crc32(data) & 0xffffffff
    return f"{crc:08x}"


def compute_layer_crc(tensor: torch.Tensor) -> str:
    """
    Compute CRC32 of a tensor for verification.
    
    This is used to verify RTL implementation matches Python.
    """
    data = tensor.detach().cpu().numpy().tobytes()
    return compute_crc32(data)


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Quantization Utilities Verification")
    print("=" * 60)
    
    # Test quantization
    print("\n--- Basic Quantization Test ---")
    config = QuantizationConfig(weight_bits=8, activation_bits=16)
    
    # Create test tensor
    x = torch.randn(32, 64, 3)  # Conv weight-like
    
    # Compute scale
    scale = compute_scale(x, config.weight_bits, per_channel=True, channel_dim=0)
    print(f"Scale shape (per-channel): {scale.shape}")
    
    # Quantize
    x_quant = quantize_tensor(x, scale, config.weight_bits)
    print(f"Quantized range: [{x_quant.min():.0f}, {x_quant.max():.0f}]")
    print(f"Expected range: [{config.weight_min}, {config.weight_max}]")
    
    # Dequantize
    x_dequant = dequantize_tensor(x_quant, scale)
    error = (x - x_dequant).abs().mean()
    print(f"Quantization error (mean abs): {error:.6f}")
    
    # Test FakeQuantize
    print("\n--- Fake Quantization Test ---")
    fake_quant = FakeQuantize(n_bits=8, per_channel=True)
    y = fake_quant(x)
    print(f"FakeQuantize output shape: {y.shape}")
    print(f"Output requires_grad: {y.requires_grad}")
    
    # Test weight export
    print("\n--- Weight Export Test ---")
    
    # Create a small test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(2, 32, 3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64, 1)
            
    model = TestModel()
    
    # Export weights
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = export_weights_fpga(model, tmpdir, config)
        print(f"Exported layers: {list(metadata['layers'].keys())}")
        
        # Check files exist
        from pathlib import Path
        export_path = Path(tmpdir)
        files = list(export_path.glob('*.bin'))
        print(f"Binary files created: {len(files)}")
        
    print("\n✓ Quantization utilities verification complete!")
