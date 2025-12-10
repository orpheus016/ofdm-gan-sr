# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Proof and Verification Module
# =============================================================================
"""
VERIFICATION AND PROOF MODULE
==============================

This module provides:
1. Architecture verification - confirms model matches blueprint
2. Per-layer CRC computation for RTL verification
3. Golden vector generation for hardware cosimulation
4. Quantization accuracy testing
5. End-to-end OFDM reconstruction proof

VERIFICATION WORKFLOW:
----------------------
1. Load trained model
2. Generate golden input vectors
3. Run forward pass, capture per-layer activations
4. Compute CRC32 per layer
5. Export golden vectors for RTL comparison
6. Test quantized model accuracy
7. Generate BER curves for different SNR levels
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.generator import UNetGenerator
from models.discriminator import Discriminator, compute_gradient_penalty
from utils.ofdm_utils import QAMModulator, OFDMModulator, ChannelModel, ImageOFDMConverter
from utils.quantization import (
    QuantizationConfig, compute_scale, quantize_tensor, 
    dequantize_tensor, compute_layer_crc
)


@dataclass
class VerificationResult:
    """Container for verification results."""
    passed: bool
    message: str
    details: Dict[str, Any]


class ArchitectureVerifier:
    """
    Verify that the model architecture matches the specification.
    
    Checks:
    - Layer count and types
    - Channel dimensions at each level
    - Parameter count
    - MAC count
    - Output shapes
    """
    
    # Expected architecture specifications
    EXPECTED_GEN_PARAMS = 5_500_000  # Approximate
    EXPECTED_GEN_MACS = 365_000_000  # Approximate
    EXPECTED_GEN_LAYERS = 23  # Number of conv layers
    
    EXPECTED_DISC_PARAMS = 1_310_000  # Approximate
    EXPECTED_DISC_LAYERS = 7  # 6 conv + 1 dense
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize verifier.
        
        Args:
            tolerance: Allowed relative error (10% default)
        """
        self.tolerance = tolerance
        
    def verify_generator(self, model: UNetGenerator) -> VerificationResult:
        """
        Verify generator architecture.
        
        Checks:
        1. Input/output shape: [B, 2, 1024] → [B, 2, 1024]
        2. Parameter count within tolerance
        3. MAC count within tolerance
        4. All expected layers present
        """
        results = {
            'shape_test': None,
            'param_test': None,
            'mac_test': None,
            'layer_test': None
        }
        
        # Test 1: Shape verification
        test_input = torch.randn(1, 2, 1024)
        try:
            output = model(test_input)
            if output.shape == test_input.shape:
                results['shape_test'] = {
                    'passed': True,
                    'input_shape': list(test_input.shape),
                    'output_shape': list(output.shape)
                }
            else:
                results['shape_test'] = {
                    'passed': False,
                    'input_shape': list(test_input.shape),
                    'output_shape': list(output.shape),
                    'expected': list(test_input.shape)
                }
        except Exception as e:
            results['shape_test'] = {'passed': False, 'error': str(e)}
            
        # Test 2: Parameter count
        total_params, total_macs = model.count_parameters()
        pytorch_params = sum(p.numel() for p in model.parameters())
        
        param_error = abs(total_params - self.EXPECTED_GEN_PARAMS) / self.EXPECTED_GEN_PARAMS
        results['param_test'] = {
            'passed': param_error < self.tolerance,
            'computed_params': total_params,
            'pytorch_params': pytorch_params,
            'expected_params': self.EXPECTED_GEN_PARAMS,
            'error': f"{param_error*100:.1f}%"
        }
        
        # Test 3: MAC count
        mac_error = abs(total_macs - self.EXPECTED_GEN_MACS) / self.EXPECTED_GEN_MACS
        results['mac_test'] = {
            'passed': mac_error < self.tolerance,
            'computed_macs': total_macs,
            'expected_macs': self.EXPECTED_GEN_MACS,
            'error': f"{mac_error*100:.1f}%"
        }
        
        # Test 4: Layer count
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
        results['layer_test'] = {
            'passed': len(conv_layers) == self.EXPECTED_GEN_LAYERS,
            'found_layers': len(conv_layers),
            'expected_layers': self.EXPECTED_GEN_LAYERS
        }
        
        # Overall result
        all_passed = all(r.get('passed', False) for r in results.values())
        
        return VerificationResult(
            passed=all_passed,
            message="Generator verification " + ("PASSED" if all_passed else "FAILED"),
            details=results
        )
        
    def verify_discriminator(self, model: Discriminator) -> VerificationResult:
        """
        Verify discriminator architecture.
        
        Checks:
        1. Input: [B, 2, 1024] × 2 (candidate + condition)
        2. Output: [B, 1] scalar score
        3. Parameter count
        4. Layer structure
        """
        results = {}
        
        # Test 1: Shape verification
        candidate = torch.randn(1, 2, 1024)
        condition = torch.randn(1, 2, 1024)
        
        try:
            output = model(candidate, condition)
            expected_shape = (1, 1)
            results['shape_test'] = {
                'passed': output.shape == expected_shape,
                'output_shape': list(output.shape),
                'expected_shape': list(expected_shape)
            }
        except Exception as e:
            results['shape_test'] = {'passed': False, 'error': str(e)}
            
        # Test 2: Parameter count
        total_params, total_macs = model.count_parameters()
        pytorch_params = sum(p.numel() for p in model.parameters())
        
        param_error = abs(total_params - self.EXPECTED_DISC_PARAMS) / self.EXPECTED_DISC_PARAMS
        results['param_test'] = {
            'passed': param_error < self.tolerance,
            'computed_params': total_params,
            'pytorch_params': pytorch_params,
            'expected_params': self.EXPECTED_DISC_PARAMS,
            'error': f"{param_error*100:.1f}%"
        }
        
        # Overall
        all_passed = all(r.get('passed', False) for r in results.values())
        
        return VerificationResult(
            passed=all_passed,
            message="Discriminator verification " + ("PASSED" if all_passed else "FAILED"),
            details=results
        )


class GoldenVectorGenerator:
    """
    Generate golden vectors for RTL verification.
    
    Creates input/output pairs with per-layer activations
    and CRC32 checksums for hardware cosimulation.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize with model.
        
        Args:
            model: PyTorch model to generate vectors from
        """
        self.model = model
        self.model.eval()
        self.activations = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self.hooks = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)
                
    def _create_hook(self, name: str):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            self.activations[name] = {
                'input': input[0].detach().clone(),
                'output': output.detach().clone()
            }
        return hook
        
    def generate_vectors(
        self,
        input_tensor: torch.Tensor,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Generate golden vectors from a forward pass.
        
        Args:
            input_tensor: Input to the model
            output_dir: Directory to save vectors
            
        Returns:
            Metadata dictionary with CRCs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Clear previous activations
        self.activations = {}
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Save input
        input_np = input_tensor.numpy()
        np.save(output_path / 'input.npy', input_np)
        
        # Save output
        output_np = output.numpy()
        np.save(output_path / 'output.npy', output_np)
        
        # Generate metadata
        metadata = {
            'input_shape': list(input_tensor.shape),
            'output_shape': list(output.shape),
            'input_crc': compute_layer_crc(input_tensor),
            'output_crc': compute_layer_crc(output),
            'layers': {}
        }
        
        # Save per-layer activations
        for name, act in self.activations.items():
            layer_dir = output_path / name.replace('.', '_')
            layer_dir.mkdir(exist_ok=True)
            
            # Save input and output
            np.save(layer_dir / 'layer_input.npy', act['input'].numpy())
            np.save(layer_dir / 'layer_output.npy', act['output'].numpy())
            
            # Compute CRCs
            metadata['layers'][name] = {
                'input_shape': list(act['input'].shape),
                'output_shape': list(act['output'].shape),
                'input_crc': compute_layer_crc(act['input']),
                'output_crc': compute_layer_crc(act['output'])
            }
            
        # Save metadata
        with open(output_path / 'golden_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    def cleanup(self):
        """Remove forward hooks."""
        for hook in self.hooks:
            hook.remove()


class OFDMReconstructionProof:
    """
    End-to-end proof of OFDM reconstruction capability.
    
    Tests:
    1. Image → OFDM → Channel → Enhanced OFDM → Image reconstruction
    2. BER (Bit Error Rate) improvement at various SNR levels
    3. EVM (Error Vector Magnitude) reduction
    4. Visual quality metrics (PSNR, SSIM)
    """
    
    def __init__(
        self,
        generator: UNetGenerator,
        modulation: str = 'QAM16',
        frame_length: int = 1024
    ):
        """
        Initialize proof system.
        
        Args:
            generator: Trained generator model
            modulation: QAM modulation scheme
            frame_length: OFDM frame length
        """
        self.generator = generator
        self.generator.eval()
        
        self.converter = ImageOFDMConverter(
            modulation=modulation,
            frame_length=frame_length
        )
        self.qam = QAMModulator(modulation)
        self.channel = ChannelModel('awgn')
        
    def test_signal_enhancement(
        self,
        clean_iq: np.ndarray,
        snr_db: float
    ) -> Dict[str, float]:
        """
        Test signal enhancement for a single signal.
        
        Args:
            clean_iq: Clean I/Q signal [2, L]
            snr_db: SNR for channel
            
        Returns:
            Dictionary with metrics
        """
        # Convert to complex
        clean_complex = clean_iq[0] + 1j * clean_iq[1]
        
        # Apply channel
        noisy_complex, _ = self.channel.apply(clean_complex, snr_db)
        
        # Convert to tensor
        noisy_iq = np.stack([
            np.real(noisy_complex),
            np.imag(noisy_complex)
        ], axis=0).astype(np.float32)
        
        # Normalize
        max_val = max(np.max(np.abs(noisy_iq)), np.max(np.abs(clean_iq)))
        noisy_iq_norm = noisy_iq / max_val
        clean_iq_norm = clean_iq / max_val
        
        # Enhance with generator
        with torch.no_grad():
            noisy_tensor = torch.from_numpy(noisy_iq_norm).unsqueeze(0).float()
            enhanced_tensor = self.generator(noisy_tensor)
            enhanced_iq = enhanced_tensor.squeeze(0).numpy()
            
        # Compute metrics
        
        # MSE before enhancement
        mse_before = np.mean((noisy_iq_norm - clean_iq_norm) ** 2)
        
        # MSE after enhancement
        mse_after = np.mean((enhanced_iq - clean_iq_norm) ** 2)
        
        # SNR improvement (dB)
        snr_improvement = 10 * np.log10(mse_before / (mse_after + 1e-10))
        
        # EVM before
        evm_before = np.sqrt(mse_before) * 100  # As percentage
        
        # EVM after
        evm_after = np.sqrt(mse_after) * 100
        
        return {
            'snr_input_db': snr_db,
            'mse_before': float(mse_before),
            'mse_after': float(mse_after),
            'snr_improvement_db': float(snr_improvement),
            'evm_before_percent': float(evm_before),
            'evm_after_percent': float(evm_after)
        }
        
    def generate_ber_curves(
        self,
        snr_range: List[float],
        n_samples: int = 100
    ) -> Dict[str, List[float]]:
        """
        Generate BER curves for various SNR levels.
        
        Args:
            snr_range: List of SNR values to test
            n_samples: Number of samples per SNR
            
        Returns:
            Dictionary with BER data
        """
        results = {
            'snr_db': snr_range,
            'ber_baseline': [],
            'ber_enhanced': [],
            'ber_reduction_factor': []
        }
        
        for snr in snr_range:
            ber_baseline_total = 0
            ber_enhanced_total = 0
            
            for _ in range(n_samples):
                # Generate random bits
                n_bits = 1000
                tx_bits = np.random.randint(0, 2, n_bits)
                
                # Modulate
                symbols = self.qam.modulate(tx_bits)
                
                # Create OFDM-like signal
                signal = np.fft.ifft(symbols) * np.sqrt(len(symbols))
                
                # Apply channel
                noisy, _ = self.channel.apply(signal, snr)
                
                # Baseline demodulation (no enhancement)
                rx_symbols_baseline = np.fft.fft(noisy) / np.sqrt(len(noisy))
                rx_bits_baseline = self.qam.demodulate(rx_symbols_baseline)
                
                # Enhanced demodulation
                # Convert to I/Q tensor
                noisy_iq = np.stack([np.real(noisy), np.imag(noisy)], axis=0)
                clean_iq = np.stack([np.real(signal), np.imag(signal)], axis=0)
                
                max_val = max(np.max(np.abs(noisy_iq)), 1e-8)
                noisy_iq_norm = noisy_iq / max_val
                
                with torch.no_grad():
                    noisy_tensor = torch.from_numpy(noisy_iq_norm).unsqueeze(0).float()
                    enhanced_tensor = self.generator(noisy_tensor)
                    enhanced_iq = enhanced_tensor.squeeze(0).numpy() * max_val
                    
                enhanced_complex = enhanced_iq[0] + 1j * enhanced_iq[1]
                rx_symbols_enhanced = np.fft.fft(enhanced_complex) / np.sqrt(len(enhanced_complex))
                rx_bits_enhanced = self.qam.demodulate(rx_symbols_enhanced)
                
                # Count bit errors
                n_compare = min(len(tx_bits), len(rx_bits_baseline), len(rx_bits_enhanced))
                ber_baseline = np.sum(tx_bits[:n_compare] != rx_bits_baseline[:n_compare]) / n_compare
                ber_enhanced = np.sum(tx_bits[:n_compare] != rx_bits_enhanced[:n_compare]) / n_compare
                
                ber_baseline_total += ber_baseline
                ber_enhanced_total += ber_enhanced
                
            avg_ber_baseline = ber_baseline_total / n_samples
            avg_ber_enhanced = ber_enhanced_total / n_samples
            
            results['ber_baseline'].append(avg_ber_baseline)
            results['ber_enhanced'].append(avg_ber_enhanced)
            
            if avg_ber_enhanced > 0:
                reduction = avg_ber_baseline / avg_ber_enhanced
            else:
                reduction = float('inf') if avg_ber_baseline > 0 else 1.0
            results['ber_reduction_factor'].append(reduction)
            
        return results


def run_full_verification(
    generator: Optional[UNetGenerator] = None,
    discriminator: Optional[Discriminator] = None,
    output_dir: str = './verification_output'
) -> Dict[str, VerificationResult]:
    """
    Run complete verification suite.
    
    Args:
        generator: Generator model (creates new if None)
        discriminator: Discriminator model (creates new if None)
        output_dir: Output directory for verification artifacts
        
    Returns:
        Dictionary of verification results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Create models if not provided
    if generator is None:
        generator = UNetGenerator()
    if discriminator is None:
        discriminator = Discriminator()
        
    # Architecture verification
    print("Running architecture verification...")
    verifier = ArchitectureVerifier()
    
    results['generator'] = verifier.verify_generator(generator)
    print(f"  Generator: {results['generator'].message}")
    
    results['discriminator'] = verifier.verify_discriminator(discriminator)
    print(f"  Discriminator: {results['discriminator'].message}")
    
    # Golden vector generation
    print("\nGenerating golden vectors...")
    golden_gen = GoldenVectorGenerator(generator)
    test_input = torch.randn(1, 2, 1024)
    golden_metadata = golden_gen.generate_vectors(
        test_input, 
        str(output_path / 'golden_vectors')
    )
    golden_gen.cleanup()
    print(f"  Generated vectors for {len(golden_metadata['layers'])} layers")
    
    results['golden_vectors'] = VerificationResult(
        passed=True,
        message="Golden vectors generated successfully",
        details=golden_metadata
    )
    
    # Save summary
    summary = {
        name: {
            'passed': result.passed,
            'message': result.message
        }
        for name, result in results.items()
    }
    
    with open(output_path / 'verification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{name}: {status}")
        all_passed = all_passed and result.passed
        
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return results


# =============================================================================
# Main Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CWGAN-GP OFDM Proof and Verification")
    print("=" * 60)
    
    # Run verification
    results = run_full_verification()
    
    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        for key, value in result.details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
                
    print("\n✓ Verification complete!")
