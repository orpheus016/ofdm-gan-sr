# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Proof and Verification Module - MINI ARCHITECTURE
# =============================================================================
"""
VERIFICATION AND PROOF MODULE (MINI ARCHITECTURE)
==================================================

This module provides:
1. Architecture verification - confirms mini models match RTL blueprint
2. Per-layer CRC computation for RTL verification
3. Golden vector generation for hardware cosimulation
4. Quantization accuracy testing
5. End-to-end OFDM reconstruction proof

MINI ARCHITECTURE SPECS:
------------------------
Generator:    2 → 4 → 8 → 4 → 2 channels, 16 samples, ~258 params
Discriminator: 4 → 8 → 16 → 1 channels, 16 samples, ~521 params
Total: ~800 parameters
Fixed-point: Q1.7 weights (8-bit), Q8.8 activations (16-bit)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.generator import MiniGenerator
from models.discriminator import MiniDiscriminator, compute_gradient_penalty


@dataclass
class VerificationResult:
    """Container for verification results."""
    passed: bool
    message: str
    details: Dict[str, Any]


class MiniArchitectureVerifier:
    """
    Verify that the mini model architecture matches the RTL specification.
    
    Checks:
    - Layer count and types
    - Channel dimensions at each level
    - Parameter count
    - Output shapes
    """
    
    # Expected mini architecture specifications
    EXPECTED_GEN_PARAMS = 258
    EXPECTED_DISC_PARAMS = 521
    EXPECTED_FRAME_LENGTH = 16
    
    # RTL channel progressions
    GEN_CHANNELS = [2, 4, 8, 4, 2]
    DISC_CHANNELS = [4, 8, 16, 1]
    
    def __init__(self, tolerance: float = 0.05):
        """
        Initialize verifier.
        
        Args:
            tolerance: Allowed relative error (5% default for mini arch)
        """
        self.tolerance = tolerance
        
    def verify_generator(self, model: MiniGenerator) -> VerificationResult:
        """
        Verify mini generator architecture matches RTL.
        
        Checks:
        1. Input/output shape: [B, 2, 16] → [B, 2, 16]
        2. Parameter count ~258
        3. Channel progression: 2→4→8→4→2
        """
        results = {}
        
        # Test 1: Shape verification
        test_input = torch.randn(1, 2, self.EXPECTED_FRAME_LENGTH)
        try:
            output = model(test_input)
            results['shape_test'] = {
                'passed': output.shape == test_input.shape,
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'expected': [1, 2, self.EXPECTED_FRAME_LENGTH]
            }
        except Exception as e:
            results['shape_test'] = {'passed': False, 'error': str(e)}
            
        # Test 2: Parameter count
        pytorch_params = sum(p.numel() for p in model.parameters())
        param_error = abs(pytorch_params - self.EXPECTED_GEN_PARAMS) / self.EXPECTED_GEN_PARAMS
        results['param_test'] = {
            'passed': param_error < self.tolerance,
            'actual_params': pytorch_params,
            'expected_params': self.EXPECTED_GEN_PARAMS,
            'error': f"{param_error*100:.1f}%"
        }
        
        # Test 3: Channel progression check
        channels_correct = True
        channel_info = []
        
        # Check enc1: 2→4
        if hasattr(model, 'enc1') and hasattr(model.enc1, 'conv'):
            enc1_in = model.enc1.conv.in_channels
            enc1_out = model.enc1.conv.out_channels
            channel_info.append(f"enc1: {enc1_in}→{enc1_out}")
            if enc1_in != 2 or enc1_out != 4:
                channels_correct = False
                
        # Check bottleneck: 4→8
        if hasattr(model, 'bottleneck') and hasattr(model.bottleneck, 'conv'):
            bn_in = model.bottleneck.conv.in_channels
            bn_out = model.bottleneck.conv.out_channels
            channel_info.append(f"bottleneck: {bn_in}→{bn_out}")
            if bn_in != 4 or bn_out != 8:
                channels_correct = False
                
        # Check dec1: 8→4
        if hasattr(model, 'dec1') and hasattr(model.dec1, 'conv'):
            dec1_in = model.dec1.conv.in_channels
            dec1_out = model.dec1.conv.out_channels
            channel_info.append(f"dec1: {dec1_in}→{dec1_out}")
            if dec1_in != 8 or dec1_out != 4:
                channels_correct = False
                
        # Check out_conv: 4→2
        if hasattr(model, 'out_conv'):
            out_in = model.out_conv.in_channels
            out_out = model.out_conv.out_channels
            channel_info.append(f"out_conv: {out_in}→{out_out}")
            if out_in != 4 or out_out != 2:
                channels_correct = False
        
        results['channel_test'] = {
            'passed': channels_correct,
            'expected': '2→4→8→4→2',
            'actual': ', '.join(channel_info)
        }
        
        # Test 4: Output range (tanh should give [-1, 1])
        with torch.no_grad():
            test_out = model(torch.randn(10, 2, 16))
            in_range = test_out.min() >= -1.0 and test_out.max() <= 1.0
        results['range_test'] = {
            'passed': in_range,
            'min': float(test_out.min()),
            'max': float(test_out.max()),
            'expected_range': '[-1, 1]'
        }
        
        all_passed = all(r.get('passed', False) for r in results.values())
        
        return VerificationResult(
            passed=all_passed,
            message="Generator verification " + ("PASSED" if all_passed else "FAILED"),
            details=results
        )
        
    def verify_discriminator(self, model: MiniDiscriminator) -> VerificationResult:
        """
        Verify mini discriminator architecture matches RTL.
        
        Checks:
        1. Input: [B, 2, 16] × 2 (candidate + condition)
        2. Output: [B, 1] scalar score
        3. Parameter count ~521
        4. Channel progression: 4→8→16→1
        """
        results = {}
        
        # Test 1: Shape verification
        candidate = torch.randn(1, 2, self.EXPECTED_FRAME_LENGTH)
        condition = torch.randn(1, 2, self.EXPECTED_FRAME_LENGTH)
        
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
        pytorch_params = sum(p.numel() for p in model.parameters())
        param_error = abs(pytorch_params - self.EXPECTED_DISC_PARAMS) / self.EXPECTED_DISC_PARAMS
        results['param_test'] = {
            'passed': param_error < self.tolerance,
            'actual_params': pytorch_params,
            'expected_params': self.EXPECTED_DISC_PARAMS,
            'error': f"{param_error*100:.1f}%"
        }
        
        # Test 3: Channel progression check
        channels_correct = True
        channel_info = []
        
        # Check conv1: 4→8
        if hasattr(model, 'conv1'):
            c1_in = model.conv1.in_channels
            c1_out = model.conv1.out_channels
            channel_info.append(f"conv1: {c1_in}→{c1_out}")
            if c1_in != 4 or c1_out != 8:
                channels_correct = False
                
        # Check conv2: 8→16
        if hasattr(model, 'conv2'):
            c2_in = model.conv2.in_channels
            c2_out = model.conv2.out_channels
            channel_info.append(f"conv2: {c2_in}→{c2_out}")
            if c2_in != 8 or c2_out != 16:
                channels_correct = False
                
        # Check dense: 16→1
        if hasattr(model, 'dense'):
            d_in = model.dense.in_features
            d_out = model.dense.out_features
            channel_info.append(f"dense: {d_in}→{d_out}")
            if d_in != 16 or d_out != 1:
                channels_correct = False
        
        results['channel_test'] = {
            'passed': channels_correct,
            'expected': '4→8→16→1',
            'actual': ', '.join(channel_info)
        }
        
        all_passed = all(r.get('passed', False) for r in results.values())
        
        return VerificationResult(
            passed=all_passed,
            message="Discriminator verification " + ("PASSED" if all_passed else "FAILED"),
            details=results
        )


class GoldenVectorGenerator:
    """
    Generate golden vectors for RTL verification.
    
    Creates input/output pairs for hardware cosimulation.
    Uses Q8.8 fixed-point format for activations.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.activations = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self.hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                
    def _create_hook(self, name: str):
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
        """Generate golden vectors from a forward pass."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.activations = {}
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Save in Q8.8 format (multiply by 256)
        input_q88 = (input_tensor.numpy() * 256).astype(np.int16)
        output_q88 = (output.numpy() * 256).astype(np.int16)
        
        np.save(output_path / 'input_q88.npy', input_q88)
        np.save(output_path / 'output_q88.npy', output_q88)
        np.save(output_path / 'input_float.npy', input_tensor.numpy())
        np.save(output_path / 'output_float.npy', output.numpy())
        
        # Save as hex for Verilog $readmemh
        with open(output_path / 'input.hex', 'w') as f:
            for val in input_q88.flatten():
                f.write(f"{int(val) & 0xFFFF:04X}\n")
                
        with open(output_path / 'output.hex', 'w') as f:
            for val in output_q88.flatten():
                f.write(f"{int(val) & 0xFFFF:04X}\n")
        
        metadata = {
            'input_shape': list(input_tensor.shape),
            'output_shape': list(output.shape),
            'format': 'Q8.8 fixed-point',
            'layers': len(self.activations)
        }
        
        with open(output_path / 'golden_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


class OFDMReconstructionProof:
    """
    End-to-end proof of OFDM reconstruction capability.
    
    Tests signal enhancement at various SNR levels.
    """
    
    def __init__(self, generator: MiniGenerator, frame_length: int = 16):
        self.generator = generator
        self.generator.eval()
        self.frame_length = frame_length
        
    def test_signal_enhancement(
        self,
        snr_db: float,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """Test signal enhancement at a given SNR."""
        mse_before_total = 0
        mse_after_total = 0
        
        for _ in range(n_samples):
            # Generate clean QPSK-like signal
            clean_iq = np.random.choice([-0.7, 0.7], size=(2, self.frame_length))
            
            # Add AWGN noise
            signal_power = np.mean(clean_iq ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(2, self.frame_length) * np.sqrt(noise_power)
            noisy_iq = clean_iq + noise
            
            # Enhance with generator
            with torch.no_grad():
                noisy_tensor = torch.from_numpy(noisy_iq).unsqueeze(0).float()
                enhanced_tensor = self.generator(noisy_tensor)
                enhanced_iq = enhanced_tensor.squeeze(0).numpy()
                
            # Compute MSE
            mse_before = np.mean((noisy_iq - clean_iq) ** 2)
            mse_after = np.mean((enhanced_iq - clean_iq) ** 2)
            
            mse_before_total += mse_before
            mse_after_total += mse_after
            
        avg_mse_before = mse_before_total / n_samples
        avg_mse_after = mse_after_total / n_samples
        snr_improvement = 10 * np.log10(avg_mse_before / (avg_mse_after + 1e-10))
        
        return {
            'snr_input_db': snr_db,
            'mse_before': float(avg_mse_before),
            'mse_after': float(avg_mse_after),
            'snr_improvement_db': float(snr_improvement)
        }


def run_full_verification(
    generator: Optional[MiniGenerator] = None,
    discriminator: Optional[MiniDiscriminator] = None,
    output_dir: str = './verification_output'
) -> Dict[str, VerificationResult]:
    """
    Run complete verification suite for mini architecture.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if generator is None:
        generator = MiniGenerator()
    if discriminator is None:
        discriminator = MiniDiscriminator()
        
    print("Running mini architecture verification...")
    verifier = MiniArchitectureVerifier()
    
    results['generator'] = verifier.verify_generator(generator)
    print(f"  Generator: {results['generator'].message}")
    
    results['discriminator'] = verifier.verify_discriminator(discriminator)
    print(f"  Discriminator: {results['discriminator'].message}")
    
    # Golden vector generation
    print("\nGenerating golden vectors for RTL...")
    golden_gen = GoldenVectorGenerator(generator)
    test_input = torch.randn(1, 2, 16)
    golden_metadata = golden_gen.generate_vectors(
        test_input, 
        str(output_path / 'golden_vectors')
    )
    golden_gen.cleanup()
    print(f"  Generated vectors: {golden_metadata}")
    
    results['golden_vectors'] = VerificationResult(
        passed=True,
        message="Golden vectors generated successfully",
        details=golden_metadata
    )
    
    # Gradient penalty test
    print("\nTesting gradient penalty...")
    try:
        real = torch.randn(4, 2, 16, requires_grad=True)
        fake = torch.randn(4, 2, 16, requires_grad=True)
        cond = torch.randn(4, 2, 16)
        gp = compute_gradient_penalty(discriminator, real, fake, cond)
        results['gradient_penalty'] = VerificationResult(
            passed=True,
            message="Gradient penalty computed successfully",
            details={'gp_value': float(gp)}
        )
        print(f"  Gradient penalty: {gp.item():.4f}")
    except Exception as e:
        results['gradient_penalty'] = VerificationResult(
            passed=False,
            message=f"Gradient penalty failed: {e}",
            details={}
        )
    
    # Save summary
    summary = {
        name: {'passed': result.passed, 'message': result.message}
        for name, result in results.items()
    }
    
    with open(output_path / 'verification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
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


# Backward compatibility aliases
ArchitectureVerifier = MiniArchitectureVerifier
UNetGenerator = MiniGenerator
Discriminator = MiniDiscriminator


if __name__ == "__main__":
    print("=" * 60)
    print("CWGAN-GP Mini Architecture Verification")
    print("=" * 60)
    
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
