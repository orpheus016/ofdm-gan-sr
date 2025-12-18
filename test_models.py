#!/usr/bin/env python3
"""
Quick validation test for the MINI models to ensure they work correctly.
Tests the FPGA-targeted mini architecture.
"""

import torch
import sys

def test_generator():
    """Test mini generator forward pass with correct dimensions."""
    print("Testing Mini Generator...")
    from models.generator import MiniGenerator
    
    # Create model
    generator = MiniGenerator(input_channels=2, output_channels=2, frame_length=16)
    generator.eval()
    
    # Test input: batch of 4, 2 channels (I/Q), 16 samples (mini)
    batch_size = 4
    x = torch.randn(batch_size, 2, 16)
    
    # Forward pass
    try:
        output = generator(x)
        assert output.shape == (batch_size, 2, 16), f"Expected shape (4, 2, 16), got {output.shape}"
        print(f"✓ Generator output shape: {output.shape}")
        print(f"✓ Generator output range: [{output.min():.3f}, {output.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Generator failed: {e}")
        return False

def test_discriminator():
    """Test mini discriminator forward pass."""
    print("\nTesting Mini Discriminator...")
    from models.discriminator import MiniDiscriminator
    
    # Create model
    discriminator = MiniDiscriminator(input_channels=4, frame_length=16)
    discriminator.eval()
    
    # Test inputs
    batch_size = 4
    candidate = torch.randn(batch_size, 2, 16)  # Generated or real signal
    condition = torch.randn(batch_size, 2, 16)  # Noisy input (condition)
    
    # Forward pass
    try:
        score = discriminator(candidate, condition)
        assert score.shape == (batch_size, 1), f"Expected shape (4, 1), got {score.shape}"
        print(f"✓ Discriminator output shape: {score.shape}")
        print(f"✓ Discriminator score range: [{score.min():.3f}, {score.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Discriminator failed: {e}")
        return False

def test_training_config():
    """Test that training config loads correctly."""
    print("\nTesting Training Config...")
    import yaml
    from pathlib import Path
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test mini architecture settings
        frame_length = config['ofdm']['frame_length']
        assert frame_length == 16, f"Expected frame_length=16, got {frame_length}"
        
        # Test learning rate parsing
        lr_g = float(config['training']['optimizer'].get('lr_generator', 1e-4))
        lr_d = float(config['training']['optimizer'].get('lr_discriminator', 1e-4))
        
        print(f"✓ Frame length: {frame_length}")
        print(f"✓ Generator learning rate: {lr_g}")
        print(f"✓ Discriminator learning rate: {lr_d}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_parameter_count():
    """Test parameter counts are small enough for FPGA."""
    print("\nTesting Parameter Counts (Mini Architecture)...")
    from models.generator import MiniGenerator
    from models.discriminator import MiniDiscriminator
    
    generator = MiniGenerator()
    discriminator = MiniDiscriminator()
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"✓ Generator parameters: {g_params:,}")
    print(f"✓ Discriminator parameters: {d_params:,}")
    print(f"✓ Total parameters: {g_params + d_params:,}")
    
    # Mini architecture should be small (under 1000 params each)
    assert g_params < 1000, f"Generator too large for FPGA: {g_params}"
    assert d_params < 1000, f"Discriminator too large for FPGA: {d_params}"
    
    print(f"✓ Models fit FPGA constraints")
    return True

def test_gradient_flow():
    """Test that gradients flow through both models."""
    print("\nTesting Gradient Flow...")
    from models.generator import MiniGenerator
    from models.discriminator import MiniDiscriminator
    
    generator = MiniGenerator()
    discriminator = MiniDiscriminator()
    
    # Create dummy data (mini: 16 samples)
    noisy = torch.randn(2, 2, 16, requires_grad=True)
    clean = torch.randn(2, 2, 16)
    
    try:
        # Generator forward
        fake = generator(noisy)
        
        # Discriminator forward
        score_real = discriminator(clean, noisy)
        score_fake = discriminator(fake, noisy)
        
        # Compute dummy loss and backprop
        loss = score_fake.mean() - score_real.mean()
        loss.backward()
        
        # Check if gradients exist
        g_has_grad = any(p.grad is not None for p in generator.parameters())
        d_has_grad = any(p.grad is not None for p in discriminator.parameters())
        
        assert g_has_grad, "Generator has no gradients"
        assert d_has_grad, "Discriminator has no gradients"
        
        print("✓ Gradients flow correctly through both models")
        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return False

def test_rtl_compatibility():
    """Test that model matches RTL architecture."""
    print("\nTesting RTL Compatibility...")
    from models.generator import MiniGenerator
    from models.discriminator import MiniDiscriminator
    
    generator = MiniGenerator()
    discriminator = MiniDiscriminator()
    
    # Check generator architecture matches RTL
    # RTL: 2→4→8→4→2 channels
    try:
        assert generator.enc1.conv.in_channels == 2
        assert generator.enc1.conv.out_channels == 4
        assert generator.bottleneck.conv.in_channels == 4
        assert generator.bottleneck.conv.out_channels == 8
        assert generator.dec1.conv.in_channels == 8
        assert generator.dec1.conv.out_channels == 4
        assert generator.out_conv.in_channels == 4
        assert generator.out_conv.out_channels == 2
        print("✓ Generator channel progression: 2→4→8→4→2")
    except AssertionError as e:
        print(f"✗ Generator architecture mismatch: {e}")
        return False
    
    # Check discriminator architecture matches RTL
    # RTL: 4→8→16→1 channels
    try:
        assert discriminator.conv1.in_channels == 4
        assert discriminator.conv1.out_channels == 8
        assert discriminator.conv2.in_channels == 8
        assert discriminator.conv2.out_channels == 16
        assert discriminator.dense.in_features == 16
        assert discriminator.dense.out_features == 1
        print("✓ Discriminator channel progression: 4→8→16→1")
    except AssertionError as e:
        print(f"✗ Discriminator architecture mismatch: {e}")
        return False
    
    print("✓ Models match RTL architecture")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("OFDM-GAN-SR MINI Architecture Validation Tests")
    print("(Matches RTL implementation for FPGA deployment)")
    print("=" * 60)
    
    tests = [
        test_generator,
        test_discriminator,
        test_training_config,
        test_parameter_count,
        test_gradient_flow,
        test_rtl_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! Mini models ready for training.")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
