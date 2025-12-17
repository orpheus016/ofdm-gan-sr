#!/usr/bin/env python3
"""
Quick validation test for the models to ensure they work correctly.
"""

import torch
import sys

def test_generator():
    """Test generator forward pass with correct dimensions."""
    print("Testing Generator...")
    from models.generator import UNetGenerator
    
    # Create model
    generator = UNetGenerator(input_channels=2, output_channels=2)
    generator.eval()
    
    # Test input: batch of 4, 2 channels (I/Q), 1024 samples
    batch_size = 4
    x = torch.randn(batch_size, 2, 1024)
    
    # Forward pass
    try:
        output = generator(x)
        assert output.shape == (batch_size, 2, 1024), f"Expected shape (4, 2, 1024), got {output.shape}"
        print(f"✓ Generator output shape: {output.shape}")
        print(f"✓ Generator output range: [{output.min():.3f}, {output.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Generator failed: {e}")
        return False

def test_discriminator():
    """Test discriminator forward pass."""
    print("\nTesting Discriminator...")
    from models.discriminator import ConditionalDiscriminator
    
    # Create model
    discriminator = ConditionalDiscriminator(input_channels=2)
    discriminator.eval()
    
    # Test inputs
    batch_size = 4
    candidate = torch.randn(batch_size, 2, 1024)  # Generated or real signal
    condition = torch.randn(batch_size, 2, 1024)  # Noisy input (condition)
    
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
        
        # Test learning rate parsing
        lr_g = float(config['training']['optimizer'].get('lr_generator', 1e-4))
        lr_d = float(config['training']['optimizer'].get('lr_discriminator', 1e-4))
        
        print(f"✓ Generator learning rate: {lr_g}")
        print(f"✓ Discriminator learning rate: {lr_d}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_parameter_count():
    """Test parameter counts match expected values."""
    print("\nTesting Parameter Counts...")
    from models.generator import UNetGenerator
    from models.discriminator import ConditionalDiscriminator
    
    generator = UNetGenerator()
    discriminator = ConditionalDiscriminator()
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"✓ Generator parameters: {g_params:,}")
    print(f"✓ Discriminator parameters: {d_params:,}")
    print(f"✓ Total parameters: {g_params + d_params:,}")
    
    # Rough sanity check (should be in millions)
    assert g_params > 1_000_000, "Generator seems too small"
    assert d_params > 500_000, "Discriminator seems too small"
    
    return True

def test_gradient_flow():
    """Test that gradients flow through both models."""
    print("\nTesting Gradient Flow...")
    from models.generator import UNetGenerator
    from models.discriminator import ConditionalDiscriminator
    
    generator = UNetGenerator()
    discriminator = ConditionalDiscriminator()
    
    # Create dummy data
    noisy = torch.randn(2, 2, 1024, requires_grad=True)
    clean = torch.randn(2, 2, 1024)
    
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

def main():
    """Run all tests."""
    print("=" * 60)
    print("OFDM-GAN-SR Model Validation Tests")
    print("=" * 60)
    
    tests = [
        test_generator,
        test_discriminator,
        test_training_config,
        test_parameter_count,
        test_gradient_flow
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
        print("✓ All tests passed! Models are ready for training.")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
