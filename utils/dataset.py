# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Dataset Utilities: Data Generation, Loading, and Augmentation
# =============================================================================
"""
DATASET GENERATION FOR OFDM-GAN TRAINING
=========================================

Training Data Pipeline:
-----------------------
1. Load source images (for image transmission simulation)
2. Convert images to bits
3. Apply QAM modulation
4. Generate OFDM time-domain signals (clean)
5. Apply channel effects (noisy)
6. Create training pairs: (noisy, clean)

The generator learns: G(noisy) ≈ clean

Data Format:
------------
- Input (noisy): [2, L] I/Q signal with channel distortion
- Target (clean): [2, L] original clean I/Q signal
- Condition: [2, L] same as input (for conditional GAN)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image
from pathlib import Path

from .ofdm_utils import ImageOFDMConverter, ChannelModel


class OFDMDataset(Dataset):
    """
    Dataset for OFDM signal reconstruction training.
    
    Generates pairs of (noisy_signal, clean_signal) for training.
    
    The noisy signal is passed through a simulated wireless channel
    (AWGN, Rayleigh, etc.) while the clean signal is the original
    OFDM modulated data.
    """
    
    def __init__(
        self,
        image_dir: str,
        frame_length: int = 16,
        modulation: str = 'QPSK',
        n_subcarriers: int = 8,
        cp_length: int = 2,
        snr_range: Tuple[float, float] = (0, 30),
        channel_type: str = 'awgn',
        samples_per_image: int = 10,
        transform: Optional[callable] = None
    ):
        """
        Initialize OFDM dataset.
        
        Args:
            image_dir: Directory containing source images
            frame_length: OFDM frame length
            modulation: QAM modulation scheme
            n_subcarriers: OFDM subcarriers
            cp_length: Cyclic prefix length
            snr_range: Range of SNR values for training (min, max) in dB
            channel_type: Channel model type
            samples_per_image: Number of noisy samples to generate per image
            transform: Optional data augmentation transform
        """
        self.image_dir = Path(image_dir)
        self.frame_length = frame_length
        self.snr_range = snr_range
        self.channel_type = channel_type
        self.samples_per_image = samples_per_image
        self.transform = transform
        
        # Initialize converters
        self.converter = ImageOFDMConverter(
            modulation=modulation,
            n_subcarriers=n_subcarriers,
            cp_length=cp_length,
            frame_length=frame_length
        )
        self.channel = ChannelModel(channel_type)
        
        # Find all image files
        self.image_files = self._find_images()
        
        # Cache for clean signals (optional, saves computation)
        self._clean_signal_cache = {}
        
    def _find_images(self) -> List[Path]:
        """Find all image files in directory."""
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        images = []
        
        if self.image_dir.exists():
            for ext in extensions:
                images.extend(self.image_dir.glob(f'*{ext}'))
                images.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        return sorted(images)
        
    def __len__(self) -> int:
        return len(self.image_files) * self.samples_per_image
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - 'noisy': Noisy I/Q signal [2, L]
                - 'clean': Clean I/Q signal [2, L]
                - 'snr': SNR value used
        """
        # Determine which image and which sample
        image_idx = idx // self.samples_per_image
        
        # Get clean signal (with caching)
        if image_idx not in self._clean_signal_cache:
            image = self._load_image(self.image_files[image_idx])
            clean_iq, metadata = self.converter.image_to_ofdm(image)
            self._clean_signal_cache[image_idx] = (clean_iq, metadata)
        else:
            clean_iq, metadata = self._clean_signal_cache[image_idx]
            
        # Convert to complex for channel
        clean_complex = clean_iq[0] + 1j * clean_iq[1]
        clean_complex = clean_complex * metadata['normalization_factor']
        
        # Random SNR
        snr = np.random.uniform(*self.snr_range)
        
        # Apply channel
        noisy_complex, _ = self.channel.apply(clean_complex, snr)
        
        # Convert back to I/Q
        noisy_iq = np.stack([
            np.real(noisy_complex),
            np.imag(noisy_complex)
        ], axis=0).astype(np.float32)
        
        # Normalize
        max_val = max(np.max(np.abs(noisy_iq)), np.max(np.abs(clean_iq)))
        if max_val > 0:
            noisy_iq = noisy_iq / max_val
            clean_iq = clean_iq / max_val
            
        # Convert to tensors
        noisy_tensor = torch.from_numpy(noisy_iq).float()
        clean_tensor = torch.from_numpy(clean_iq).float()
        
        # Apply transforms if any
        if self.transform:
            noisy_tensor, clean_tensor = self.transform(noisy_tensor, clean_tensor)
            
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'snr': torch.tensor(snr, dtype=torch.float32)
        }
        
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = Image.open(path)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
            
        # Resize to standard size if needed
        if image.size[0] * image.size[1] > 4096:
            # Resize to fit in 64x64
            image = image.resize((64, 64), Image.Resampling.LANCZOS)
            
        return np.array(image)


class SyntheticOFDMDataset(Dataset):
    """
    Synthetic OFDM dataset for training without real images.
    
    Generates random QAM symbols, modulates them, and applies
    channel effects. Useful for initial training and testing.
    
    Enhanced with non-linear impairment support for LSI Contest.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        frame_length: int = 16,
        snr_range: Tuple[float, float] = (0, 30),
        channel_type: str = 'awgn',
        nonlinear: bool = False,
        pa_saturation: float = 1.0,
        iq_imbalance_db: float = 1.0,
        iq_phase_deg: float = 5.0,
        phase_noise_dbchz: float = -80
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            n_samples: Number of samples to generate
            frame_length: OFDM frame length
            snr_range: SNR range in dB
            channel_type: Channel model
            nonlinear: Enable non-linear impairments
            pa_saturation: PA saturation level (lower = more compression)
            iq_imbalance_db: IQ amplitude imbalance in dB
            iq_phase_deg: IQ phase imbalance in degrees
            phase_noise_dbchz: Phase noise power in dBc/Hz
        """
        self.n_samples = n_samples
        self.frame_length = frame_length
        self.snr_range = snr_range
        self.channel = ChannelModel(channel_type)
        
        # Non-linear impairment settings
        self.nonlinear = nonlinear
        self.pa_saturation = pa_saturation
        self.iq_imbalance_db = iq_imbalance_db
        self.iq_phase_deg = iq_phase_deg
        self.phase_noise_dbchz = phase_noise_dbchz
        
    def __len__(self) -> int:
        return self.n_samples
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a random training sample."""
        # Generate random complex signal (simulating QAM symbols after OFDM)
        # Using OFDM-like spectral characteristics
        
        # Random frequency-domain symbols
        n_freq = self.frame_length
        freq_symbols = np.random.randn(n_freq) + 1j * np.random.randn(n_freq)
        freq_symbols = freq_symbols / np.sqrt(2)  # Normalize
        
        # IFFT to get time-domain (OFDM-like signal)
        clean_complex = np.fft.ifft(freq_symbols) * np.sqrt(n_freq)
        
        # Apply non-linear impairments if enabled
        distorted_complex = clean_complex.copy()
        if self.nonlinear:
            from .ofdm_utils import NonLinearImpairments
            distorted_complex = NonLinearImpairments.apply_all(
                distorted_complex,
                pa_enabled=True,
                pa_saturation=self.pa_saturation,
                iq_imbalance_enabled=True,
                iq_amplitude_db=self.iq_imbalance_db,
                iq_phase_deg=self.iq_phase_deg,
                phase_noise_enabled=True,
                phase_noise_dbchz=self.phase_noise_dbchz,
                dc_offset_enabled=False,
                cfo_enabled=False
            )
        
        # Random SNR
        snr = np.random.uniform(*self.snr_range)
        
        # Apply channel
        noisy_complex, _ = self.channel.apply(distorted_complex, snr)
        
        # Convert to I/Q
        clean_iq = np.stack([
            np.real(clean_complex),
            np.imag(clean_complex)
        ], axis=0).astype(np.float32)
        
        noisy_iq = np.stack([
            np.real(noisy_complex),
            np.imag(noisy_complex)
        ], axis=0).astype(np.float32)
        
        # Normalize to [-1, 1]
        max_val = max(np.max(np.abs(noisy_iq)), np.max(np.abs(clean_iq)))
        if max_val > 0:
            noisy_iq = noisy_iq / max_val
            clean_iq = clean_iq / max_val
            
        return {
            'noisy': torch.from_numpy(noisy_iq).float(),
            'clean': torch.from_numpy(clean_iq).float(),
            'snr': torch.tensor(snr, dtype=torch.float32)
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


def generate_test_samples(
    n_samples: int = 100,
    frame_length: int = 16,
    snr_values: List[float] = [5, 10, 15, 20, 25],
    channel_type: str = 'awgn'
) -> Dict[float, List[Dict[str, torch.Tensor]]]:
    """
    Generate test samples at specific SNR values.
    
    Args:
        n_samples: Number of samples per SNR value
        frame_length: OFDM frame length
        snr_values: List of SNR values to test
        channel_type: Channel model
        
    Returns:
        Dictionary mapping SNR to list of test samples
    """
    channel = ChannelModel(channel_type)
    test_data = {}
    
    for snr in snr_values:
        samples = []
        for _ in range(n_samples):
            # Generate clean signal
            freq_symbols = np.random.randn(frame_length) + 1j * np.random.randn(frame_length)
            freq_symbols = freq_symbols / np.sqrt(2)
            clean_complex = np.fft.ifft(freq_symbols) * np.sqrt(frame_length)
            
            # Apply channel at specific SNR
            noisy_complex, _ = channel.apply(clean_complex, snr)
            
            # Convert to I/Q tensors
            clean_iq = np.stack([
                np.real(clean_complex),
                np.imag(clean_complex)
            ], axis=0).astype(np.float32)
            
            noisy_iq = np.stack([
                np.real(noisy_complex),
                np.imag(noisy_complex)
            ], axis=0).astype(np.float32)
            
            # Normalize
            max_val = max(np.max(np.abs(noisy_iq)), np.max(np.abs(clean_iq)))
            if max_val > 0:
                noisy_iq = noisy_iq / max_val
                clean_iq = clean_iq / max_val
                
            samples.append({
                'noisy': torch.from_numpy(noisy_iq).float(),
                'clean': torch.from_numpy(clean_iq).float(),
                'snr': snr
            })
            
        test_data[snr] = samples
        
    return test_data


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Utilities Verification")
    print("=" * 60)
    
    # Test synthetic dataset
    print("\n--- Synthetic Dataset Test ---")
    dataset = SyntheticOFDMDataset(n_samples=100, frame_length=16)
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Noisy shape: {sample['noisy'].shape}")
    print(f"Clean shape: {sample['clean'].shape}")
    print(f"SNR: {sample['snr'].item():.2f} dB")
    
    # Test dataloader
    print("\n--- DataLoader Test ---")
    loader = create_dataloader(dataset, batch_size=4, num_workers=0)
    
    batch = next(iter(loader))
    print(f"Batch noisy shape: {batch['noisy'].shape}")
    print(f"Batch clean shape: {batch['clean'].shape}")
    print(f"Batch SNR range: [{batch['snr'].min():.1f}, {batch['snr'].max():.1f}] dB")
    
    # Test test sample generation
    print("\n--- Test Sample Generation ---")
    test_data = generate_test_samples(
        n_samples=10,
        snr_values=[5, 10, 15, 20],
        channel_type='awgn'
    )
    
    for snr, samples in test_data.items():
        print(f"SNR {snr} dB: {len(samples)} samples")
        
    print("\n✓ Dataset utilities verification complete!")
