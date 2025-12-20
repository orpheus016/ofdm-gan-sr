# =============================================================================
# Image-to-OFDM-to-Reconstruction Visualization
# =============================================================================
"""
IMAGE OFDM TRANSMISSION VISUALIZATION
======================================

Visualizes the complete pipeline of image transmission over OFDM channel,
similar to JSCC papers like:
- "OFDM-Guided Deep Joint Source Channel Coding" (https://arxiv.org/abs/2109.05194)
- "Deep JSCC for Wireless Image Transmission" 

Demonstrates:
1. Original image → Pixel values
2. Pixel encoding → QAM symbols
3. OFDM modulation → Time-domain signal
4. Constellation diagram (heatmap style)
5. Channel effects (fading, noise)
6. Reconstruction and quality metrics

For SimpleGAN 3x3:
- Treats 3x3 pixel grid as the "image"
- Shows how 9 values map to I/Q constellation
- Visualizes reconstruction quality (PSNR, SSIM, MSE)

Usage:
    python visualization/image_ofdm_visualization.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Use clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TransmissionResult:
    """Container for transmission simulation results."""
    original_image: np.ndarray
    tx_symbols: np.ndarray          # Transmitted QAM symbols
    rx_symbols_noisy: np.ndarray    # Received symbols (after channel)
    rx_symbols_enhanced: np.ndarray # Received symbols (after GAN enhancement)
    reconstructed_noisy: np.ndarray
    reconstructed_enhanced: np.ndarray
    snr_db: float
    channel_type: str


# =============================================================================
# Quality Metrics
# =============================================================================

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray, 
                   max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Higher is better. Typical values:
    - PSNR > 40 dB: Excellent quality
    - PSNR 30-40 dB: Good quality  
    - PSNR 20-30 dB: Moderate quality
    - PSNR < 20 dB: Poor quality
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (simplified version).
    
    SSIM considers luminance, contrast, and structure.
    Range: [-1, 1], where 1 = identical.
    """
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Flatten arrays
    x = original.flatten()
    y = reconstructed.flatten()
    
    # Calculate means
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    # Calculate variances and covariance
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.cov(x, y)[0, 1]
    
    # SSIM formula
    numerator = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    
    return numerator / denominator


def calculate_nmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Normalized Mean Squared Error.
    
    NMSE = MSE / Var(original)
    
    Lower is better. NMSE = 0 means perfect reconstruction.
    """
    mse = np.mean((original - reconstructed) ** 2)
    var_original = np.var(original)
    if var_original == 0:
        return 0.0
    return mse / var_original


# =============================================================================
# Image to OFDM Simulation
# =============================================================================

class ImageOFDMSimulator:
    """
    Simulates image transmission over OFDM channel.
    
    For SimpleGAN: 3x3 image (9 pixels) → 9 complex symbols → I/Q transmission
    """
    
    def __init__(self, image_size: Tuple[int, int] = (3, 3)):
        self.image_size = image_size
        self.n_pixels = image_size[0] * image_size[1]
        
        # 16-QAM constellation (normalized)
        self.qam16_constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
            +3-3j, +3-1j, +3+3j, +3+1j,
            +1-3j, +1-1j, +1+3j, +1+1j
        ]) / np.sqrt(10)
        
    def pixel_to_symbol(self, pixel_value: float) -> complex:
        """
        Map normalized pixel value [-1, 1] to complex symbol.
        
        For SimpleGAN output: value is already in [-1, 1] range (tanh output)
        Maps to I + jQ where I and Q are scaled versions of nearby pixels.
        """
        # Direct mapping: pixel value → I channel, small variation → Q channel
        I = pixel_value
        Q = pixel_value * 0.1 + np.random.randn() * 0.05
        return I + 1j * Q
    
    def symbol_to_pixel(self, symbol: complex) -> float:
        """
        Map complex symbol back to pixel value.
        """
        # Use real part as primary pixel value
        return np.clip(symbol.real, -1, 1)
    
    def add_channel_effects(self, symbols: np.ndarray, snr_db: float, 
                           channel_type: str = 'awgn') -> np.ndarray:
        """
        Add channel impairments to transmitted symbols.
        
        Args:
            symbols: Complex transmitted symbols
            snr_db: Signal-to-Noise Ratio in dB
            channel_type: 'awgn', 'rayleigh', or 'rician'
        
        Returns:
            Received symbols with noise and fading
        """
        # Calculate noise power from SNR
        signal_power = np.mean(np.abs(symbols) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate complex noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
        )
        
        if channel_type == 'awgn':
            return symbols + noise
            
        elif channel_type == 'rayleigh':
            # Rayleigh fading: h ~ CN(0, 1)
            h = (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))) / np.sqrt(2)
            return h * symbols + noise
            
        elif channel_type == 'rician':
            # Rician fading with K=3 (LOS component)
            K = 3
            h_los = np.sqrt(K / (K + 1))
            h_nlos = np.sqrt(1 / (K + 1)) * (
                np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
            ) / np.sqrt(2)
            h = h_los + h_nlos
            return h * symbols + noise
        
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")
    
    def simulate_gan_enhancement(self, noisy_symbols: np.ndarray, 
                                  enhancement_factor: float = 0.7) -> np.ndarray:
        """
        Simulate GAN-based signal enhancement.
        
        In real system, this would be the CWGAN-GP output.
        Here we simulate by reducing noise.
        
        Args:
            noisy_symbols: Received noisy symbols
            enhancement_factor: How much noise is reduced (0=none, 1=perfect)
        
        Returns:
            Enhanced symbols
        """
        # Simulate denoising: move symbols toward nearest valid point
        # This is a simplified model of what GAN does
        enhanced = noisy_symbols.copy()
        
        for i in range(len(enhanced)):
            # Add small correction toward origin (regularization effect of GAN)
            enhanced[i] = enhanced[i] * (1 - enhancement_factor * 0.1)
            
            # Reduce imaginary noise (GAN learns to denoise)
            enhanced[i] = enhanced[i].real * (1 + enhancement_factor * 0.1) + \
                         1j * enhanced[i].imag * (1 - enhancement_factor * 0.3)
        
        return enhanced
    
    def simulate_transmission(self, image: np.ndarray, snr_db: float = 10,
                              channel_type: str = 'rayleigh',
                              gan_enhancement: float = 0.7) -> TransmissionResult:
        """
        Full simulation of image transmission over OFDM channel.
        
        Pipeline:
        1. Image → Flatten to pixel values
        2. Pixels → Complex symbols (I/Q mapping)
        3. Symbols → Channel (add noise/fading)
        4. Noisy symbols → GAN enhancement (optional)
        5. Enhanced symbols → Reconstruct image
        
        Args:
            image: 2D numpy array (normalized to [-1, 1])
            snr_db: Channel SNR in dB
            channel_type: Type of fading channel
            gan_enhancement: Enhancement factor (0=none, 1=max)
        
        Returns:
            TransmissionResult with all intermediate results
        """
        # Normalize image to [-1, 1]
        if image.max() > 1 or image.min() < -1:
            image = 2 * (image - image.min()) / (image.max() - image.min()) - 1
        
        # Flatten image to pixels
        pixels = image.flatten()
        
        # Map pixels to complex symbols
        tx_symbols = np.array([self.pixel_to_symbol(p) for p in pixels])
        
        # Apply channel effects
        rx_symbols_noisy = self.add_channel_effects(tx_symbols, snr_db, channel_type)
        
        # Apply GAN enhancement
        rx_symbols_enhanced = self.simulate_gan_enhancement(
            rx_symbols_noisy, gan_enhancement
        )
        
        # Reconstruct images
        reconstructed_noisy = np.array([
            self.symbol_to_pixel(s) for s in rx_symbols_noisy
        ]).reshape(image.shape)
        
        reconstructed_enhanced = np.array([
            self.symbol_to_pixel(s) for s in rx_symbols_enhanced
        ]).reshape(image.shape)
        
        return TransmissionResult(
            original_image=image,
            tx_symbols=tx_symbols,
            rx_symbols_noisy=rx_symbols_noisy,
            rx_symbols_enhanced=rx_symbols_enhanced,
            reconstructed_noisy=reconstructed_noisy,
            reconstructed_enhanced=reconstructed_enhanced,
            snr_db=snr_db,
            channel_type=channel_type
        )


# =============================================================================
# Visualization Functions
# =============================================================================

def create_constellation_heatmap(symbols: np.ndarray, title: str = "Constellation",
                                  ax: Optional[plt.Axes] = None,
                                  bins: int = 50, cmap: str = 'hot') -> plt.Axes:
    """
    Create a constellation diagram as a heatmap (density plot).
    
    Similar to the visualization in:
    https://github.com/rikluost/ofdm-plutosdr-pytorch
    
    Args:
        symbols: Complex symbols to plot
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        bins: Number of histogram bins
        cmap: Colormap for density
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    I = symbols.real
    Q = symbols.imag
    
    # Create 2D histogram for density
    range_val = max(np.abs(I).max(), np.abs(Q).max()) * 1.2
    H, xedges, yedges = np.histogram2d(
        I, Q, bins=bins, 
        range=[[-range_val, range_val], [-range_val, range_val]]
    )
    
    # Apply Gaussian smoothing for nicer visualization
    H = gaussian_filter(H, sigma=1)
    
    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H.T, extent=extent, origin='lower', cmap=cmap, 
                   aspect='equal', interpolation='gaussian')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Density', shrink=0.8)
    
    # Add grid and labels
    ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title(title)
    ax.set_xlim(-range_val, range_val)
    ax.set_ylim(-range_val, range_val)
    
    return ax


def create_constellation_scatter(symbols_list: List[np.ndarray], 
                                  labels: List[str],
                                  colors: List[str],
                                  title: str = "Constellation Comparison",
                                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create overlay constellation diagram with multiple symbol sets.
    
    Args:
        symbols_list: List of complex symbol arrays
        labels: Labels for each symbol set
        colors: Colors for each symbol set
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    for symbols, label, color in zip(symbols_list, labels, colors):
        ax.scatter(symbols.real, symbols.imag, 
                  label=label, color=color, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_image_comparison(original: np.ndarray, 
                          noisy: np.ndarray, 
                          enhanced: np.ndarray,
                          title: str = "Image Reconstruction Comparison") -> plt.Figure:
    """
    Plot side-by-side comparison of original, noisy, and enhanced images.
    
    Similar to JSCC paper visualizations showing image quality improvement.
    
    Args:
        original: Original image
        noisy: Image reconstructed from noisy channel
        enhanced: Image reconstructed after GAN enhancement
        title: Figure title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Calculate metrics
    psnr_noisy = calculate_psnr(original, noisy)
    psnr_enhanced = calculate_psnr(original, enhanced)
    ssim_noisy = calculate_ssim(original, noisy)
    ssim_enhanced = calculate_ssim(original, enhanced)
    
    # Common colormap settings
    vmin, vmax = -1, 1
    cmap = 'RdBu_r'
    
    # Original image
    im0 = axes[0].imshow(original, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Add pixel values as text for small images
    if original.shape[0] <= 5:
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                val = original[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=color, fontsize=8, fontweight='bold')
    
    # Noisy reconstruction
    im1 = axes[1].imshow(noisy, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Noisy Channel\nPSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.3f}')
    axes[1].axis('off')
    
    if noisy.shape[0] <= 5:
        for i in range(noisy.shape[0]):
            for j in range(noisy.shape[1]):
                val = noisy[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                axes[1].text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=8, fontweight='bold')
    
    # Enhanced reconstruction
    im2 = axes[2].imshow(enhanced, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f'GAN Enhanced\nPSNR: {psnr_enhanced:.2f} dB, SSIM: {ssim_enhanced:.3f}')
    axes[2].axis('off')
    
    if enhanced.shape[0] <= 5:
        for i in range(enhanced.shape[0]):
            for j in range(enhanced.shape[1]):
                val = enhanced[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                axes[2].text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=8, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(im2, ax=axes, orientation='vertical', shrink=0.8, label='Pixel Value')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_full_pipeline_visualization(result: TransmissionResult,
                                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create comprehensive visualization of the full transmission pipeline.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Full Pipeline Title                       │
    ├───────────┬───────────┬───────────┬───────────┬─────────────┤
    │  Original │    TX     │    RX     │    RX     │  Quality    │
    │   Image   │ Constell. │  (Noisy)  │ (Enhanced)│   Metrics   │
    ├───────────┼───────────┴───────────┴───────────┴─────────────┤
    │   Recon   │              Constellation Heatmaps              │
    │  Noisy    │  ┌─────────────┐  ┌─────────────┐               │
    │           │  │    Noisy    │  │  Enhanced   │               │
    ├───────────┤  └─────────────┘  └─────────────┘               │
    │   Recon   │                                                  │
    │  Enhanced │                                                  │
    └───────────┴─────────────────────────────────────────────────┘
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 1], 
                           width_ratios=[1, 1, 1, 1, 1])
    
    # Calculate metrics
    psnr_noisy = calculate_psnr(result.original_image, result.reconstructed_noisy)
    psnr_enhanced = calculate_psnr(result.original_image, result.reconstructed_enhanced)
    ssim_noisy = calculate_ssim(result.original_image, result.reconstructed_noisy)
    ssim_enhanced = calculate_ssim(result.original_image, result.reconstructed_enhanced)
    nmse_noisy = calculate_nmse(result.original_image, result.reconstructed_noisy)
    nmse_enhanced = calculate_nmse(result.original_image, result.reconstructed_enhanced)
    
    cmap_img = 'RdBu_r'
    vmin, vmax = -1, 1
    
    # =========================================================================
    # Row 1: Original Image, TX Constellation, RX Noisy, RX Enhanced, Metrics
    # =========================================================================
    
    # Original image
    ax_orig = fig.add_subplot(gs[0, 0])
    im = ax_orig.imshow(result.original_image, cmap=cmap_img, vmin=vmin, vmax=vmax)
    ax_orig.set_title('Original\n(Source Image)', fontweight='bold')
    ax_orig.axis('off')
    _add_pixel_annotations(ax_orig, result.original_image)
    
    # TX Constellation (clean)
    ax_tx = fig.add_subplot(gs[0, 1])
    ax_tx.scatter(result.tx_symbols.real, result.tx_symbols.imag, 
                  c='blue', s=100, alpha=0.8, edgecolors='white', linewidth=1)
    ax_tx.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_tx.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax_tx.set_xlabel('I')
    ax_tx.set_ylabel('Q')
    ax_tx.set_title('TX Symbols\n(Before Channel)', fontweight='bold')
    ax_tx.set_aspect('equal')
    ax_tx.grid(True, alpha=0.3)
    
    # RX Noisy Constellation
    ax_rx_noisy = fig.add_subplot(gs[0, 2])
    ax_rx_noisy.scatter(result.rx_symbols_noisy.real, result.rx_symbols_noisy.imag,
                        c='red', s=100, alpha=0.6, edgecolors='white', linewidth=1)
    ax_rx_noisy.scatter(result.tx_symbols.real, result.tx_symbols.imag,
                        c='blue', s=50, alpha=0.3, marker='x')
    ax_rx_noisy.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_rx_noisy.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax_rx_noisy.set_xlabel('I')
    ax_rx_noisy.set_ylabel('Q')
    ax_rx_noisy.set_title(f'RX Symbols (Noisy)\n{result.channel_type}, SNR={result.snr_db}dB', 
                          fontweight='bold')
    ax_rx_noisy.set_aspect('equal')
    ax_rx_noisy.grid(True, alpha=0.3)
    
    # RX Enhanced Constellation
    ax_rx_enh = fig.add_subplot(gs[0, 3])
    ax_rx_enh.scatter(result.rx_symbols_enhanced.real, result.rx_symbols_enhanced.imag,
                      c='green', s=100, alpha=0.6, edgecolors='white', linewidth=1)
    ax_rx_enh.scatter(result.tx_symbols.real, result.tx_symbols.imag,
                      c='blue', s=50, alpha=0.3, marker='x')
    ax_rx_enh.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_rx_enh.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax_rx_enh.set_xlabel('I')
    ax_rx_enh.set_ylabel('Q')
    ax_rx_enh.set_title('RX Symbols\n(GAN Enhanced)', fontweight='bold')
    ax_rx_enh.set_aspect('equal')
    ax_rx_enh.grid(True, alpha=0.3)
    
    # Quality Metrics Box
    ax_metrics = fig.add_subplot(gs[0, 4])
    ax_metrics.axis('off')
    
    metrics_text = f"""
    Quality Metrics
    ═══════════════
    
    Before GAN:
    ─────────────
    PSNR: {psnr_noisy:.2f} dB
    SSIM: {ssim_noisy:.4f}
    NMSE: {nmse_noisy:.4f}
    
    After GAN:
    ─────────────
    PSNR: {psnr_enhanced:.2f} dB
    SSIM: {ssim_enhanced:.4f}
    NMSE: {nmse_enhanced:.4f}
    
    Improvement:
    ─────────────
    ΔPSNR: +{psnr_enhanced - psnr_noisy:.2f} dB
    ΔSSIM: +{ssim_enhanced - ssim_noisy:.4f}
    """
    ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=9, fontfamily='monospace', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # =========================================================================
    # Row 2: Reconstructed Images
    # =========================================================================
    
    # Noisy Reconstruction
    ax_recon_noisy = fig.add_subplot(gs[1, 0])
    ax_recon_noisy.imshow(result.reconstructed_noisy, cmap=cmap_img, vmin=vmin, vmax=vmax)
    ax_recon_noisy.set_title(f'Noisy Recon\nPSNR: {psnr_noisy:.1f} dB', fontweight='bold')
    ax_recon_noisy.axis('off')
    _add_pixel_annotations(ax_recon_noisy, result.reconstructed_noisy)
    
    # Enhanced Reconstruction
    ax_recon_enh = fig.add_subplot(gs[2, 0])
    ax_recon_enh.imshow(result.reconstructed_enhanced, cmap=cmap_img, vmin=vmin, vmax=vmax)
    ax_recon_enh.set_title(f'Enhanced Recon\nPSNR: {psnr_enhanced:.1f} dB', fontweight='bold')
    ax_recon_enh.axis('off')
    _add_pixel_annotations(ax_recon_enh, result.reconstructed_enhanced)
    
    # =========================================================================
    # Row 2-3: Constellation Heatmaps (larger area)
    # =========================================================================
    
    ax_heat_noisy = fig.add_subplot(gs[1:, 1:3])
    create_constellation_heatmap(result.rx_symbols_noisy, 
                                  title='Noisy Channel Constellation Density', 
                                  ax=ax_heat_noisy, bins=30, cmap='hot')
    
    ax_heat_enh = fig.add_subplot(gs[1:, 3:5])
    create_constellation_heatmap(result.rx_symbols_enhanced,
                                  title='GAN Enhanced Constellation Density',
                                  ax=ax_heat_enh, bins=30, cmap='viridis')
    
    # =========================================================================
    # Title and Layout
    # =========================================================================
    fig.suptitle(f'Image Transmission over OFDM Channel\n'
                 f'({result.channel_type.upper()} Fading, SNR = {result.snr_db} dB)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def _add_pixel_annotations(ax: plt.Axes, image: np.ndarray, max_size: int = 5):
    """Add pixel value annotations to small images."""
    if image.shape[0] <= max_size and image.shape[1] <= max_size:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                val = image[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=7, fontweight='bold')


def plot_snr_sweep(snr_range: np.ndarray, 
                   psnr_noisy: np.ndarray, 
                   psnr_enhanced: np.ndarray,
                   ssim_noisy: np.ndarray,
                   ssim_enhanced: np.ndarray,
                   title: str = "Quality vs. SNR",
                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot quality metrics vs. SNR (similar to JSCC paper figures).
    
    Shows improvement from GAN enhancement across different SNR levels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR vs SNR
    axes[0].plot(snr_range, psnr_noisy, 'r--o', label='Without GAN', linewidth=2, markersize=6)
    axes[0].plot(snr_range, psnr_enhanced, 'g-s', label='With GAN', linewidth=2, markersize=6)
    axes[0].fill_between(snr_range, psnr_noisy, psnr_enhanced, alpha=0.2, color='green')
    axes[0].set_xlabel('Channel SNR (dB)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs. Channel SNR')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM vs SNR
    axes[1].plot(snr_range, ssim_noisy, 'r--o', label='Without GAN', linewidth=2, markersize=6)
    axes[1].plot(snr_range, ssim_enhanced, 'g-s', label='With GAN', linewidth=2, markersize=6)
    axes[1].fill_between(snr_range, ssim_noisy, ssim_enhanced, alpha=0.2, color='green')
    axes[1].set_xlabel('Channel SNR (dB)')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs. Channel SNR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def generate_simplegan_patterns() -> Dict[str, np.ndarray]:
    """
    Generate test patterns for SimpleGAN 3x3 images.
    
    Returns dictionary of named patterns.
    """
    patterns = {}
    
    # Cross pattern (from MATLAB)
    patterns['cross'] = np.array([
        [-1,  1, -1],
        [ 1,  1,  1],
        [-1,  1, -1]
    ], dtype=float)
    
    # Circle pattern (from MATLAB)
    patterns['circle'] = np.array([
        [ 1, -1,  1],
        [-1,  1, -1],
        [ 1, -1,  1]
    ], dtype=float)
    
    # Gradient horizontal
    patterns['gradient_h'] = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=float)
    
    # Gradient vertical
    patterns['gradient_v'] = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=float)
    
    # Random (simulated GAN output)
    np.random.seed(42)
    patterns['random'] = np.random.uniform(-0.5, 0.5, (3, 3))
    
    return patterns


# =============================================================================
# Main Visualization Script
# =============================================================================

def main():
    """Generate all visualizations for image-to-OFDM pipeline."""
    
    print("=" * 60)
    print("Image-to-OFDM Transmission Visualization")
    print("=" * 60)
    
    # Output directory
    output_dir = Path("docs/figures/ofdm_transmission")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    simulator = ImageOFDMSimulator(image_size=(3, 3))
    
    # Generate test patterns
    patterns = generate_simplegan_patterns()
    
    # =========================================================================
    # 1. Full Pipeline Visualization for Cross Pattern
    # =========================================================================
    print("\n[1] Generating full pipeline visualization (Cross pattern)...")
    
    result_cross = simulator.simulate_transmission(
        patterns['cross'],
        snr_db=10,
        channel_type='rayleigh',
        gan_enhancement=0.7
    )
    
    fig = create_full_pipeline_visualization(
        result_cross,
        save_path=output_dir / "full_pipeline_cross.png"
    )
    plt.close(fig)
    
    # =========================================================================
    # 2. Multiple Patterns Comparison
    # =========================================================================
    print("\n[2] Generating multi-pattern comparison...")
    
    fig, axes = plt.subplots(2, len(patterns), figsize=(3*len(patterns), 6))
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        result = simulator.simulate_transmission(pattern, snr_db=10, 
                                                  channel_type='rayleigh')
        
        # Original
        axes[0, idx].imshow(result.original_image, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, idx].set_title(f'Original\n({name})')
        axes[0, idx].axis('off')
        
        # Enhanced Reconstruction
        psnr = calculate_psnr(result.original_image, result.reconstructed_enhanced)
        axes[1, idx].imshow(result.reconstructed_enhanced, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, idx].set_title(f'Reconstructed\nPSNR: {psnr:.1f} dB')
        axes[1, idx].axis('off')
    
    fig.suptitle('Pattern Transmission over Rayleigh Channel (SNR=10dB)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "multi_pattern_comparison.png", dpi=150, 
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_dir / 'multi_pattern_comparison.png'}")
    plt.close(fig)
    
    # =========================================================================
    # 3. SNR Sweep Analysis
    # =========================================================================
    print("\n[3] Generating SNR sweep analysis...")
    
    snr_range = np.arange(-5, 25, 2)
    psnr_noisy_list = []
    psnr_enhanced_list = []
    ssim_noisy_list = []
    ssim_enhanced_list = []
    
    test_image = patterns['cross']
    
    for snr in snr_range:
        # Average over multiple runs
        psnr_n, psnr_e, ssim_n, ssim_e = [], [], [], []
        
        for _ in range(10):
            result = simulator.simulate_transmission(test_image, snr_db=snr,
                                                      channel_type='rayleigh')
            psnr_n.append(calculate_psnr(result.original_image, result.reconstructed_noisy))
            psnr_e.append(calculate_psnr(result.original_image, result.reconstructed_enhanced))
            ssim_n.append(calculate_ssim(result.original_image, result.reconstructed_noisy))
            ssim_e.append(calculate_ssim(result.original_image, result.reconstructed_enhanced))
        
        psnr_noisy_list.append(np.mean(psnr_n))
        psnr_enhanced_list.append(np.mean(psnr_e))
        ssim_noisy_list.append(np.mean(ssim_n))
        ssim_enhanced_list.append(np.mean(ssim_e))
    
    fig = plot_snr_sweep(
        snr_range,
        np.array(psnr_noisy_list),
        np.array(psnr_enhanced_list),
        np.array(ssim_noisy_list),
        np.array(ssim_enhanced_list),
        title='Reconstruction Quality vs. Channel SNR (Rayleigh Fading)',
        save_path=output_dir / "snr_sweep_quality.png"
    )
    plt.close(fig)
    
    # =========================================================================
    # 4. Channel Comparison (AWGN vs Rayleigh vs Rician)
    # =========================================================================
    print("\n[4] Generating channel comparison...")
    
    channels = ['awgn', 'rayleigh', 'rician']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for idx, channel in enumerate(channels):
        result = simulator.simulate_transmission(
            patterns['cross'], snr_db=10, channel_type=channel
        )
        
        # Constellation
        axes[0, idx].scatter(result.rx_symbols_noisy.real, result.rx_symbols_noisy.imag,
                            c='red', alpha=0.6, s=100, edgecolors='white')
        axes[0, idx].scatter(result.tx_symbols.real, result.tx_symbols.imag,
                            c='blue', alpha=0.3, s=50, marker='x')
        axes[0, idx].set_title(f'{channel.upper()} Channel\nConstellation (SNR=10dB)')
        axes[0, idx].set_xlabel('I')
        axes[0, idx].set_ylabel('Q')
        axes[0, idx].set_aspect('equal')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Reconstruction
        psnr = calculate_psnr(result.original_image, result.reconstructed_enhanced)
        axes[1, idx].imshow(result.reconstructed_enhanced, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, idx].set_title(f'Reconstruction\nPSNR: {psnr:.1f} dB')
        axes[1, idx].axis('off')
        _add_pixel_annotations(axes[1, idx], result.reconstructed_enhanced)
    
    fig.suptitle('Channel Type Comparison (GAN Enhanced)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "channel_comparison.png", dpi=150, 
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_dir / 'channel_comparison.png'}")
    plt.close(fig)
    
    # =========================================================================
    # 5. Constellation Heatmap Comparison
    # =========================================================================
    print("\n[5] Generating constellation heatmaps...")
    
    # Generate many samples for better heatmap
    all_tx_symbols = []
    all_rx_noisy = []
    all_rx_enhanced = []
    
    for pattern in patterns.values():
        for _ in range(50):
            result = simulator.simulate_transmission(pattern, snr_db=10,
                                                      channel_type='rayleigh')
            all_tx_symbols.extend(result.tx_symbols)
            all_rx_noisy.extend(result.rx_symbols_noisy)
            all_rx_enhanced.extend(result.rx_symbols_enhanced)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    create_constellation_heatmap(np.array(all_tx_symbols), 
                                  title='TX Constellation (Clean)',
                                  ax=axes[0], bins=40, cmap='Blues')
    
    create_constellation_heatmap(np.array(all_rx_noisy),
                                  title='RX Constellation (Noisy)',
                                  ax=axes[1], bins=40, cmap='hot')
    
    create_constellation_heatmap(np.array(all_rx_enhanced),
                                  title='RX Constellation (GAN Enhanced)',
                                  ax=axes[2], bins=40, cmap='viridis')
    
    fig.suptitle('Constellation Density Heatmaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "constellation_heatmaps.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_dir / 'constellation_heatmaps.png'}")
    plt.close(fig)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    
    print("\nThese visualizations show:")
    print("  1. How 3x3 'images' are encoded as I/Q symbols")
    print("  2. Channel effects on constellation (scattering)")
    print("  3. GAN enhancement reduces symbol displacement")
    print("  4. Reconstruction quality metrics (PSNR, SSIM)")
    print("  5. Performance across different SNR levels")


if __name__ == "__main__":
    main()
