# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# OFDM Utilities: Signal Generation, Modulation, Channel Simulation
# =============================================================================
"""
OFDM MATHEMATICAL FOUNDATION
============================

ORTHOGONAL FREQUENCY DIVISION MULTIPLEXING (OFDM):
---------------------------------------------------
OFDM divides a high-rate data stream into multiple lower-rate subcarriers.
Each subcarrier is modulated independently using QAM/PSK.

OFDM SIGNAL GENERATION:
-----------------------
1. Serial-to-Parallel conversion of input bits
2. Symbol mapping (QAM modulation)
3. IFFT to create time-domain signal
4. Cyclic prefix insertion

Time-domain OFDM symbol:
    x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^{j·2π·k·n/N}

Where:
- N = number of subcarriers
- X[k] = QAM symbol on subcarrier k
- x[n] = time-domain sample n

CYCLIC PREFIX:
--------------
To combat inter-symbol interference (ISI), we prepend the last L_cp samples:
    x_cp = [x[N-L_cp:N], x[0:N]]

QAM MODULATION:
---------------
Maps bits to complex symbols on a 2D constellation.

QPSK (2 bits/symbol):
    Constellation points: (±1 ± j)/√2

16-QAM (4 bits/symbol):
    Constellation points: (±1, ±3) × (±1, ±3) / √10

64-QAM (6 bits/symbol):
    Constellation points: (±1, ±3, ±5, ±7) × same / √42

CHANNEL MODELS:
---------------

1. AWGN (Additive White Gaussian Noise):
    y = x + n, where n ~ CN(0, σ²)
    
    SNR (dB) = 10·log₁₀(P_signal / P_noise)
    σ² = P_signal / 10^(SNR/10)

2. Rayleigh Fading:
    y = h·x + n, where h ~ CN(0, 1)
    
    Models non-line-of-sight propagation with many scattered paths.

3. Rician Fading:
    h = √(K/(K+1))·h_LOS + √(1/(K+1))·h_NLOS
    
    Where K is the Rician K-factor (ratio of LOS to scattered power).

4. Multipath Channel:
    y[n] = Σ_{l=0}^{L-1} h[l]·x[n-l] + n[n]
    
    Convolution with channel impulse response.

I/Q REPRESENTATION:
-------------------
Complex signal x = I + jQ is represented as 2-channel real tensor:
    x_tensor = [I, Q] ∈ ℝ^(2×L)

This is the input/output format for the neural network.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from scipy import signal
from PIL import Image


# =============================================================================
# QAM Modulation
# =============================================================================

class QAMModulator:
    """
    QAM Modulation/Demodulation with support for QPSK, 16-QAM, 64-QAM.
    
    Mathematical formulation:
    -------------------------
    For M-QAM with M = 2^b constellation points:
    
    Symbol mapping: bits → complex symbol
        s = (2i - √M + 1) + j(2q - √M + 1), normalized by √(2(M-1)/3)
    
    Where i, q are the in-phase and quadrature indices derived from bits.
    """
    
    # Constellation definitions with normalization factors
    CONSTELLATIONS = {
        'QPSK': {
            'bits_per_symbol': 2,
            'points': np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2),
        },
        'QAM16': {
            'bits_per_symbol': 4,
            'points': None,  # Generated dynamically
            'norm_factor': np.sqrt(10),  # √(2×(M-1)/3) for M=16
        },
        'QAM64': {
            'bits_per_symbol': 6,
            'points': None,  # Generated dynamically
            'norm_factor': np.sqrt(42),  # √(2×(M-1)/3) for M=64
        }
    }
    
    def __init__(self, modulation: str = 'QAM16'):
        """
        Initialize QAM modulator.
        
        Args:
            modulation: Modulation scheme ('QPSK', 'QAM16', 'QAM64')
        """
        self.modulation = modulation.upper()
        if self.modulation not in self.CONSTELLATIONS:
            raise ValueError(f"Unsupported modulation: {modulation}")
            
        self.config = self.CONSTELLATIONS[self.modulation]
        self.bits_per_symbol = self.config['bits_per_symbol']
        self.constellation = self._generate_constellation()
        
    def _generate_constellation(self) -> np.ndarray:
        """
        Generate constellation points.
        
        For M-QAM:
            Real part: ±1, ±3, ... (gray coded)
            Imag part: ±1, ±3, ... (gray coded)
        """
        if self.modulation == 'QPSK':
            return self.config['points']
            
        M = 2 ** self.bits_per_symbol
        sqrt_M = int(np.sqrt(M))
        
        # Generate PAM levels
        levels = np.arange(-sqrt_M + 1, sqrt_M, 2)
        
        # Create 2D grid
        I, Q = np.meshgrid(levels, levels)
        constellation = (I + 1j * Q).flatten()
        
        # Normalize to unit average power
        constellation = constellation / self.config['norm_factor']
        
        return constellation
        
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Map bits to QAM symbols.
        
        Args:
            bits: Binary array, length must be multiple of bits_per_symbol
            
        Returns:
            Complex symbol array
            
        Mathematical operation:
            For each group of b bits, map to constellation point:
            symbol_index = binary_to_int(bits[i:i+b])
            symbol = constellation[symbol_index]
        """
        # Ensure bits are binary
        bits = bits.astype(int)
        
        # Reshape into groups of bits_per_symbol
        n_symbols = len(bits) // self.bits_per_symbol
        bits = bits[:n_symbols * self.bits_per_symbol]
        bits = bits.reshape(n_symbols, self.bits_per_symbol)
        
        # Convert bit groups to indices (MSB first)
        powers = 2 ** np.arange(self.bits_per_symbol - 1, -1, -1)
        indices = np.sum(bits * powers, axis=1)
        
        # Map to constellation
        symbols = self.constellation[indices]
        
        return symbols
        
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard decision demodulation: map symbols to nearest constellation point.
        
        Args:
            symbols: Complex symbol array
            
        Returns:
            Demodulated bits
            
        Mathematical operation:
            For each symbol s:
            index = argmin_k |s - constellation[k]|²
            bits = int_to_binary(index)
        """
        # Calculate distance to all constellation points
        # Shape: (n_symbols, n_constellation)
        distances = np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :]) ** 2
        
        # Find closest constellation point
        indices = np.argmin(distances, axis=1)
        
        # Convert indices to bits
        bits = np.zeros((len(indices), self.bits_per_symbol), dtype=int)
        for i in range(self.bits_per_symbol):
            bits[:, self.bits_per_symbol - 1 - i] = (indices >> i) & 1
            
        return bits.flatten()


# =============================================================================
# OFDM Modulation
# =============================================================================

class OFDMModulator:
    """
    OFDM modulation/demodulation.
    
    Implements:
    - IFFT/FFT for time-frequency conversion
    - Cyclic prefix insertion/removal
    - Pilot insertion for channel estimation
    
    Mathematical operations:
    ------------------------
    OFDM symbol generation:
        1. Map data to subcarriers: X[k] = QAM_symbol or pilot
        2. IFFT: x[n] = (1/N) Σ_{k=0}^{N-1} X[k]·exp(j2πkn/N)
        3. Add CP: x_cp = [x[N-Ncp:N], x[0:N]]
    
    OFDM demodulation:
        1. Remove CP
        2. FFT: X[k] = Σ_{n=0}^{N-1} x[n]·exp(-j2πkn/N)
        3. Extract data symbols
    """
    
    def __init__(
        self,
        n_subcarriers: int = 64,
        cp_length: int = 16,
        pilot_spacing: int = 8,
        pilot_value: complex = 1 + 0j
    ):
        """
        Initialize OFDM modulator.
        
        Args:
            n_subcarriers: Number of OFDM subcarriers (FFT size)
            cp_length: Cyclic prefix length
            pilot_spacing: Spacing between pilot subcarriers
            pilot_value: Complex value for pilot symbols
        """
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.pilot_spacing = pilot_spacing
        self.pilot_value = pilot_value
        
        # Calculate indices
        self.pilot_indices = np.arange(0, n_subcarriers, pilot_spacing)
        self.data_indices = np.array([i for i in range(n_subcarriers) 
                                      if i not in self.pilot_indices])
        self.n_data_subcarriers = len(self.data_indices)
        
        # Samples per OFDM symbol (with CP)
        self.samples_per_symbol = n_subcarriers + cp_length
        
    def modulate(self, qam_symbols: np.ndarray) -> np.ndarray:
        """
        Generate OFDM time-domain signal from QAM symbols.
        
        Args:
            qam_symbols: Complex QAM symbols to modulate
            
        Returns:
            Complex time-domain OFDM signal
            
        Mathematical steps:
            1. Allocate data symbols to data subcarriers
            2. Insert pilots at pilot indices
            3. Perform IFFT
            4. Add cyclic prefix
        """
        # Calculate number of OFDM symbols needed
        n_data_per_symbol = self.n_data_subcarriers
        n_ofdm_symbols = int(np.ceil(len(qam_symbols) / n_data_per_symbol))
        
        # Pad QAM symbols if needed
        total_data = n_ofdm_symbols * n_data_per_symbol
        qam_padded = np.zeros(total_data, dtype=complex)
        qam_padded[:len(qam_symbols)] = qam_symbols
        
        # Reshape into OFDM symbols
        qam_reshaped = qam_padded.reshape(n_ofdm_symbols, n_data_per_symbol)
        
        # Create frequency-domain OFDM symbols
        freq_symbols = np.zeros((n_ofdm_symbols, self.n_subcarriers), dtype=complex)
        
        # Insert data
        freq_symbols[:, self.data_indices] = qam_reshaped
        
        # Insert pilots
        freq_symbols[:, self.pilot_indices] = self.pilot_value
        
        # IFFT to get time-domain signal
        # x[n] = (1/N) Σ_k X[k] exp(j2πkn/N)
        time_symbols = np.fft.ifft(freq_symbols, axis=1) * self.n_subcarriers
        
        # Add cyclic prefix
        cp = time_symbols[:, -self.cp_length:]
        time_with_cp = np.concatenate([cp, time_symbols], axis=1)
        
        # Flatten to 1D signal
        ofdm_signal = time_with_cp.flatten()
        
        return ofdm_signal
        
    def demodulate(self, ofdm_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demodulate OFDM signal to QAM symbols.
        
        Args:
            ofdm_signal: Complex time-domain OFDM signal
            
        Returns:
            Tuple of (data_symbols, channel_estimates)
            
        Mathematical steps:
            1. Reshape into OFDM symbols
            2. Remove cyclic prefix
            3. Perform FFT
            4. Extract data symbols and pilots
        """
        # Calculate number of OFDM symbols
        n_ofdm_symbols = len(ofdm_signal) // self.samples_per_symbol
        
        # Reshape into OFDM symbols with CP
        ofdm_reshaped = ofdm_signal[:n_ofdm_symbols * self.samples_per_symbol]
        ofdm_reshaped = ofdm_reshaped.reshape(n_ofdm_symbols, self.samples_per_symbol)
        
        # Remove cyclic prefix
        time_symbols = ofdm_reshaped[:, self.cp_length:]
        
        # FFT to get frequency-domain symbols
        # X[k] = Σ_n x[n] exp(-j2πkn/N)
        freq_symbols = np.fft.fft(time_symbols, axis=1) / self.n_subcarriers
        
        # Extract data symbols
        data_symbols = freq_symbols[:, self.data_indices].flatten()
        
        # Extract pilot symbols for channel estimation
        pilot_symbols = freq_symbols[:, self.pilot_indices]
        
        # Simple channel estimate from pilots
        # H = received_pilot / transmitted_pilot
        channel_estimates = pilot_symbols / self.pilot_value
        
        return data_symbols, channel_estimates


# =============================================================================
# Channel Models
# =============================================================================

class ChannelModel:
    """
    Wireless channel models for OFDM simulation.
    
    Supports:
    - AWGN (Additive White Gaussian Noise)
    - Rayleigh fading
    - Rician fading
    - Multipath channel
    
    Mathematical models:
    --------------------
    AWGN: y = x + n, n ~ CN(0, σ²)
    Rayleigh: y = h·x + n, h ~ CN(0, 1)
    Rician: y = h·x + n, h = √(K/(K+1))·exp(jθ) + √(1/(K+1))·h_scatter
    Multipath: y = conv(x, h) + n
    """
    
    def __init__(self, channel_type: str = 'awgn'):
        """
        Initialize channel model.
        
        Args:
            channel_type: Type of channel ('awgn', 'rayleigh', 'rician', 'multipath')
        """
        self.channel_type = channel_type.lower()
        
    def apply(
        self,
        signal: np.ndarray,
        snr_db: float,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply channel effects to signal.
        
        Args:
            signal: Complex input signal
            snr_db: Signal-to-noise ratio in dB
            **kwargs: Additional channel parameters
            
        Returns:
            Tuple of (received_signal, channel_info)
            
        SNR calculation:
            P_signal = E[|x|²]
            P_noise = P_signal / 10^(SNR_dB/10)
            σ = √(P_noise / 2)  (per dimension for complex noise)
        """
        if self.channel_type == 'awgn':
            return self._apply_awgn(signal, snr_db)
        elif self.channel_type == 'rayleigh':
            return self._apply_rayleigh(signal, snr_db)
        elif self.channel_type == 'rician':
            k_factor = kwargs.get('k_factor', 3.0)
            return self._apply_rician(signal, snr_db, k_factor)
        elif self.channel_type == 'multipath':
            delays = kwargs.get('delays', [0, 1, 2])
            powers = kwargs.get('powers', [1.0, 0.5, 0.25])
            return self._apply_multipath(signal, snr_db, delays, powers)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")
            
    def _apply_awgn(
        self, 
        signal: np.ndarray, 
        snr_db: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply AWGN channel.
        
        y = x + n, where n ~ CN(0, σ²)
        
        σ² = P_signal / 10^(SNR_dB/10)
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)
        
        # Calculate noise power from SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate complex Gaussian noise
        # Variance per dimension = noise_power / 2
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*signal.shape) + 
                            1j * np.random.randn(*signal.shape))
        
        received = signal + noise
        
        channel_info = {
            'type': 'awgn',
            'snr_db': snr_db,
            'noise_power': noise_power,
            'channel_response': np.array([1.0])  # Identity channel
        }
        
        return received, channel_info
        
    def _apply_rayleigh(
        self, 
        signal: np.ndarray, 
        snr_db: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Rayleigh fading channel.
        
        y = h·x + n, where h ~ CN(0, 1)
        
        Models non-line-of-sight propagation with many scattered paths.
        |h| follows Rayleigh distribution.
        """
        # Generate Rayleigh fading coefficient
        h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        
        # Apply fading
        faded = h * signal
        
        # Add AWGN
        received, awgn_info = self._apply_awgn(faded, snr_db)
        
        channel_info = {
            'type': 'rayleigh',
            'snr_db': snr_db,
            'channel_response': h,
            'channel_magnitude': np.abs(h),
            'noise_power': awgn_info['noise_power']
        }
        
        return received, channel_info
        
    def _apply_rician(
        self,
        signal: np.ndarray,
        snr_db: float,
        k_factor: float = 3.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Rician fading channel.
        
        h = √(K/(K+1))·exp(jθ) + √(1/(K+1))·h_nlos
        
        Where:
        - K = Rician K-factor (LOS power / scattered power)
        - θ = random LOS phase
        - h_nlos ~ CN(0, 1) is the scattered component
        
        K → ∞: Pure LOS (no fading)
        K → 0: Rayleigh fading
        """
        # LOS component with random phase
        theta = np.random.uniform(0, 2 * np.pi)
        h_los = np.sqrt(k_factor / (k_factor + 1)) * np.exp(1j * theta)
        
        # Scattered (NLOS) component
        h_nlos = np.sqrt(1 / (k_factor + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        
        # Combined channel
        h = h_los + h_nlos
        
        # Apply fading
        faded = h * signal
        
        # Add AWGN
        received, awgn_info = self._apply_awgn(faded, snr_db)
        
        channel_info = {
            'type': 'rician',
            'snr_db': snr_db,
            'k_factor': k_factor,
            'channel_response': h,
            'channel_magnitude': np.abs(h),
            'noise_power': awgn_info['noise_power']
        }
        
        return received, channel_info
        
    def _apply_multipath(
        self,
        signal: np.ndarray,
        snr_db: float,
        delays: list = [0, 1, 2],
        powers: list = [1.0, 0.5, 0.25]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply multipath channel.
        
        y[n] = Σ_{l=0}^{L-1} h[l]·x[n-l] + n[n]
        
        Channel impulse response:
        h[l] = √(power[l]) · (randn + j·randn) / √2
        
        Each tap has independent Rayleigh fading.
        """
        # Normalize powers
        powers = np.array(powers)
        powers = powers / np.sum(powers)
        
        # Generate channel taps with Rayleigh fading
        max_delay = max(delays)
        h = np.zeros(max_delay + 1, dtype=complex)
        
        for delay, power in zip(delays, powers):
            h[delay] = np.sqrt(power) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            
        # Apply multipath channel (convolution)
        # y = conv(x, h)
        faded = np.convolve(signal, h, mode='same')
        
        # Add AWGN
        received, awgn_info = self._apply_awgn(faded, snr_db)
        
        channel_info = {
            'type': 'multipath',
            'snr_db': snr_db,
            'delays': delays,
            'powers': powers.tolist(),
            'channel_response': h,
            'noise_power': awgn_info['noise_power']
        }
        
        return received, channel_info


# =============================================================================
# Image to OFDM Conversion
# =============================================================================

class ImageOFDMConverter:
    """
    Convert images to OFDM signals and back.
    
    Pipeline:
    ---------
    Image → OFDM:
        1. Load image → pixel array
        2. Flatten and normalize to [0, 255]
        3. Convert to bits
        4. QAM modulation → complex symbols
        5. OFDM modulation → time-domain I/Q signal
        
    OFDM → Image:
        1. OFDM demodulation → complex symbols
        2. QAM demodulation → bits
        3. Convert bits to pixels
        4. Reshape to image dimensions
        
    Signal format for neural network:
        I/Q tensor of shape [2, L] where:
        - Channel 0: In-phase (I) component
        - Channel 1: Quadrature (Q) component
    """
    
    def __init__(
        self,
        modulation: str = 'QAM16',
        n_subcarriers: int = 64,
        cp_length: int = 16,
        frame_length: int = 1024
    ):
        """
        Initialize converter.
        
        Args:
            modulation: QAM modulation scheme
            n_subcarriers: OFDM subcarriers
            cp_length: Cyclic prefix length
            frame_length: Target OFDM frame length for neural network
        """
        self.qam = QAMModulator(modulation)
        self.ofdm = OFDMModulator(n_subcarriers, cp_length)
        self.frame_length = frame_length
        self.modulation = modulation
        
    def image_to_ofdm(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert image to OFDM signal.
        
        Args:
            image: Image array (grayscale or RGB)
            normalize: Whether to normalize output to [-1, 1]
            
        Returns:
            Tuple of (ofdm_iq_signal, metadata)
            
        Output shape: [2, L] where L is padded to frame_length
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            
        original_shape = image.shape
        
        # Flatten image to 1D pixel array
        pixels = image.flatten().astype(np.uint8)
        
        # Convert pixels to bits (8 bits per pixel)
        bits = np.unpackbits(pixels)
        
        # QAM modulation
        qam_symbols = self.qam.modulate(bits)
        
        # OFDM modulation
        ofdm_signal = self.ofdm.modulate(qam_symbols)
        
        # Pad or truncate to frame_length
        if len(ofdm_signal) < self.frame_length:
            padded = np.zeros(self.frame_length, dtype=complex)
            padded[:len(ofdm_signal)] = ofdm_signal
            ofdm_signal = padded
        else:
            ofdm_signal = ofdm_signal[:self.frame_length]
            
        # Convert to I/Q representation [2, L]
        I = np.real(ofdm_signal)
        Q = np.imag(ofdm_signal)
        iq_signal = np.stack([I, Q], axis=0)
        
        # Normalize to [-1, 1] if requested
        if normalize:
            max_val = np.max(np.abs(iq_signal))
            if max_val > 0:
                iq_signal = iq_signal / max_val
                
        metadata = {
            'original_shape': original_shape,
            'n_pixels': len(pixels),
            'n_bits': len(bits),
            'n_qam_symbols': len(qam_symbols),
            'signal_length': len(ofdm_signal),
            'normalization_factor': max_val if normalize else 1.0
        }
        
        return iq_signal.astype(np.float32), metadata
        
    def ofdm_to_image(
        self,
        iq_signal: np.ndarray,
        original_shape: Tuple[int, ...],
        denormalize_factor: float = 1.0
    ) -> np.ndarray:
        """
        Convert OFDM signal back to image.
        
        Args:
            iq_signal: I/Q signal of shape [2, L]
            original_shape: Original image dimensions
            denormalize_factor: Factor to undo normalization
            
        Returns:
            Reconstructed image array
        """
        # Denormalize if needed
        iq_signal = iq_signal * denormalize_factor
        
        # Convert I/Q to complex
        ofdm_signal = iq_signal[0] + 1j * iq_signal[1]
        
        # OFDM demodulation
        qam_symbols, _ = self.ofdm.demodulate(ofdm_signal)
        
        # QAM demodulation
        bits = self.qam.demodulate(qam_symbols)
        
        # Calculate required bits for image
        n_pixels = np.prod(original_shape)
        n_bits_needed = n_pixels * 8
        
        # Truncate or pad bits
        if len(bits) >= n_bits_needed:
            bits = bits[:n_bits_needed]
        else:
            # Pad with zeros
            bits = np.concatenate([bits, np.zeros(n_bits_needed - len(bits), dtype=int)])
            
        # Convert bits to pixels
        bits = bits.astype(np.uint8)
        pixels = np.packbits(bits)[:n_pixels]
        
        # Reshape to original image dimensions
        image = pixels.reshape(original_shape)
        
        return image
        
    def to_tensor(self, iq_signal: np.ndarray) -> torch.Tensor:
        """
        Convert I/Q signal to PyTorch tensor.
        
        Args:
            iq_signal: NumPy array of shape [2, L]
            
        Returns:
            PyTorch tensor of shape [1, 2, L]
        """
        return torch.from_numpy(iq_signal).unsqueeze(0).float()
        
    def from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to I/Q signal.
        
        Args:
            tensor: PyTorch tensor of shape [B, 2, L] or [2, L]
            
        Returns:
            NumPy array of shape [2, L]
        """
        if tensor.dim() == 3:
            tensor = tensor[0]
        return tensor.detach().cpu().numpy()


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OFDM Utilities Verification")
    print("=" * 60)
    
    # Test QAM Modulator
    print("\n--- QAM Modulation Test ---")
    qam = QAMModulator('QAM16')
    test_bits = np.random.randint(0, 2, 100)
    symbols = qam.modulate(test_bits)
    demod_bits = qam.demodulate(symbols)
    
    # Check bit error rate (should be 0 without noise)
    n_correct = len(test_bits) // qam.bits_per_symbol * qam.bits_per_symbol
    ber = np.sum(test_bits[:n_correct] != demod_bits[:n_correct]) / n_correct
    print(f"QAM16 BER (no noise): {ber:.6f}")
    
    # Test OFDM Modulator
    print("\n--- OFDM Modulation Test ---")
    ofdm = OFDMModulator(n_subcarriers=64, cp_length=16)
    test_symbols = np.random.randn(100) + 1j * np.random.randn(100)
    ofdm_signal = ofdm.modulate(test_symbols)
    print(f"OFDM signal length: {len(ofdm_signal)}")
    print(f"Samples per symbol: {ofdm.samples_per_symbol}")
    
    # Test Channel Model
    print("\n--- Channel Model Test ---")
    for ch_type in ['awgn', 'rayleigh', 'rician']:
        channel = ChannelModel(ch_type)
        rx, info = channel.apply(ofdm_signal, snr_db=20)
        print(f"{ch_type.upper()}: SNR={info['snr_db']}dB, "
              f"|h|={np.abs(info['channel_response']).mean():.4f}")
    
    # Test Image Converter
    print("\n--- Image to OFDM Test ---")
    converter = ImageOFDMConverter(modulation='QAM16', frame_length=1024)
    
    # Create test image (8x8 grayscale)
    test_image = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    
    iq_signal, metadata = converter.image_to_ofdm(test_image)
    print(f"Test image shape: {test_image.shape}")
    print(f"I/Q signal shape: {iq_signal.shape}")
    print(f"Metadata: {metadata}")
    
    # Reconstruct image
    reconstructed = converter.ofdm_to_image(
        iq_signal, 
        metadata['original_shape'],
        metadata['normalization_factor']
    )
    
    # Check reconstruction error
    mse = np.mean((test_image.astype(float) - reconstructed.astype(float)) ** 2)
    print(f"Reconstruction MSE (no channel): {mse:.4f}")
    
    # Convert to tensor
    tensor = converter.to_tensor(iq_signal)
    print(f"Tensor shape: {tensor.shape}")
    
    print("\n✓ OFDM utilities verification complete!")
