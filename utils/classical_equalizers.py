# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Classical Equalizers for Baseline Comparison
# =============================================================================
"""
CLASSICAL EQUALIZERS FOR OFDM
=============================

This module implements classical equalization methods to serve as baselines
for comparing against the CWGAN-GP approach. These methods are well-established
in the literature and provide clear benchmarks.

Implemented Methods:
--------------------
1. Zero-Forcing (ZF) Equalizer
2. Minimum Mean Square Error (MMSE) Equalizer  
3. Decision Feedback Equalizer (DFE)
4. Least Mean Squares (LMS) Adaptive Equalizer
5. Recursive Least Squares (RLS) Adaptive Equalizer

Key Insight for LSI Contest:
----------------------------
Classical equalizers struggle with NON-LINEAR impairments (PA compression,
IQ imbalance, phase noise) because they assume linear channel models.
This is where neural network approaches like CWGAN-GP excel.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.linalg import toeplitz, inv


class ZeroForcingEqualizer:
    """
    Zero-Forcing (ZF) Equalizer.
    
    Mathematical formulation:
    -------------------------
    Given channel H in frequency domain:
        Y = H * X + N
    
    ZF estimate:
        X_hat = Y / H = X + N/H
    
    Pros: Completely removes ISI when H is known
    Cons: Amplifies noise when |H| is small (noise enhancement)
    
    Complexity: O(N) per symbol
    """
    
    def __init__(self, n_subcarriers: int = 64):
        self.n_subcarriers = n_subcarriers
        self.channel_estimate = None
        
    def estimate_channel(
        self, 
        received_pilots: np.ndarray, 
        transmitted_pilots: np.ndarray
    ) -> np.ndarray:
        """
        Estimate channel from pilot symbols.
        
        H_est = Y_pilot / X_pilot
        """
        # Avoid division by zero
        eps = 1e-10
        self.channel_estimate = received_pilots / (transmitted_pilots + eps)
        return self.channel_estimate
    
    def equalize(
        self, 
        received_signal: np.ndarray, 
        channel_estimate: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply ZF equalization.
        
        X_hat = Y / H
        """
        if channel_estimate is not None:
            self.channel_estimate = channel_estimate
            
        if self.channel_estimate is None:
            raise ValueError("Channel estimate required. Call estimate_channel first.")
        
        eps = 1e-10
        equalized = received_signal / (self.channel_estimate + eps)
        return equalized
    
    def equalize_iq(
        self,
        noisy_iq: np.ndarray,
        clean_iq: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Equalize I/Q signal format [2, L].
        
        For fair comparison with neural network.
        """
        # Convert to complex
        noisy_complex = noisy_iq[0] + 1j * noisy_iq[1]
        
        # If clean signal provided, estimate channel
        if clean_iq is not None:
            clean_complex = clean_iq[0] + 1j * clean_iq[1]
            self.estimate_channel(noisy_complex, clean_complex)
        
        # Equalize
        equalized_complex = self.equalize(noisy_complex)
        
        # Convert back to I/Q
        equalized_iq = np.stack([
            np.real(equalized_complex),
            np.imag(equalized_complex)
        ], axis=0).astype(np.float32)
        
        # Compute metrics
        metrics = {}
        if clean_iq is not None:
            mse = np.mean((equalized_iq - clean_iq) ** 2)
            metrics['mse'] = float(mse)
            metrics['snr_improvement_db'] = float(10 * np.log10(
                np.mean(noisy_iq ** 2) / (mse + 1e-10)
            ))
        
        return equalized_iq, metrics


class MMSEEqualizer:
    """
    Minimum Mean Square Error (MMSE) Equalizer.
    
    Mathematical formulation:
    -------------------------
    MMSE estimate balances ISI removal with noise enhancement:
        X_hat = (H* / (|H|² + 1/SNR)) * Y
    
    This is equivalent to Wiener filter solution.
    
    Pros: Better noise handling than ZF
    Cons: Requires SNR estimate, still linear
    
    Complexity: O(N) per symbol
    """
    
    def __init__(self, n_subcarriers: int = 64):
        self.n_subcarriers = n_subcarriers
        self.channel_estimate = None
        self.noise_variance = None
        
    def estimate_channel(
        self, 
        received_pilots: np.ndarray, 
        transmitted_pilots: np.ndarray
    ) -> np.ndarray:
        """Estimate channel from pilots."""
        eps = 1e-10
        self.channel_estimate = received_pilots / (transmitted_pilots + eps)
        return self.channel_estimate
    
    def estimate_noise(
        self,
        received_signal: np.ndarray,
        channel_estimate: np.ndarray,
        transmitted_signal: np.ndarray
    ) -> float:
        """Estimate noise variance."""
        predicted = channel_estimate * transmitted_signal
        noise = received_signal - predicted
        self.noise_variance = np.var(noise)
        return self.noise_variance
    
    def equalize(
        self, 
        received_signal: np.ndarray, 
        channel_estimate: Optional[np.ndarray] = None,
        snr_db: float = 20.0
    ) -> np.ndarray:
        """
        Apply MMSE equalization.
        
        X_hat = conj(H) / (|H|² + 1/SNR) * Y
        """
        if channel_estimate is not None:
            self.channel_estimate = channel_estimate
            
        if self.channel_estimate is None:
            raise ValueError("Channel estimate required.")
        
        H = self.channel_estimate
        snr_linear = 10 ** (snr_db / 10)
        
        # MMSE filter
        H_conj = np.conj(H)
        H_power = np.abs(H) ** 2
        
        mmse_filter = H_conj / (H_power + 1.0 / snr_linear)
        equalized = mmse_filter * received_signal
        
        return equalized
    
    def equalize_iq(
        self,
        noisy_iq: np.ndarray,
        clean_iq: np.ndarray = None,
        snr_db: float = 20.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Equalize I/Q signal format [2, L]."""
        noisy_complex = noisy_iq[0] + 1j * noisy_iq[1]
        
        if clean_iq is not None:
            clean_complex = clean_iq[0] + 1j * clean_iq[1]
            self.estimate_channel(noisy_complex, clean_complex)
        
        equalized_complex = self.equalize(noisy_complex, snr_db=snr_db)
        
        equalized_iq = np.stack([
            np.real(equalized_complex),
            np.imag(equalized_complex)
        ], axis=0).astype(np.float32)
        
        metrics = {}
        if clean_iq is not None:
            mse = np.mean((equalized_iq - clean_iq) ** 2)
            metrics['mse'] = float(mse)
            metrics['snr_improvement_db'] = float(10 * np.log10(
                np.mean(noisy_iq ** 2) / (mse + 1e-10)
            ))
        
        return equalized_iq, metrics


class DecisionFeedbackEqualizer:
    """
    Decision Feedback Equalizer (DFE).
    
    Mathematical formulation:
    -------------------------
    DFE uses previously detected symbols to cancel post-cursor ISI:
        y[n] = sum(w_ff[k] * r[n-k]) - sum(w_fb[k] * x_hat[n-k])
    
    where:
        w_ff = feedforward filter coefficients
        w_fb = feedback filter coefficients
        r = received signal
        x_hat = previously decided symbols
    
    Pros: Better than linear equalizers for ISI channels
    Cons: Error propagation, still struggles with non-linear distortion
    
    Complexity: O(L_ff + L_fb) per symbol
    """
    
    def __init__(
        self, 
        n_ff_taps: int = 5,
        n_fb_taps: int = 3,
        mu: float = 0.01
    ):
        """
        Initialize DFE.
        
        Args:
            n_ff_taps: Number of feedforward taps
            n_fb_taps: Number of feedback taps
            mu: LMS step size for adaptation
        """
        self.n_ff_taps = n_ff_taps
        self.n_fb_taps = n_fb_taps
        self.mu = mu
        
        # Initialize filter coefficients
        self.w_ff = np.zeros(n_ff_taps, dtype=complex)
        self.w_ff[n_ff_taps // 2] = 1.0  # Center tap
        self.w_fb = np.zeros(n_fb_taps, dtype=complex)
        
    def train(
        self,
        received_signal: np.ndarray,
        training_symbols: np.ndarray
    ) -> None:
        """
        Train DFE using LMS algorithm with known training symbols.
        """
        n_symbols = len(training_symbols)
        
        # Buffers
        ff_buffer = np.zeros(self.n_ff_taps, dtype=complex)
        fb_buffer = np.zeros(self.n_fb_taps, dtype=complex)
        
        for n in range(n_symbols):
            # Fill feedforward buffer
            start_idx = max(0, n - self.n_ff_taps // 2)
            end_idx = min(len(received_signal), n + self.n_ff_taps // 2 + 1)
            
            ff_buffer[:] = 0
            buf_start = self.n_ff_taps // 2 - (n - start_idx)
            buf_end = buf_start + (end_idx - start_idx)
            ff_buffer[buf_start:buf_end] = received_signal[start_idx:end_idx]
            
            # DFE output
            y = np.dot(self.w_ff, ff_buffer) - np.dot(self.w_fb, fb_buffer)
            
            # Error
            e = training_symbols[n] - y
            
            # LMS update
            self.w_ff += self.mu * e * np.conj(ff_buffer)
            self.w_fb -= self.mu * e * np.conj(fb_buffer)
            
            # Update feedback buffer
            fb_buffer = np.roll(fb_buffer, 1)
            fb_buffer[0] = training_symbols[n]
    
    def equalize(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Apply trained DFE to received signal.
        """
        n_symbols = len(received_signal)
        equalized = np.zeros(n_symbols, dtype=complex)
        
        ff_buffer = np.zeros(self.n_ff_taps, dtype=complex)
        fb_buffer = np.zeros(self.n_fb_taps, dtype=complex)
        
        for n in range(n_symbols):
            # Fill feedforward buffer
            start_idx = max(0, n - self.n_ff_taps // 2)
            end_idx = min(n_symbols, n + self.n_ff_taps // 2 + 1)
            
            ff_buffer[:] = 0
            buf_start = self.n_ff_taps // 2 - (n - start_idx)
            buf_end = buf_start + (end_idx - start_idx)
            ff_buffer[buf_start:buf_end] = received_signal[start_idx:end_idx]
            
            # DFE output
            y = np.dot(self.w_ff, ff_buffer) - np.dot(self.w_fb, fb_buffer)
            equalized[n] = y
            
            # Hard decision for feedback
            decision = np.sign(np.real(y)) + 1j * np.sign(np.imag(y))
            decision = decision / np.sqrt(2)  # Normalize for QPSK
            
            # Update feedback buffer
            fb_buffer = np.roll(fb_buffer, 1)
            fb_buffer[0] = decision
        
        return equalized
    
    def equalize_iq(
        self,
        noisy_iq: np.ndarray,
        clean_iq: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Equalize I/Q signal format [2, L]."""
        noisy_complex = noisy_iq[0] + 1j * noisy_iq[1]
        
        # Train if clean signal provided
        if clean_iq is not None:
            clean_complex = clean_iq[0] + 1j * clean_iq[1]
            self.train(noisy_complex, clean_complex)
        
        equalized_complex = self.equalize(noisy_complex)
        
        equalized_iq = np.stack([
            np.real(equalized_complex),
            np.imag(equalized_complex)
        ], axis=0).astype(np.float32)
        
        metrics = {}
        if clean_iq is not None:
            mse = np.mean((equalized_iq - clean_iq) ** 2)
            metrics['mse'] = float(mse)
        
        return equalized_iq, metrics


class LMSEqualizer:
    """
    Least Mean Squares (LMS) Adaptive Equalizer.
    
    Mathematical formulation:
    -------------------------
    LMS updates filter coefficients to minimize MSE:
        e[n] = d[n] - y[n]
        w[n+1] = w[n] + mu * e[n] * conj(x[n])
    
    Pros: Simple, low complexity O(N) per iteration
    Cons: Slow convergence, sensitive to step size
    
    Complexity: O(N) per symbol
    """
    
    def __init__(self, n_taps: int = 11, mu: float = 0.01):
        """
        Initialize LMS equalizer.
        
        Args:
            n_taps: Number of filter taps
            mu: Step size (learning rate)
        """
        self.n_taps = n_taps
        self.mu = mu
        self.weights = np.zeros(n_taps, dtype=complex)
        self.weights[n_taps // 2] = 1.0  # Center tap initialization
        
    def train(
        self,
        received_signal: np.ndarray,
        desired_signal: np.ndarray,
        n_iterations: int = 1
    ) -> np.ndarray:
        """
        Train LMS equalizer.
        
        Returns error history for convergence analysis.
        """
        n_samples = len(received_signal)
        errors = []
        
        for _ in range(n_iterations):
            for n in range(self.n_taps // 2, n_samples - self.n_taps // 2):
                # Input vector
                x = received_signal[n - self.n_taps // 2 : n + self.n_taps // 2 + 1]
                
                # Filter output
                y = np.dot(self.weights, x)
                
                # Error
                e = desired_signal[n] - y
                errors.append(np.abs(e) ** 2)
                
                # LMS update
                self.weights += self.mu * e * np.conj(x)
        
        return np.array(errors)
    
    def equalize(self, received_signal: np.ndarray) -> np.ndarray:
        """Apply trained LMS equalizer."""
        n_samples = len(received_signal)
        equalized = np.zeros(n_samples, dtype=complex)
        
        for n in range(self.n_taps // 2, n_samples - self.n_taps // 2):
            x = received_signal[n - self.n_taps // 2 : n + self.n_taps // 2 + 1]
            equalized[n] = np.dot(self.weights, x)
        
        return equalized
    
    def equalize_iq(
        self,
        noisy_iq: np.ndarray,
        clean_iq: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Equalize I/Q signal format [2, L]."""
        noisy_complex = noisy_iq[0] + 1j * noisy_iq[1]
        
        if clean_iq is not None:
            clean_complex = clean_iq[0] + 1j * clean_iq[1]
            errors = self.train(noisy_complex, clean_complex)
        
        equalized_complex = self.equalize(noisy_complex)
        
        equalized_iq = np.stack([
            np.real(equalized_complex),
            np.imag(equalized_complex)
        ], axis=0).astype(np.float32)
        
        metrics = {}
        if clean_iq is not None:
            mse = np.mean((equalized_iq - clean_iq) ** 2)
            metrics['mse'] = float(mse)
            metrics['convergence_mse'] = float(errors[-100:].mean()) if len(errors) > 100 else float(errors.mean())
        
        return equalized_iq, metrics


class RLSEqualizer:
    """
    Recursive Least Squares (RLS) Adaptive Equalizer.
    
    Mathematical formulation:
    -------------------------
    RLS provides faster convergence than LMS by using inverse correlation:
        k[n] = P[n-1] * x[n] / (lambda + x[n]^H * P[n-1] * x[n])
        e[n] = d[n] - w[n-1]^H * x[n]
        w[n] = w[n-1] + k[n] * conj(e[n])
        P[n] = (P[n-1] - k[n] * x[n]^H * P[n-1]) / lambda
    
    Pros: Fast convergence, excellent tracking
    Cons: Higher complexity O(N²), numerical stability issues
    
    Complexity: O(N²) per symbol
    """
    
    def __init__(
        self, 
        n_taps: int = 11, 
        forgetting_factor: float = 0.99,
        delta: float = 0.1
    ):
        """
        Initialize RLS equalizer.
        
        Args:
            n_taps: Number of filter taps
            forgetting_factor: Lambda (0.9-1.0, higher = slower adaptation)
            delta: Regularization for initial P matrix
        """
        self.n_taps = n_taps
        self.lam = forgetting_factor
        self.delta = delta
        
        # Initialize
        self.weights = np.zeros(n_taps, dtype=complex)
        self.weights[n_taps // 2] = 1.0
        self.P = np.eye(n_taps, dtype=complex) / delta
        
    def train(
        self,
        received_signal: np.ndarray,
        desired_signal: np.ndarray
    ) -> np.ndarray:
        """Train RLS equalizer. Returns error history."""
        n_samples = len(received_signal)
        errors = []
        
        for n in range(self.n_taps // 2, n_samples - self.n_taps // 2):
            # Input vector
            x = received_signal[n - self.n_taps // 2 : n + self.n_taps // 2 + 1]
            x = x.reshape(-1, 1)
            
            # Gain vector
            Px = self.P @ x
            denom = self.lam + (x.conj().T @ Px)[0, 0]
            k = Px / denom
            
            # A priori error
            y = (self.weights.reshape(-1, 1).conj().T @ x)[0, 0]
            e = desired_signal[n] - y
            errors.append(np.abs(e) ** 2)
            
            # Update weights
            self.weights += (k * np.conj(e)).flatten()
            
            # Update inverse correlation matrix
            self.P = (self.P - k @ x.conj().T @ self.P) / self.lam
        
        return np.array(errors)
    
    def equalize(self, received_signal: np.ndarray) -> np.ndarray:
        """Apply trained RLS equalizer."""
        n_samples = len(received_signal)
        equalized = np.zeros(n_samples, dtype=complex)
        
        for n in range(self.n_taps // 2, n_samples - self.n_taps // 2):
            x = received_signal[n - self.n_taps // 2 : n + self.n_taps // 2 + 1]
            equalized[n] = np.dot(self.weights.conj(), x)
        
        return equalized
    
    def equalize_iq(
        self,
        noisy_iq: np.ndarray,
        clean_iq: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Equalize I/Q signal format [2, L]."""
        noisy_complex = noisy_iq[0] + 1j * noisy_iq[1]
        
        if clean_iq is not None:
            clean_complex = clean_iq[0] + 1j * clean_iq[1]
            errors = self.train(noisy_complex, clean_complex)
        
        equalized_complex = self.equalize(noisy_complex)
        
        equalized_iq = np.stack([
            np.real(equalized_complex),
            np.imag(equalized_complex)
        ], axis=0).astype(np.float32)
        
        metrics = {}
        if clean_iq is not None:
            mse = np.mean((equalized_iq - clean_iq) ** 2)
            metrics['mse'] = float(mse)
        
        return equalized_iq, metrics


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_equalizers(
    noisy_iq: np.ndarray,
    clean_iq: np.ndarray,
    snr_db: float = 20.0
) -> Dict[str, Dict[str, float]]:
    """
    Compare all classical equalizers on the same signal.
    
    Args:
        noisy_iq: Noisy I/Q signal [2, L]
        clean_iq: Clean I/Q signal [2, L]
        snr_db: SNR for MMSE
        
    Returns:
        Dictionary of equalizer results
    """
    results = {}
    
    # Zero-Forcing
    zf = ZeroForcingEqualizer()
    _, zf_metrics = zf.equalize_iq(noisy_iq, clean_iq)
    results['ZF'] = zf_metrics
    
    # MMSE
    mmse = MMSEEqualizer()
    _, mmse_metrics = mmse.equalize_iq(noisy_iq, clean_iq, snr_db)
    results['MMSE'] = mmse_metrics
    
    # DFE
    dfe = DecisionFeedbackEqualizer(n_ff_taps=5, n_fb_taps=3)
    _, dfe_metrics = dfe.equalize_iq(noisy_iq, clean_iq)
    results['DFE'] = dfe_metrics
    
    # LMS
    lms = LMSEqualizer(n_taps=11, mu=0.01)
    _, lms_metrics = lms.equalize_iq(noisy_iq, clean_iq)
    results['LMS'] = lms_metrics
    
    # RLS
    rls = RLSEqualizer(n_taps=11, forgetting_factor=0.99)
    _, rls_metrics = rls.equalize_iq(noisy_iq, clean_iq)
    results['RLS'] = rls_metrics
    
    return results


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Classical Equalizers Verification")
    print("=" * 60)
    
    # Generate test signal
    np.random.seed(42)
    n_samples = 64
    
    # Clean signal (QPSK-like)
    clean_complex = (np.random.choice([-1, 1], n_samples) + 
                     1j * np.random.choice([-1, 1], n_samples)) / np.sqrt(2)
    
    # Add channel effect + noise
    snr_db = 15
    signal_power = np.mean(np.abs(clean_complex) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(n_samples) + 
                                         1j * np.random.randn(n_samples))
    
    # Simple multipath channel
    h = np.array([1.0, 0.3 + 0.2j, 0.1 - 0.1j])
    noisy_complex = np.convolve(clean_complex, h, mode='same') + noise
    
    # Convert to I/Q format
    clean_iq = np.stack([np.real(clean_complex), np.imag(clean_complex)], axis=0).astype(np.float32)
    noisy_iq = np.stack([np.real(noisy_complex), np.imag(noisy_complex)], axis=0).astype(np.float32)
    
    # Compare equalizers
    print("\n--- Equalizer Comparison (SNR=15dB, multipath channel) ---")
    results = compare_equalizers(noisy_iq, clean_iq, snr_db=snr_db)
    
    print("\nEqualizer Performance (MSE):")
    print("-" * 40)
    for name, metrics in results.items():
        mse = metrics.get('mse', float('nan'))
        print(f"  {name:8s}: MSE = {mse:.6f}")
    
    print("\n✓ Classical equalizers verification complete!")
