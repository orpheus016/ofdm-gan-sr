# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Mathematical Foundation Document - MINI ARCHITECTURE
# =============================================================================

"""
================================================================================
                    MATHEMATICAL FOUNDATION
                    CWGAN-GP for OFDM Signal Reconstruction
                    MINI ARCHITECTURE FOR FPGA
================================================================================

TABLE OF CONTENTS
-----------------
1. OFDM Signal Model
2. Wireless Channel Models
3. GAN Theory (WGAN-GP)
4. Conditional GAN (CWGAN-GP)
5. Mini U-Net Architecture
6. Convolution Mathematics
7. Activation Functions
8. Loss Functions
9. Fixed-Point Quantization (Q1.7 / Q8.8)
10. RTL Hardware Implementation

================================================================================
1. OFDM SIGNAL MODEL
================================================================================

ORTHOGONAL FREQUENCY DIVISION MULTIPLEXING (OFDM)
-------------------------------------------------

Basic Principle:
    Divide data into N parallel streams, each modulated onto orthogonal subcarriers.

Time-Domain OFDM Symbol Generation:
    
    x[n] = (1/√N) Σ_{k=0}^{N-1} X[k] · exp(j·2π·k·n/N)
    
    Where:
        - N = Number of subcarriers (8 for mini architecture)
        - X[k] = QPSK symbol on subcarrier k
        - x[n] = Time-domain sample n

I/Q Representation:
    Complex signal x = I + jQ stored as 2-channel real tensor:
    
    x_tensor = [I[0], I[1], ..., I[15]]   ∈ ℝ^(2×16)
               [Q[0], Q[1], ..., Q[15]]  

QPSK MODULATION:
    Symbols in {(+1+j)/sqrt(2), (+1-j)/sqrt(2), (-1+j)/sqrt(2), (-1-j)/sqrt(2)}
    Normalized power = 1

================================================================================
2. WIRELESS CHANNEL MODELS
================================================================================

AWGN CHANNEL (Additive White Gaussian Noise)
--------------------------------------------
    y = x + n
    
    Where n ~ CN(0, σ²I), complex circular Gaussian noise.
    
    σ² = P_x / (10^(SNR_dB/10))
    P_x = E[|x|²] = signal power

SNR Range for Training: 5 - 20 dB

================================================================================
3. GAN THEORY (WGAN-GP)
================================================================================

WASSERSTEIN GAN (WGAN)
----------------------
Uses Wasserstein-1 distance (Earth Mover's Distance):
    
    W(P_r, P_g) = sup_{||f||_L ≤ 1} E_{x~P_r}[f(x)] - E_{x~P_g}[f(x)]

WGAN CRITIC OBJECTIVE:
    L_D = E_{x~P_g}[D(x)] - E_{x~P_r}[D(x)]

GRADIENT PENALTY (WGAN-GP)
--------------------------
Enforce 1-Lipschitz constraint:
    
    GP = E_{x̂~P_{x̂}}[(||∇_{x̂}D(x̂)||_2 - 1)²]
    
    Where x̂ = εx_real + (1-ε)x_fake, ε ~ U(0,1)

FULL WGAN-GP CRITIC LOSS:
    L_D = E[D(G(c))] - E[D(x)] + λ·GP
    
    Where λ = 10 (gradient penalty coefficient).

================================================================================
4. CONDITIONAL GAN (CWGAN-GP)
================================================================================

CONDITIONAL GENERATION
----------------------
    G: (c) → y      (noisy → enhanced)
    D: (x, c) → score

For OFDM enhancement:
    - c = noisy I/Q signal [2×16]
    - y = clean I/Q signal [2×16]
    - G(c) = enhanced I/Q signal [2×16]

CONDITIONAL DISCRIMINATOR
-------------------------
    input = concat(candidate[2×16], condition[2×16]) = [4×16]
    
    D(x, c) outputs scalar score.

CWGAN-GP LOSSES:
    
    Critic Loss:
        L_D = E[D(G(c), c)] - E[D(x, c)] + λ·GP
    
    Generator Loss:
        L_G = -E[D(G(c), c)] + λ_rec·||G(c) - x||_1
    
    Where λ_rec = 100 is reconstruction loss weight.

================================================================================
5. MINI U-NET ARCHITECTURE
================================================================================

MINI U-NET STRUCTURE (Matches RTL)
----------------------------------

    Input [2×16]
         ↓
    [Enc1: 2→4, s=2] ──────────┐
         ↓                      │ (Skip Connection)
    [Bottleneck: 4→8, s=2]     │
         ↓                      │
    [Upsample ×2]              │
         ↓                      │
    [Dec1: 8→4, s=1] ←─────────┘ (Add)
         ↓
    [Upsample ×2]
         ↓
    [OutConv: 4→2, s=1]
         ↓
    [Tanh]
         ↓
    Output [2×16]

LAYER SPECIFICATIONS:
---------------------
| Layer      | In Ch | Out Ch | Stride | L_in | L_out | Params |
|------------|-------|--------|--------|------|-------|--------|
| Enc1       | 2     | 4      | 2      | 16   | 8     | 28     |
| Bottleneck | 4     | 8      | 2      | 8    | 4     | 104    |
| Dec1       | 8     | 4      | 1      | 8    | 8     | 100    |
| OutConv    | 4     | 2      | 1      | 16   | 16    | 26     |
|------------|-------|--------|--------|------|-------|--------|
| TOTAL      |       |        |        |      |       | 258    |

SKIP CONNECTION (ADDITIVE):
    dec1_out = LeakyReLU(Conv(upsample(bottleneck_out)))
    merged = dec1_out + enc1_out

MINI DISCRIMINATOR STRUCTURE
----------------------------
| Layer  | In Ch | Out Ch | Stride | L_out | Params |
|--------|-------|--------|--------|-------|--------|
| Conv1  | 4     | 8      | 2      | 8     | 104    |
| Conv2  | 8     | 16     | 2      | 4     | 400    |
| Pool   | 16    | 16     | -      | 1     | 0      |
| Dense  | 16    | 1      | -      | 1     | 17     |
|--------|-------|--------|--------|-------|--------|
| TOTAL  |       |        |        |       | 521    |

================================================================================
6. CONVOLUTION MATHEMATICS
================================================================================

1D CONVOLUTION
--------------
    y[n] = Σ_{k=0}^{K-1} w[k] · x[n·s + k - p] + b
    
OUTPUT LENGTH:
    L_out = floor((L_in + 2p - K) / s) + 1
    
    For K=3, s=2, p=1:
        L_out = floor((L_in - 1) / 2) + 1
        For L_in = 16: L_out = 8
        For L_in = 8: L_out = 4

PARAMETER COUNT:
    params = K · C_in · C_out + C_out (bias)
    
    Enc1: 3 × 2 × 4 + 4 = 28
    Bottleneck: 3 × 4 × 8 + 8 = 104
    Dec1: 3 × 8 × 4 + 4 = 100
    OutConv: 3 × 4 × 2 + 2 = 26

MAC COUNT (per layer):
    MACs = K · C_in · C_out · L_out
    
    Enc1: 3 × 2 × 4 × 8 = 192
    Bottleneck: 3 × 4 × 8 × 4 = 384
    Dec1: 3 × 8 × 4 × 8 = 768
    OutConv: 3 × 4 × 2 × 16 = 384
    TOTAL: ~1,728 MACs/frame

================================================================================
7. ACTIVATION FUNCTIONS
================================================================================

LEAKY RELU
----------
    LeakyReLU(x) = max(αx, x) = {  x    if x > 0
                                  αx   if x ≤ 0
    
    Where α = 0.2 (negative slope).

HARDWARE IMPLEMENTATION (RTL):
    For α = 0.2 ≈ 0.203125 = 13/64:
        if (x > 0):
            y = x
        else:
            y = (x * 13) >> 6
    
    Or simpler with α = 0.25:
        y = (x < 0) ? (x >>> 2) : x

TANH (Hyperbolic Tangent)
-------------------------
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)

HARDWARE IMPLEMENTATION (LUT):
    - 256-entry lookup table indexed by Q8.8 input
    - Output: Q8.8 tanh value
    - Linear interpolation for better accuracy

================================================================================
8. LOSS FUNCTIONS
================================================================================

WASSERSTEIN LOSS (CRITIC)
-------------------------
    L_D = E[D(fake)] - E[D(real)] + λ·GP
    
    λ = 10 (gradient penalty coefficient)

GENERATOR LOSS
--------------
    L_G = L_adv + λ_rec · L_rec
    
    Adversarial: L_adv = -E[D(G(c), c)]
    Reconstruction: L_rec = E[|G(c) - x_real|]  (L1 loss)
    
    λ_rec = 100 (reconstruction weight)

================================================================================
9. FIXED-POINT QUANTIZATION (Q1.7 / Q8.8)
================================================================================

Q1.7 FORMAT (WEIGHTS - 8-bit signed)
------------------------------------
    x_float ≈ x_int / 128
    
    Range: [-1.0, +0.9921875]
    Resolution: 1/128 ≈ 0.0078
    
    Conversion:
        x_q17 = clamp(round(x_float × 128), -128, 127)
        x_float = x_q17 / 128

Q8.8 FORMAT (ACTIVATIONS - 16-bit signed)
-----------------------------------------
    x_float ≈ x_int / 256
    
    Range: [-128.0, +127.996]
    Resolution: 1/256 ≈ 0.0039
    
    Conversion:
        x_q88 = clamp(round(x_float × 256), -32768, 32767)
        x_float = x_q88 / 256

ACCUMULATOR (Q16.16 - 32-bit signed)
------------------------------------
    For MAC operations: weight (Q1.7) × activation (Q8.8)
    Product is Q9.15, accumulated in Q16.16
    
    After accumulation, shift right by 7 to get Q8.8 result:
        result_q88 = (accumulator + 64) >> 7

================================================================================
10. RTL HARDWARE IMPLEMENTATION
================================================================================

ARCHITECTURE OVERVIEW
---------------------
    Input Buffer [2×16 Q8.8] → Conv Engine → Skip RAM → Upsample → Tanh LUT → Output

CONV1D ENGINE (Parallel k=3):
    - 3 multipliers per output channel
    - Pipelined: fetch weights, multiply, accumulate, activate
    - Supports stride=1 and stride=2

STATE MACHINE (Generator):
    ST_IDLE → ST_LOAD → ST_ENC1 → ST_BOTTLE → ST_DEC1 → ST_OUT → ST_TANH → ST_OUTPUT → ST_DONE

RESOURCE ESTIMATES:
    LUTs: ~2,000-4,000
    FFs:  ~1,000-2,000
    DSPs: 4-8 (for parallel MAC)
    BRAM: ~1-2 KB (weights + skip buffer)

LATENCY:
    Estimated: ~500-1000 cycles per frame
    At 100 MHz: ~5-10 μs per frame

THROUGHPUT:
    At 100 MHz with 1000 cycles/frame: 100,000 frames/sec

================================================================================
                    END OF MATHEMATICAL FOUNDATION
================================================================================
"""

# Make this module runnable to display the documentation
if __name__ == "__main__":
    print(__doc__)
