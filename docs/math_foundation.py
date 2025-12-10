# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Mathematical Foundation Document
# =============================================================================

"""
================================================================================
                    MATHEMATICAL FOUNDATION
                    CWGAN-GP for OFDM Signal Reconstruction
================================================================================

TABLE OF CONTENTS
-----------------
1. OFDM Signal Model
2. Wireless Channel Models
3. GAN Theory (WGAN-GP)
4. Conditional GAN (CWGAN-GP)
5. 1D U-Net Architecture
6. Convolution Mathematics
7. Activation Functions
8. Loss Functions
9. Quantization for FPGA
10. Hardware Implementation Math

================================================================================
1. OFDM SIGNAL MODEL
================================================================================

ORTHOGONAL FREQUENCY DIVISION MULTIPLEXING (OFDM)
-------------------------------------------------

Basic Principle:
    Divide a high-rate data stream into N parallel low-rate streams,
    each modulated onto orthogonal subcarriers.

Time-Domain OFDM Symbol Generation:
    
    x[n] = (1/√N) Σ_{k=0}^{N-1} X[k] · exp(j·2π·k·n/N)
    
    Where:
        - N = Number of subcarriers (e.g., 64)
        - X[k] = Complex QAM symbol on subcarrier k
        - x[n] = Time-domain sample n
        - j = √(-1)
    
    This is equivalent to the Inverse DFT (IDFT).

Frequency-Domain Representation:
    
    X[k] = (1/√N) Σ_{n=0}^{N-1} x[n] · exp(-j·2π·k·n/N)
    
    This is the DFT, used at the receiver.

Cyclic Prefix:
    To combat Inter-Symbol Interference (ISI), copy last L_cp samples:
    
    x_cp[n] = x[(n - L_cp) mod N], for n = 0, 1, ..., N + L_cp - 1
    
    Creates circular convolution with channel, enabling simple equalization.

I/Q Representation:
    Complex signal x = I + jQ is stored as 2-channel real tensor:
    
    x_tensor = [I[0], I[1], ..., I[L-1]]
               [Q[0], Q[1], ..., Q[L-1]]  ∈ ℝ^(2×L)

================================================================================
2. WIRELESS CHANNEL MODELS
================================================================================

AWGN CHANNEL (Additive White Gaussian Noise)
--------------------------------------------
    y = x + n
    
    Where n ~ CN(0, σ²I), complex circular Gaussian noise.
    
    σ² = P_x / (10^(SNR_dB/10))
    
    P_x = E[|x|²] = signal power

RAYLEIGH FADING CHANNEL
-----------------------
    y = h·x + n
    
    Where h ~ CN(0, 1), |h| follows Rayleigh distribution:
    
    f_{|h|}(r) = 2r·exp(-r²), r ≥ 0
    
    E[|h|²] = 1 (normalized)
    
    Models non-line-of-sight propagation with rich scattering.

RICIAN FADING CHANNEL
---------------------
    h = √(K/(K+1))·exp(jθ) + √(1/(K+1))·h_NLOS
    
    Where:
        - K = Rician K-factor (LOS/NLOS power ratio)
        - θ ~ U(0, 2π) = random LOS phase
        - h_NLOS ~ CN(0, 1) = scattered component
    
    |h| follows Rician distribution:
    
    f_{|h|}(r) = 2r(K+1)·exp(-K-(K+1)r²)·I_0(2r√(K(K+1)))
    
    Where I_0 is the modified Bessel function of first kind.

MULTIPATH CHANNEL
-----------------
    y[n] = Σ_{l=0}^{L-1} h[l]·x[n-l] + n[n]
    
    Channel impulse response h[l] with L taps.
    
    Frequency-selective fading in OFDM:
        Y[k] = H[k]·X[k] + N[k]
    
    Where H[k] = DFT{h[l]} is the channel frequency response.

================================================================================
3. GAN THEORY (WGAN-GP)
================================================================================

GENERATIVE ADVERSARIAL NETWORKS
-------------------------------
Two networks competing in a minimax game:
    
    min_G max_D V(D, G) = E_{x~P_data}[log D(x)] + E_{z~P_z}[log(1 - D(G(z)))]
    
    - Generator G: maps noise z to data-like samples
    - Discriminator D: classifies real vs fake

WASSERSTEIN GAN (WGAN)
----------------------
Uses Wasserstein-1 distance (Earth Mover's Distance):
    
    W(P_r, P_g) = inf_{γ∈Π(P_r,P_g)} E_{(x,y)~γ}[||x - y||]
    
    Kantorovich-Rubinstein duality:
    
    W(P_r, P_g) = sup_{||f||_L ≤ 1} E_{x~P_r}[f(x)] - E_{x~P_g}[f(x)]
    
    Where ||f||_L ≤ 1 means f is 1-Lipschitz continuous.

WGAN CRITIC OBJECTIVE:
    L_D = E_{x~P_g}[D(x)] - E_{x~P_r}[D(x)]
    
    Minimize this (discriminator outputs unbounded scores, not probabilities).

WGAN GENERATOR OBJECTIVE:
    L_G = -E_{x~P_g}[D(x)]
    
    Maximize discriminator score on generated samples.

GRADIENT PENALTY (WGAN-GP)
--------------------------
Enforce 1-Lipschitz constraint via gradient penalty:
    
    GP = E_{x̂~P_{x̂}}[(||∇_{x̂}D(x̂)||_2 - 1)²]
    
    Where x̂ = εx_real + (1-ε)x_fake, ε ~ U(0,1)
    
    Interpolated samples between real and fake.

FULL WGAN-GP CRITIC LOSS:
    L_D = E[D(G(z))] - E[D(x)] + λ·GP
    
    Where λ = 10 (gradient penalty coefficient).

================================================================================
4. CONDITIONAL GAN (CWGAN-GP)
================================================================================

CONDITIONAL GENERATION
----------------------
Learn mapping from condition c to output:
    
    G: (c) → y
    D: (x, c) → validity score
    
For OFDM enhancement:
    - c = noisy I/Q signal
    - y = clean I/Q signal
    - G(c) = enhanced I/Q signal

CONDITIONAL DISCRIMINATOR
-------------------------
Discriminator receives concatenated input:
    
    input = concat(candidate, condition)
    
    Shape: [B, 4, L] for our case (2 candidate + 2 condition channels)
    
    D(x, c) outputs scalar score for pair (x, c).

TRAINING PAIRS:
    Real pair: (x_clean, x_noisy) → high score
    Fake pair: (G(x_noisy), x_noisy) → low score

CWGAN-GP LOSSES:
    
    Critic Loss:
        L_D = E[D(G(c), c)] - E[D(x, c)] + λ·GP
    
    Generator Loss:
        L_G = -E[D(G(c), c)] + λ_rec·||G(c) - x||_1
    
    Where λ_rec = 100 is reconstruction loss weight.

================================================================================
5. 1D U-NET ARCHITECTURE
================================================================================

U-NET STRUCTURE
---------------
Encoder-decoder with skip connections:

    Input                                                     Output
      ↓                                                         ↑
    [Enc1] ──────────────────────────────────────────────→ [Dec1]
      ↓                                                         ↑
    [Enc2] ──────────────────────────────────────────→ [Dec2]
      ↓                                                     ↑
    [Enc3] ──────────────────────────────────→ [Dec3]
      ↓                                             ↑
    [Enc4] ──────────────────────────→ [Dec4]
      ↓                                     ↑
    [Enc5] ──────────────→ [Dec5]
      ↓                         ↑
    [Bottleneck] ───────────────┘

ENCODER BLOCK:
    E_i(x) = LeakyReLU(Conv1D_s1(LeakyReLU(Conv1D_s2(x))))
    
    - Conv1D_s2: stride=2 convolution (downsample by 2×)
    - Conv1D_s1: stride=1 convolution (same resolution)
    - Output resolution: L/2 compared to input

DECODER BLOCK:
    D_i(x, skip) = LeakyReLU(Conv1D(LeakyReLU(Conv1D(Upsample(x) + skip))))
    
    - Upsample: Nearest neighbor 2× upsampling
    - Skip: Additive skip connection from encoder
    - Output resolution: 2L compared to input

SKIP CONNECTIONS (ADDITIVE):
    merged = upsample(decoder_input) + encoder_output
    
    Requires matching dimensions (ensured by symmetric architecture).
    
    Advantages over concatenation:
    - Fewer parameters in subsequent conv layers
    - Gradient flow improvement
    - Memory efficiency

CHANNEL PROGRESSION:
    Level  Encoder         Bottleneck      Decoder
    -----  --------------  --------------  --------------
    0      2 → 32          -               32 → 2
    1      32 → 64         -               64 → 32
    2      64 → 128        -               128 → 64
    3      128 → 256       -               256 → 128
    4      256 → 512       -               512 → 256
    BN     -               512 → 512       -

================================================================================
6. CONVOLUTION MATHEMATICS
================================================================================

1D CONVOLUTION
--------------
Discrete convolution with kernel w of size K:
    
    y[n] = Σ_{k=0}^{K-1} w[k] · x[n·s + k - p] + b
    
    Where:
        - s = stride
        - p = padding
        - b = bias

OUTPUT LENGTH:
    L_out = floor((L_in + 2p - K) / s) + 1
    
    For K=3, s=2, p=1:
        L_out = floor((L_in + 2 - 3) / 2) + 1 = floor((L_in - 1) / 2) + 1
        
        For L_in = 1024: L_out = 512

PARAMETER COUNT:
    params = K · C_in · C_out + C_out
    
    Where:
        - K = kernel size
        - C_in = input channels
        - C_out = output channels
        - +C_out for bias

MAC COUNT (Multiply-Accumulate):
    MACs = K · C_in · C_out · L_out
    
    Per output sample: K · C_in · C_out multiplications and additions.

EXAMPLE (enc1_1):
    K=3, C_in=2, C_out=32, L_out=512
    
    params = 3 · 2 · 32 + 32 = 192 + 32 = 224  (wait, should include all)
           = 3 · 2 · 32 + 32 = 224
    
    Actual with bias: 3 × 2 × 32 + 32 = 224

================================================================================
7. ACTIVATION FUNCTIONS
================================================================================

LEAKY RELU
----------
    LeakyReLU(x) = max(αx, x) = {  x    if x > 0
                                  αx   if x ≤ 0

    Where α = 0.2 (negative slope).
    
    Gradient:
        d/dx LeakyReLU(x) = { 1   if x > 0
                             α   if x ≤ 0

    Advantages:
    - Prevents "dying ReLU" problem
    - Non-zero gradient for negative inputs

HARDWARE IMPLEMENTATION:
    if (x > 0):
        y = x
    else:
        y = x >> 2  (for α = 0.25, use right shift)
        # Or: y = (x * α_fixed) >> scale
        
    For α = 0.2 ≈ 0.203125 = 13/64:
        y = (x * 13) >> 6

TANH (Hyperbolic Tangent)
-------------------------
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
            = 2σ(2x) - 1
            = (e^(2x) - 1) / (e^(2x) + 1)
    
    Range: (-1, 1)
    
    Used for final layer to bound output to normalized I/Q range.

HARDWARE IMPLEMENTATION (LUT):
    - Use 256-entry lookup table
    - Input: INT8 index from saturated input
    - Output: INT16 tanh value
    
    Piecewise linear approximation:
        tanh(x) ≈ { -1           if x < -3
                   x/3          if -3 ≤ x < 3
                   1            if x ≥ 3
                   
    Or higher accuracy with more segments.

================================================================================
8. LOSS FUNCTIONS
================================================================================

WASSERSTEIN LOSS (CRITIC)
-------------------------
    L_D = E[D(fake)] - E[D(real)] + λ·GP
    
    Components:
    - E[D(fake)]: Mean critic score on generated samples
    - E[D(real)]: Mean critic score on real samples
    - GP: Gradient penalty term
    - λ = 10: Gradient penalty coefficient

GRADIENT PENALTY
----------------
    GP = E[(||∇_x̂ D(x̂)||_2 - 1)²]
    
    Computation:
    1. Sample ε ~ U(0, 1)
    2. Create interpolated: x̂ = ε·real + (1-ε)·fake
    3. Forward pass: D(x̂)
    4. Compute gradient: ∇_x̂ D(x̂)
    5. L2 norm: ||∇||_2
    6. Penalty: (||∇||_2 - 1)²

GENERATOR LOSS
--------------
    L_G = L_adv + λ_rec · L_rec
    
    Adversarial Loss:
        L_adv = -E[D(G(c), c)]
    
    Reconstruction Loss (L1):
        L_rec = E[|G(c) - x_real|]
        
        L1 preferred over L2:
        - L1: Encourages sparse errors, less blurry
        - L2: Penalizes large errors more, can cause blur

    λ_rec = 100: Reconstruction weight (high for supervised signal recovery)

================================================================================
9. QUANTIZATION FOR FPGA
================================================================================

FIXED-POINT REPRESENTATION
--------------------------
Signed integer representation with implicit scaling:
    
    x_float ≈ x_int · 2^(-f)
    
    Where f = number of fractional bits.
    
    For pure integer (no fractional):
        x_float = x_int · scale

SYMMETRIC QUANTIZATION
----------------------
    scale = max|x| / (2^(b-1) - 1)
    
    x_quant = round(clamp(x / scale, -2^(b-1), 2^(b-1) - 1))
    
    x_dequant = x_quant · scale

PER-CHANNEL QUANTIZATION
------------------------
    For weights W with shape [C_out, C_in, K]:
    
    scale_c = max(|W[c, :, :]|) / 127
    
    Each output channel has its own scale factor.
    
    Advantage: Better precision for channels with different magnitudes.

QUANTIZATION RANGES:
    INT8 weights: [-128, 127]
    INT16 activations: [-32768, 32767]
    INT32 accumulators: [-2^31, 2^31 - 1]

ACCUMULATOR OVERFLOW PREVENTION:
    For K×C_in multiply-accumulates with INT8 weights and INT16 activations:
    
    Max accumulator value: K · C_in · 127 · 32767
    
    For K=3, C_in=512:
        3 × 512 × 127 × 32767 ≈ 6.4 × 10^9
    
    INT32 max: 2.1 × 10^9
    
    → Need careful scaling or use INT48/INT64 accumulators.

REQUANTIZATION
--------------
After accumulation, scale back to INT16:
    
    y_int16 = clamp(round(y_int32 · output_scale / (weight_scale · input_scale)), -32768, 32767)
    
    This is often implemented as:
        y = (y_acc · mult) >> shift
    
    Where mult and shift are precomputed for each layer.

================================================================================
10. HARDWARE IMPLEMENTATION MATH
================================================================================

MAC OPERATIONS
--------------
Total MACs for generator: ~365 million per frame
    
    Throughput requirement for real-time:
    - 30 fps: 365M × 30 = 10.95 GMAC/s
    - 60 fps: 365M × 60 = 21.9 GMAC/s

DSP UTILIZATION
---------------
    DSP48 slice: 1 MAC per cycle (INT16 × INT8)
    
    At 200 MHz with N DSPs:
        Throughput = N × 200 × 10^6 MAC/s
    
    For 30 fps: N ≥ 10.95 × 10^9 / (200 × 10^6) = 54.75 ≈ 55 DSPs
    For 60 fps: N ≥ 110 DSPs

MEMORY BANDWIDTH
----------------
Weight memory (generator): 5.5 MB (INT8)
    
    For 30 fps: 5.5 MB × 30 = 165 MB/s
    
    DDR4 bandwidth: ~25 GB/s → Easily sufficient

Skip buffer memory: 160 KB (INT16)
    - Can fit in BRAM for most FPGAs

LATENCY CALCULATION
-------------------
Pipeline latency (approximate):
    
    L_total = Σ_layers (L_compute + L_memory)
    
    L_compute per layer ≈ (MACs_layer) / (N_DSP × f_clk)
    L_memory ≈ weight_fetch_cycles + activation_read_cycles

For streaming implementation:
    Latency ≈ 2 × frame_length × (encoder_depth + decoder_depth)
             ≈ 2 × 1024 × 10 = 20,480 cycles
             At 200 MHz: ~102 μs

LINE BUFFER REQUIREMENTS
------------------------
For K=3 convolution:
    Need K-1 = 2 previous samples per channel
    
    Per layer: 2 × C_in samples
    
    Maximum (at 512 channels): 2 × 512 = 1024 samples
    Storage: 1024 × 16 bits = 2 KB per layer

UPSAMPLE IMPLEMENTATION
-----------------------
Nearest neighbor 2×:
    y[2n] = y[2n+1] = x[n]
    
    Hardware: Just duplicate each sample (no multiplies)
    
    Can be done with FSM or simple counter logic.

TANH LUT SIZING
---------------
For INT8 input index: 256 entries
For INT16 output: 256 × 2 bytes = 512 bytes

Linear interpolation for better accuracy:
    idx = x >> input_shift
    frac = x & ((1 << input_shift) - 1)
    y = lut[idx] + (lut[idx+1] - lut[idx]) × frac >> frac_bits

================================================================================
END OF MATHEMATICAL FOUNDATION
================================================================================
"""

# Make this module runnable to display the documentation
if __name__ == "__main__":
    print(__doc__)
