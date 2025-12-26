# CWGAN-GP for OFDM Signal Reconstruction

A Conditional Wasserstein GAN with Gradient Penalty for OFDM signal reconstruction, designed for FPGA deployment.

## Overview

This project implements a compact 1D U-Net based GAN for enhancing noisy OFDM I/Q signals. The architecture is specifically designed for resource-constrained FPGA deployment with minimal parameters while maintaining signal reconstruction capability.

### Key Features

- **Mini U-Net Generator**: Compact encoder-decoder with additive skip connections
- **CWGAN-GP Training**: Stable training with gradient penalty
- **FPGA-Ready**: Q1.7 weights (8-bit), Q8.8 activations (16-bit), ~800 total parameters
- **OFDM Simulation**: QPSK modulation, OFDM encoding, and AWGN channel simulation
- **Non-Linear Impairments**: PA compression, IQ imbalance, phase noise simulation
- **Classical Equalizers**: ZF, MMSE, DFE, LMS, RLS for baseline comparison
- **RTL Implementation**: Complete Verilog implementation in `rtl/` folder

### LSI Design Contest Innovation

**Why Neural Networks Beat Classical Methods for Non-Linear Distortion:**

Classical equalizers (ZF, MMSE, DFE) assume linear channel models:
```
y = H·x + n  (linear)
```

Real RF systems have non-linear impairments:
```
y = f_nonlinear(H·x) + n  (PA compression, IQ imbalance, phase noise)
```

The CWGAN-GP learns to compensate these non-linearities through data-driven training, achieving:
- **3-5 dB improvement** over MMSE with PA compression
- **Single-pass inference** vs iterative classical algorithms
- **Fixed-point FPGA implementation** with ~800 parameters

## Architecture

```
Generator (Mini U-Net):
Input [2×16] → Encoder → Bottleneck → Decoder → Output [2×16]
     ↓              ↓                      ↑           ↑
   I/Q signal    Skip connections ─────────┘      Enhanced I/Q

Discriminator (Mini Critic):
[Candidate + Condition] → Conv Stack → Sum Pool → Dense → Score
        [4×16]                                              [1]
```

### Generator Specifications (Mini Architecture)

| Layer      | Channels    | Kernel | Output Size |
|------------|-------------|--------|-------------|
| Input      | 2           | -      | 2×16        |
| Encoder 1  | 2 → 4       | k=3    | 4×8         |
| Bottleneck | 4 → 8       | k=3    | 8×8         |
| Decoder 1  | 8+4 → 4     | k=3    | 4×16        |
| Output     | 4 → 2       | k=3    | 2×16        |

**Total Generator Parameters**: ~258  
**Total Discriminator Parameters**: ~521  
**Total Parameters**: ~800

### Discriminator Specifications (Mini Architecture)

| Layer  | Channels    | Kernel | Output Size |
|--------|-------------|--------|-------------|
| Input  | 4           | -      | 4×16        |
| Conv 1 | 4 → 8       | k=3    | 8×8         |
| Conv 2 | 8 → 16      | k=3    | 16×4        |
| Pool   | Sum Pool    | -      | 16×1        |
| Dense  | 16 → 1      | -      | 1           |

## Project Structure

```
ofdm-gan-sr/
├── config/
│   └── config.yaml          # Configuration file (mini architecture)
├── models/
│   ├── __init__.py
│   ├── generator.py         # Mini U-Net Generator (258 params)
│   └── discriminator.py     # Mini Conditional Critic (521 params)
├── utils/
│   ├── __init__.py
│   ├── ofdm_utils.py        # OFDM/QPSK/Channel utilities
│   ├── dataset.py           # Data loading and generation
│   ├── quantization.py      # FPGA quantization utilities
│   └── export_mini_weights.py  # RTL weight export
├── rtl/
│   ├── generator_mini.v     # Verilog generator implementation
│   ├── discriminator_mini.v # Verilog discriminator implementation
│   ├── cwgan_gp_top.v       # Top-level RTL module
│   ├── tb_*.v               # Testbenches
│   └── *.v                  # Supporting RTL modules
├── proof/
│   ├── __init__.py
│   └── verification.py      # Architecture verification
├── visualization/
│   ├── __init__.py
│   └── architecture_diagrams.py  # Graphviz visualizations
├── docs/
│   └── math_foundation.py   # Mathematical documentation
├── train.py                 # Training script
├── training.ipynb           # Google Colab training notebook
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/orpheus016/ofdm-gan-sr.git
cd ofdm-gan-sr

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install Graphviz for visualization (optional)
# Windows: Download from https://graphviz.org/download/
# Linux: apt-get install graphviz
# Mac: brew install graphviz
```

## Quick Start

### Training

```bash
# Train with synthetic data (mini architecture)
python train.py --synthetic --epochs 500 --lr 0.0002 --batch_size 64

# Train with NON-LINEAR impairments (PA compression, IQ imbalance, phase noise)
python train.py --synthetic --epochs 500 --nonlinear --pa_saturation 0.8

# Train with skip export (for quick testing)
python train.py --synthetic --epochs 500 --lr 0.0002 --batch_size 64 --skip_export

# Train with custom config
python train.py --config config/config.yaml

# Resume training
python train.py --resume checkpoints/checkpoint_epoch_50.pt
```

### Benchmark: GAN vs Classical Equalizers

```bash
# Run comprehensive benchmark comparison
python benchmark_comparison.py --n_trials 100 --snr_min 0 --snr_max 30

# Benchmark with non-linear impairments (demonstrates GAN advantage)
python benchmark_comparison.py --nonlinear --pa_saturation 0.8

# Results saved to ./benchmark_results/
```

### Verification

```python
from models import MiniGenerator, MiniDiscriminator

# Create models
generator = MiniGenerator()
discriminator = MiniDiscriminator()

# Check parameter counts
gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())
print(f"Generator: {gen_params} params")      # ~258
print(f"Discriminator: {disc_params} params") # ~521
```

### Visualization

```python
from visualization import generate_all_diagrams

# Generate all architecture diagrams
generate_all_diagrams(output_dir='./diagrams')
```

### Export for FPGA

```python
from models import MiniGenerator
from utils.export_mini_weights import export_mini_weights

# Load trained model
generator = MiniGenerator()
generator.load_state_dict(torch.load('checkpoints/best_model.pt')['generator_state_dict'])

# Export quantized weights for RTL
export_mini_weights(generator, './export')
# Creates: weight_rom.v with Q1.7 weights
```

## Mathematical Foundation

### CWGAN-GP Loss Functions

**Critic Loss**:
$$L_D = \mathbb{E}[D(G(c), c)] - \mathbb{E}[D(x, c)] + \lambda \cdot GP$$

**Gradient Penalty**:
$$GP = \mathbb{E}[(||\nabla_{\hat{x}} D(\hat{x}, c)||_2 - 1)^2]$$

Where $\hat{x} = \epsilon \cdot x_{real} + (1-\epsilon) \cdot x_{fake}$, $\epsilon \sim U(0,1)$

**Generator Loss**:
$$L_G = -\mathbb{E}[D(G(c), c)] + \lambda_{rec} \cdot ||G(c) - x||_1$$

### Convolution Operation

For 1D convolution with kernel size $K$, stride $s$:
$$y[n] = \sum_{k=0}^{K-1} w[k] \cdot x[s \cdot n + k] + b$$

**Parameter Count**: $K \cdot C_{in} \cdot C_{out} + C_{out}$

**MACs per Layer**: $K \cdot C_{in} \cdot C_{out} \cdot L_{out}$

### Quantization

**Fixed-Point Formats**:
- **Weights**: Q1.7 format (8-bit signed, 7 fractional bits)
  - Range: [-1, +0.9921875] 
  - Resolution: 1/128 ≈ 0.0078
- **Activations**: Q8.8 format (16-bit signed, 8 fractional bits)
  - Range: [-128, +127.996]
  - Resolution: 1/256 ≈ 0.0039

**Weight Quantization**:
$$w_{q1.7} = clamp(round(w \times 128), -128, 127)$$

**Activation Quantization**:
$$a_{q8.8} = clamp(round(a \times 256), -32768, 32767)$$

### Non-Linear Impairments

**Power Amplifier (Rapp Model)**:
$$G(|x|) = \frac{|x|}{(1 + (|x|/A_{sat})^{2p})^{1/2p}}$$

Where $A_{sat}$ is saturation amplitude, $p$ is smoothness factor.

**IQ Imbalance**:
$$y = I + j \cdot g \cdot (\cos(\phi) \cdot Q + \sin(\phi) \cdot I)$$

Where $g$ is amplitude imbalance, $\phi$ is phase imbalance.

**Phase Noise (Wiener Process)**:
$$\theta[n] = \theta[n-1] + w[n], \quad w \sim \mathcal{N}(0, \sigma^2)$$

### Classical Equalizers (Baselines)

| Equalizer | Complexity | Non-Linear Handling |
|-----------|------------|---------------------|
| ZF        | O(N)       | Poor - amplifies noise |
| MMSE      | O(N)       | Poor - linear model |
| DFE       | O(L)       | Moderate - error propagation |
| LMS       | O(N)       | Poor - slow convergence |
| RLS       | O(N²)      | Poor - linear assumption |
| **GAN**   | O(N)       | **Excellent** - learned |

For detailed mathematical derivations, see `docs/math_foundation.py`.

## FPGA Deployment

### RTL Architecture

The `rtl/` folder contains complete Verilog implementation:

| Module | Description |
|--------|-------------|
| generator_mini.v | Mini U-Net generator (2→4→8→4→2) |
| discriminator_mini.v | Mini critic (4→8→16→1) |
| cwgan_gp_top.v | Top-level integration |
| conv1d_engine.v | Convolution computation engine |
| weight_rom.v | Quantized weight storage |
| activation_lrelu.v | LeakyReLU activation |
| activation_tanh.v | Tanh activation (LUT-based) |
| upsample_nn.v | Nearest-neighbor upsampling |

### Resource Estimates

| Resource | Estimate | Purpose |
|----------|----------|---------|
| LUTs | ~2,000-4,000 | Logic |
| FFs | ~1,000-2,000 | Registers |
| DSP48 | 4-8 | MAC operations |
| BRAM | ~1-2 KB | Weight/activation storage |

### Simulation

```powershell
cd rtl
# Run generator testbench
iverilog -o tb_generator_mini.vvp tb_generator_mini.v generator_mini.v ...
vvp tb_generator_mini.vvp

# Run discriminator testbench
iverilog -o tb_discriminator_mini.vvp tb_discriminator_mini.v ...
vvp tb_discriminator_mini.vvp
```

## Testing

```bash
# Test mini architecture shapes and RTL compatibility
python test_models.py

# Run OFDM utilities test
python -m utils.ofdm_utils

# Generate visualizations
python -m visualization.architecture_diagrams
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
ofdm:
  frame_length: 16        # Mini: 16 samples (matches RTL)
  n_subcarriers: 8
  modulation: "QPSK"

channel:
  snr_range: [5, 20]
  channel_type: "awgn"

generator:
  # Mini architecture: 2→4→8→4→2
  encoder_channels: [2, 4]
  bottleneck_channels: 8
  decoder_channels: [4, 2]
  kernel_size: 3

discriminator:
  # Mini architecture: 4→8→16→1
  channels: [4, 8, 16]
  kernel_size: 3

training:
  epochs: 500
  batch_size: 64
  learning_rate: 0.0002
  n_critic: 5
  gp_weight: 10.0

quantization:
  weight_bits: 8          # Q1.7 format
  activation_bits: 16     # Q8.8 format
```

## References

- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028) - Improved Training of Wasserstein GANs
- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Convolutional Networks for Biomedical Image Segmentation
- [Deep Learning OFDM](https://github.com/haoyye/OFDM_DNN) - Reference implementation
