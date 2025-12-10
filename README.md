# CWGAN-GP for OFDM Signal Reconstruction

A Conditional Wasserstein GAN with Gradient Penalty for OFDM signal reconstruction, designed for FPGA deployment.

## Overview

This project implements a 1D U-Net based GAN for enhancing noisy OFDM I/Q signals for image file transmission. The generator is designed to be deployed on FPGA for real-time signal enhancement.

### Key Features

- **1D U-Net Generator**: 5-level encoder-decoder with additive skip connections
- **CWGAN-GP Training**: Stable training with gradient penalty
- **FPGA-Ready**: INT8 weight quantization, ~5.5M parameters, ~365M MACs/frame
- **OFDM Simulation**: Complete QAM modulation, OFDM encoding, and channel simulation
- **Comprehensive Documentation**: Full mathematical foundation for every operation

## Architecture

```
Generator (1D U-Net):
Input [2×1024] → Encoder → Bottleneck → Decoder → Output [2×1024]
     ↓              ↓                      ↑           ↑
   I/Q signal    Skip connections ─────────┘      Enhanced I/Q

Discriminator (Conditional Critic):
[Candidate + Condition] → Conv Stack → Global Sum Pool → Score
        [4×1024]                                          [1]
```

### Generator Specifications

| Level | Encoder Channels | Decoder Channels | Length |
|-------|------------------|------------------|--------|
| 0     | 2 → 32          | 32 → 2           | 1024   |
| 1     | 32 → 64         | 64 → 32          | 512    |
| 2     | 64 → 128        | 128 → 64         | 256    |
| 3     | 128 → 256       | 256 → 128        | 128    |
| 4     | 256 → 512       | 512 → 256        | 64     |
| BN    | 512 → 512       | -                | 32     |

**Total Parameters**: ~5.5M  
**Total MACs/Frame**: ~365M  
**Skip Buffer Memory**: ~160KB

## Project Structure

```
ofdm-gan-sr/
├── config/
│   └── config.yaml          # Configuration file
├── models/
│   ├── __init__.py
│   ├── generator.py         # 1D U-Net Generator
│   └── discriminator.py     # Conditional Critic
├── utils/
│   ├── __init__.py
│   ├── ofdm_utils.py        # OFDM/QAM/Channel utilities
│   ├── dataset.py           # Data loading and generation
│   └── quantization.py      # FPGA quantization utilities
├── proof/
│   ├── __init__.py
│   └── verification.py      # Architecture verification
├── visualization/
│   ├── __init__.py
│   └── architecture_diagrams.py  # Graphviz visualizations
├── docs/
│   └── math_foundation.py   # Mathematical documentation
├── train.py                 # Training script
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
# Train with synthetic data
python train.py --synthetic --epochs 100

# Train with custom images
python train.py --config config/config.yaml

# Resume training
python train.py --resume checkpoints/checkpoint_epoch_50.pt
```

### Verification

```python
from models import UNetGenerator, Discriminator
from proof import run_full_verification

# Create models
generator = UNetGenerator()
discriminator = Discriminator()

# Run verification
results = run_full_verification(generator, discriminator)
```

### Visualization

```python
from visualization import generate_all_diagrams

# Generate all architecture diagrams
generate_all_diagrams(output_dir='./diagrams')
```

### Export for FPGA

```python
from models import UNetGenerator
from utils import export_weights_fpga, QuantizationConfig

# Load trained model
generator = UNetGenerator()
generator.load_state_dict(torch.load('checkpoints/best_model.pt')['generator_state_dict'])

# Export quantized weights
config = QuantizationConfig(weight_bits=8, activation_bits=16)
export_weights_fpga(generator, './export', config)
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

**Symmetric Quantization**:
$$scale = \frac{\max|x|}{2^{b-1} - 1}$$
$$x_{quant} = round(clamp(x / scale, -2^{b-1}, 2^{b-1} - 1))$$

For detailed mathematical derivations, see `docs/math_foundation.py`.

## FPGA Deployment

### Resource Estimates (Xilinx ZCU104)

| Resource | Usage | Purpose |
|----------|-------|---------|
| DSP48 Slices | ~100-200 | MAC operations |
| BRAM | ~160 KB | Skip buffers |
| DDR Bandwidth | ~165 MB/s @30fps | Weight streaming |
| Clock Frequency | 200 MHz | Target |

### Throughput

- **55 DSPs @200MHz**: ~30 fps
- **110 DSPs @200MHz**: ~60 fps

### Exported Files

After `export_weights_fpga()`:
```
export/
├── generator/
│   ├── enc1_1_weights.bin     # INT8 weights
│   ├── enc1_1_scale.bin       # Per-channel scales
│   ├── enc1_1_bias.bin        # FP32 biases
│   ├── ...
│   └── metadata.json          # Layer specifications + CRCs
```

## Testing

```bash
# Run architecture verification
python -m proof.verification

# Verify model shapes
python -m models.generator
python -m models.discriminator

# Test OFDM utilities
python -m utils.ofdm_utils

# Generate visualizations
python -m visualization.architecture_diagrams
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
ofdm:
  frame_length: 1024
  modulation: "QAM16"

channel:
  snr_range: [0, 30]
  channel_type: "awgn"

training:
  epochs: 200
  batch_size: 32
  n_critic: 5
  gp_weight: 10.0
  loss:
    reconstruction_weight: 100.0

quantization:
  weight_bits: 8
  activation_bits: 16
```

## References

- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028) - Improved Training of Wasserstein GANs
- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Convolutional Networks for Biomedical Image Segmentation
- [Deep Learning OFDM](https://github.com/haoyye/OFDM_DNN) - Reference implementation
