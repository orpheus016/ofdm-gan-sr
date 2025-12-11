# RTL Implementation of CWGAN-GP for OFDM

## Overview

This folder contains Verilog RTL implementation of a simplified but representative CWGAN-GP architecture for OFDM signal reconstruction.

## Architecture Specifications

### Simplified Design (for demonstration)
- **Frame Length**: 16 samples (scaled down from 1024)
- **Input Channels**: 2 (I/Q)
- **Data Width**: 16-bit signed fixed-point (Q8.8)
- **Weight Width**: 8-bit signed fixed-point (Q1.7)
- **Accumulator Width**: 32-bit

### Generator (Mini U-Net)
```
Input [2×16] → Enc1 [4×8] → Bottleneck [8×4] → Dec1 [4×8] → Output [2×16]
              ↓                                    ↑
              └──────── Skip Connection ───────────┘
```

### Discriminator (Mini Critic)
```
[4×16] → Conv1 [8×8] → Conv2 [16×4] → Sum Pool → Dense → Score [1]
```

## Module Hierarchy

```
cwgan_gp_top
├── generator
│   ├── conv1d_encoder (strided conv, downsample)
│   ├── conv1d_block (bottleneck)
│   ├── upsample_nn (nearest neighbor 2x)
│   ├── conv1d_decoder (with skip add)
│   └── activation_lrelu (LeakyReLU LUT)
├── discriminator
│   ├── conv1d_encoder (strided conv)
│   ├── sum_pool
│   └── dense_layer
├── weight_rom (stores pre-trained weights)
└── control_fsm (layer sequencing)
```

## File Descriptions

### Core Modules
| File | Description |
|------|-------------|
| `conv1d_engine.v` | Basic 1D convolution unit with state machine |
| `conv1d_pipelined.v` | High-performance pipelined parallel MAC convolution |
| `activation_lrelu.v` | LeakyReLU activation (α=0.2) |
| `activation_tanh.v` | Tanh activation with LUT |
| `upsample_nn.v` | Nearest neighbor 2x upsampler |
| `sum_pool.v` | Global sum/average pooling |
| `weight_rom.v` | Weight and bias memory |

### Top-Level Modules
| File | Description |
|------|-------------|
| `generator_mini.v` | Mini U-Net generator (2→4→8→4→2 channels) |
| `discriminator_mini.v` | Mini CNN discriminator/critic |
| `cwgan_gp_top.v` | Top-level with inference/training modes |

### Testbenches
| File | Description |
|------|-------------|
| `tb_cwgan_gp.v` | Comprehensive system testbench |
| `tb_generator_mini.v` | Generator-focused testbench |

### Build Scripts
| File | Description |
|------|-------------|
| `Makefile` | GNU Make build script (Linux/Mac) |
| `simulate.ps1` | PowerShell simulation script (Windows) |

## Fixed-Point Format

### Data (Activations): Q8.8 (16-bit)
- Range: [-128.0, 127.99609375]
- Resolution: 1/256 ≈ 0.00390625

### Weights: Q1.7 (8-bit)
- Range: [-1.0, 0.9921875]
- Resolution: 1/128 ≈ 0.0078125

### Accumulator: Q16.16 (32-bit)
- Used during MAC operations
- Requantized to Q8.8 after each layer

## Simulation

### Windows (PowerShell)
```powershell
# Simulate generator only
.\simulate.ps1 -Target gen

# Simulate full system
.\simulate.ps1 -Target full

# Run all simulations
.\simulate.ps1 -Target all

# Clean generated files
.\simulate.ps1 -Target clean
```

### Linux/Mac (Make)
```bash
# Simulate generator only
make sim_gen

# Simulate full system
make sim_full

# View waveforms
make wave_gen
make wave_full

# Lint check (requires Verilator)
make lint

# Clean
make clean
```

### Manual Compilation (Icarus Verilog)
```bash
# Compile
iverilog -g2012 -o tb_cwgan_gp.vvp \
    conv1d_engine.v activation_lrelu.v activation_tanh.v \
    upsample_nn.v sum_pool.v weight_rom.v \
    generator_mini.v discriminator_mini.v cwgan_gp_top.v \
    tb_cwgan_gp.v

# Run
vvp tb_cwgan_gp.vvp

# View waveforms
gtkwave tb_cwgan_gp.vcd
```

### ModelSim
```tcl
vlog *.v
vsim -c tb_cwgan_gp -do "run -all"
```

## Resource Estimates (Scaled to Full Design)

For full 1024-sample generator:
- DSP48: ~100-200 slices
- BRAM: ~160 KB for skip buffers
- LUTs: ~50K
- FFs: ~30K
- Clock: 200 MHz target

## Architecture Strategies

### Pipeline Architecture
The `conv1d_pipelined.v` module demonstrates a fully pipelined design:
```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Stage 0 │ → │ Stage 1 │ → │ Stage 2 │ → │ Stage 3 │ → │ Stage 4 │ → │ Stage 5 │
│ DataFetch│   │WeightFetch│  │Multiply │   │AccumTree│   │Bias+Act │   │ Output  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```
- **Throughput**: 1 MAC operation per cycle per pipeline
- **Latency**: 6 cycles

### Parallel Architecture
Multiple MAC units process simultaneously:
```
    ┌──────┐
    │ MAC0 │──┐
    └──────┘  │
    ┌──────┐  │   ┌───────────┐   ┌────────┐
    │ MAC1 │──┼──→│ Adder Tree│──→│ Output │
    └──────┘  │   └───────────┘   └────────┘
    ┌──────┐  │
    │ MAC2 │──┘
    └──────┘
    ┌──────┐
    │ MAC3 │──┘
    └──────┘
```
- **Throughput**: NUM_MACS operations per cycle
- **Resource**: Linear increase in DSPs

### Memory Efficiency
- **Double Buffering**: Overlap loading of next frame with current processing
- **Skip Connection Buffer**: Stores encoder outputs for residual addition
- **Weight Caching**: Reuse weights across spatial positions

## Scaling to Full Architecture

To scale from mini (16 samples) to full (1024 samples):

| Parameter | Mini | Full | Scale |
|-----------|------|------|-------|
| Frame Length | 16 | 1024 | 64x |
| Encoder Levels | 2 | 5 | 2.5x |
| Max Channels | 8 | 512 | 64x |
| Weights | ~500 | ~5.5M | 11000x |
| Skip Buffer | 128B | 160KB | 1280x |

### Recommendations for Full Design
1. **Use external DDR** for weight storage
2. **Implement layer-by-layer processing** to reuse compute units
3. **Add DMA controller** for efficient memory transfers
4. **Consider HBM** for high-bandwidth weight access

