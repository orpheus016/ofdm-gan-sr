# Simple GAN RTL Implementation

A minimal GAN (Generative Adversarial Network) implemented in Verilog, designed to generate 3×3 binary patterns (circle and cross shapes).

## Architecture

Based on the MATLAB reference implementation:

### Generator
```
Latent Vector (2) → Hidden Layer (3, tanh) → Output (9, tanh)
```
- **Input**: 2-dimensional latent vector in range [-1, +1]
- **Hidden**: 3 neurons with tanh activation
- **Output**: 9 values (3×3 image) with tanh activation

### Discriminator
```
Image (9) → Hidden Layer (3, tanh) → Output (1, sigmoid)
```
- **Input**: 9-element flattened 3×3 image
- **Hidden**: 3 neurons with tanh activation
- **Output**: Single probability score [0, 1]

## Fixed-Point Format

| Data Type | Format | Range |
|-----------|--------|-------|
| Activations | Q8.8 (16-bit signed) | [-128, +127.996] |
| Weights | Q1.7 (8-bit signed) | [-1, +0.992] |
| Sigmoid Output | Q8.8 | [0, +1.0] |

## Files

| File | Description |
|------|-------------|
| `activation_sigmoid.v` | Sigmoid activation using 256-entry LUT |
| `simple_generator.v` | Generator with 2 dense layers + tanh |
| `simple_discriminator.v` | Discriminator with tanh + sigmoid |
| `simple_gan_weights.v` | Weight ROM with initial random weights |
| `simple_gan_top.v` | Top-level module with mode control |
| `tb_simple_gan.v` | Comprehensive testbench |
| `dense_layer.v` | Reusable dense layer module |

## Training Data (from MATLAB)

### Circle Pattern
```
 1  -1   1
-1   1  -1
 1  -1   1
```

### Cross Pattern
```
-1   1  -1
 1   1   1
-1   1  -1
```

## Usage

### Simulation with Icarus Verilog

```bash
# Using Makefile
make sim

# Using PowerShell
.\simulate.ps1

# View waveforms
make view
# or
.\simulate.ps1 -View
```

### Operating Modes

The `simple_gan_top` module supports three modes:

| Mode | Operation |
|------|-----------|
| `2'b00` | Generate only: latent → image |
| `2'b01` | Discriminate only: image → score |
| `2'b10` | Full GAN: latent → image → score |

### Interface Signals

```verilog
// Control
input  [1:0]  mode,           // Operating mode
input         start,          // Start processing
output        busy,           // Processing in progress
output        done,           // Processing complete

// Generator Input
input  [15:0] latent_in[0:1], // Q8.8 latent vector

// Discriminator Input  
input  [15:0] image_in[0:8],  // Q8.8 flattened image

// Outputs
output [15:0] gen_image[0:8], // Q8.8 generated image
output        gen_valid,      // Generator output valid
output [15:0] disc_score,     // Q8.8 discriminator score
output        disc_valid      // Discriminator output valid
```

## Resource Estimates

| Resource | Count |
|----------|-------|
| Weights (Generator) | 33 (6 + 27) |
| Weights (Discriminator) | 30 (27 + 3) |
| Biases (Generator) | 12 (3 + 9) |
| Biases (Discriminator) | 4 (3 + 1) |
| LUT entries (tanh) | 256 |
| LUT entries (sigmoid) | 256 |

## MATLAB Reference

The architecture matches this MATLAB implementation:

```matlab
% Generator
Wg2 = 0.1 * randn(hidden_dim, latent_dim);  % 3x2
bg2 = zeros(hidden_dim, 1);
Wg3 = 0.1 * randn(output_dim, hidden_dim);  % 9x3
bg3 = zeros(output_dim, 1);

% Discriminator
Wd2 = 0.1 * randn(hidden_dim, input_dim);   % 3x9
bd2 = zeros(hidden_dim, 1);
Wd3 = 0.1 * randn(1, hidden_dim);           % 1x3
bd3 = zeros(1, 1);
```

## Dependencies

- Icarus Verilog (`iverilog`) for simulation
- GTKWave (optional) for waveform viewing
- Uses `activation_tanh.v` from parent `rtl/` directory
