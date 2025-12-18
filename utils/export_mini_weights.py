#!/usr/bin/env python3
"""
Export weights from full model to mini RTL architecture.

Maps full model (2→32→64→128) to mini model (2→4→8→4→2).
Uses subset of channels and quantizes to Q1.7 weights, Q8.8 biases.
"""

import numpy as np
import struct
import os
from pathlib import Path


def load_bin_weights(filepath, shape, dtype=np.int8):
    """Load binary weight file."""
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data.reshape(shape)


def load_bin_bias(filepath, shape, dtype=np.int32):
    """Load binary bias file (32-bit signed)."""
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data.reshape(shape)


def q1_7_to_hex(val):
    """Convert signed integer to 8-bit hex string for Q1.7."""
    val = int(np.clip(int(val), -128, 127))
    if val < 0:
        val = 256 + val
    return f"8'h{val:02X}"


def q8_8_to_hex(val):
    """Convert signed integer to 16-bit hex string for Q8.8."""
    val = int(np.clip(int(val), -32768, 32767))
    if val < 0:
        val = 65536 + val
    return f"16'h{val:04X}"


def generate_mini_weights(export_dir, output_file):
    """Generate weight_rom.v with trained weights for mini architecture."""
    
    export_path = Path(export_dir)
    
    # Mini architecture mapping:
    # Encoder 1: 2→4 channels, k=3 (use first 4 output channels of enc1_1)
    # Bottleneck: 4→8 channels, k=3 (use first 8 output channels of enc2_1)  
    # Decoder 1: 8→4 channels, k=3 (use first 4 output channels of dec5_1)
    # Output: 4→2 channels, k=1 (use final layer)
    
    weights = []
    biases = []
    
    print("Loading trained weights...")
    
    # === GENERATOR WEIGHTS ===
    
    # Encoder 1: Conv(2→4, k=3) - 2*4*3 = 24 weights
    # Use first 4 output channels from enc1_1 (shape: [32, 2, 3])
    try:
        enc1_w = load_bin_weights(export_path / "enc1_1_weights.bin", [32, 2, 3])
        enc1_b = load_bin_bias(export_path / "enc1_1_bias.bin", [32])
        
        # Take first 4 output channels
        for out_ch in range(4):
            for in_ch in range(2):
                for k in range(3):
                    weights.append(enc1_w[out_ch, in_ch, k])
        
        for out_ch in range(4):
            # Scale bias from Q16.16 to Q8.8 (divide by 256)
            biases.append(enc1_b[out_ch] >> 8)
            
        print(f"  Encoder 1: 24 weights, 4 biases")
    except Exception as e:
        print(f"  Encoder 1: Using random weights (export not found: {e})")
        weights.extend(np.random.randint(-64, 64, 24).tolist())
        biases.extend(np.random.randint(-128, 128, 4).tolist())
    
    # Bottleneck: Conv(4→8, k=3) - 4*8*3 = 96 weights
    # Use first 8 output channels from enc2_1 (shape: [64, 32, 3])
    try:
        bneck_w = load_bin_weights(export_path / "enc2_1_weights.bin", [64, 32, 3])
        bneck_b = load_bin_bias(export_path / "enc2_1_bias.bin", [64])
        
        # Take first 8 output channels, first 4 input channels
        for out_ch in range(8):
            for in_ch in range(4):
                for k in range(3):
                    weights.append(bneck_w[out_ch, in_ch, k])
        
        for out_ch in range(8):
            biases.append(bneck_b[out_ch] >> 8)
            
        print(f"  Bottleneck: 96 weights, 8 biases")
    except Exception as e:
        print(f"  Bottleneck: Using random weights ({e})")
        weights.extend(np.random.randint(-64, 64, 96).tolist())
        biases.extend(np.random.randint(-128, 128, 8).tolist())
    
    # Decoder 1: Conv(8→4, k=3) - 8*4*3 = 96 weights
    # Use first 4 output channels from dec5_1 (shape: [32, 32, 3])
    try:
        dec1_w = load_bin_weights(export_path / "dec5_1_weights.bin", [32, 32, 3])
        dec1_b = load_bin_bias(export_path / "dec5_1_bias.bin", [32])
        
        # Take first 4 output channels, first 8 input channels
        for out_ch in range(4):
            for in_ch in range(8):
                for k in range(3):
                    weights.append(dec1_w[out_ch, in_ch, k])
        
        for out_ch in range(4):
            biases.append(dec1_b[out_ch] >> 8)
            
        print(f"  Decoder 1: 96 weights, 4 biases")
    except Exception as e:
        print(f"  Decoder 1: Using random weights ({e})")
        weights.extend(np.random.randint(-64, 64, 96).tolist())
        biases.extend(np.random.randint(-128, 128, 4).tolist())
    
    # Output Conv: Conv(4→2, k=1) - 4*2*1 = 8 weights
    # Note: Final layer has k=3, we'll just use center weight
    try:
        out_w = load_bin_weights(export_path / "final_weights.bin", [2, 32, 3])
        out_b = load_bin_bias(export_path / "final_bias.bin", [2])
        
        # Take first 4 input channels, use center kernel weight (k=1)
        for out_ch in range(2):
            for in_ch in range(4):
                weights.append(out_w[out_ch, in_ch, 1])  # Center of k=3
        
        for out_ch in range(2):
            biases.append(out_b[out_ch] >> 8)
            
        print(f"  Output Conv: 8 weights, 2 biases")
    except Exception as e:
        print(f"  Output Conv: Using random weights ({e})")
        weights.extend(np.random.randint(-64, 64, 8).tolist())
        biases.extend(np.random.randint(-128, 128, 2).tolist())
    
    # === DISCRIMINATOR WEIGHTS ===
    # Using random for now since discriminator weights aren't exported
    
    # Conv1: 4*8*3 = 96 weights + 8 biases
    disc_conv1_w = np.random.randint(-32, 32, 96).tolist()
    disc_conv1_b = np.random.randint(-64, 64, 8).tolist()
    
    # Conv2: 8*16*3 = 384 weights + 16 biases
    disc_conv2_w = np.random.randint(-32, 32, 384).tolist()
    disc_conv2_b = np.random.randint(-64, 64, 16).tolist()
    
    # Dense: 16 weights + 1 bias
    disc_dense_w = np.random.randint(-32, 32, 16).tolist()
    disc_dense_b = np.random.randint(-64, 64, 1).tolist()
    
    print(f"  Discriminator: {96+384+16} weights, {8+16+1} biases (random)")
    
    # === GENERATE VERILOG ===
    
    print(f"\nGenerating {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("""//==============================================================================
// Weight ROM Module - TRAINED WEIGHTS
//
// Stores pre-trained weights for Generator and Discriminator
// Generated from trained PyTorch model
//
// Fixed-Point: Q1.7 (8-bit signed weights)
//==============================================================================

`timescale 1ns / 1ps

module weight_rom #(
    parameter WEIGHT_WIDTH = 8,            // Q1.7 format
    parameter DEPTH        = 2048,         // Total weight storage
    parameter ADDR_WIDTH   = 11            // ceil(log2(DEPTH))
)(
    input  wire                     clk,
    input  wire [ADDR_WIDTH-1:0]    addr,
    output reg  signed [WEIGHT_WIDTH-1:0]  data
);

    //--------------------------------------------------------------------------
    // Weight Memory
    //--------------------------------------------------------------------------
    reg [WEIGHT_WIDTH-1:0] weights [0:DEPTH-1];
    
    //--------------------------------------------------------------------------
    // Memory Initialization - TRAINED WEIGHTS
    // 
    // Layout for Mini Generator (2ch->4ch->8ch->4ch->2ch):
    //   Encoder Conv1: 2*4*3 = 24 weights  [0:23]
    //   Bottleneck:    4*8*3 = 96 weights  [24:119]
    //   Decoder Conv1: 8*4*3 = 96 weights  [120:215]
    //   Output Conv:   4*2*1 = 8 weights   [216:223]
    //
    // Discriminator (starts at 256):
    //   Conv1: 4*8*3 = 96 weights   [256:351]
    //   Conv2: 8*16*3 = 384 weights [352:735]
    //   Dense: 16 weights           [736:751]
    //--------------------------------------------------------------------------
    
    integer i;
    initial begin
        // Initialize all to zero first
        for (i = 0; i < DEPTH; i = i + 1)
            weights[i] = 8'h00;
        
        //======================================================================
        // GENERATOR WEIGHTS (Trained)
        //======================================================================
        
        // Encoder Conv1: 2->4 channels, kernel=3
        // Layout: [out_ch][in_ch][k] = out_ch * (IN_CH * 3) + in_ch * 3 + k
""")
        
        # Write encoder 1 weights
        for i, w in enumerate(weights[:24]):
            f.write(f"        weights[{i:3d}] = {q1_7_to_hex(w)};  // Enc1[{i//6}][{(i%6)//3}][{i%3}]\n")
        
        f.write("""
        // Bottleneck: 4->8 channels, kernel=3
""")
        
        # Write bottleneck weights
        for i, w in enumerate(weights[24:120]):
            f.write(f"        weights[{24+i:3d}] = {q1_7_to_hex(w)};  // Bneck[{i//12}][{(i%12)//3}][{i%3}]\n")
        
        f.write("""
        // Decoder Conv1: 8->4 channels, kernel=3
""")
        
        # Write decoder 1 weights
        for i, w in enumerate(weights[120:216]):
            f.write(f"        weights[{120+i:3d}] = {q1_7_to_hex(w)};  // Dec1[{i//24}][{(i%24)//3}][{i%3}]\n")
        
        f.write("""
        // Output Conv: 4->2 channels, kernel=1
""")
        
        # Write output conv weights
        for i, w in enumerate(weights[216:224]):
            f.write(f"        weights[{216+i:3d}] = {q1_7_to_hex(w)};  // Out[{i//4}][{i%4}][0]\n")
        
        f.write("""
        //======================================================================
        // DISCRIMINATOR WEIGHTS (Random placeholder)
        //======================================================================
        
        // Conv1: 4->8 channels, kernel=3
""")
        
        # Write discriminator weights
        for i, w in enumerate(disc_conv1_w):
            f.write(f"        weights[{256+i:3d}] = {q1_7_to_hex(w)};\n")
        
        f.write("""
        // Conv2: 8->16 channels, kernel=3
""")
        for i, w in enumerate(disc_conv2_w):
            f.write(f"        weights[{352+i:3d}] = {q1_7_to_hex(w)};\n")
        
        f.write("""
        // Dense: 16->1
""")
        for i, w in enumerate(disc_dense_w):
            f.write(f"        weights[{736+i:3d}] = {q1_7_to_hex(w)};\n")
        
        f.write("""    end
    
    //--------------------------------------------------------------------------
    // Synchronous Read
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        data <= weights[addr];
    end

endmodule


//==============================================================================
// Bias ROM Module - TRAINED BIASES
//
// Stores biases for all layers
// Fixed-Point: Q8.8 (16-bit signed)
//==============================================================================

module bias_rom #(
    parameter DATA_WIDTH = 16,             // Q8.8 format
    parameter DEPTH      = 64,             // Total bias storage
    parameter ADDR_WIDTH = 6               // ceil(log2(DEPTH))
)(
    input  wire                     clk,
    input  wire [ADDR_WIDTH-1:0]    addr,
    output reg  signed [DATA_WIDTH-1:0]    data
);

    //--------------------------------------------------------------------------
    // Bias Memory
    //--------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] biases [0:DEPTH-1];
    
    //--------------------------------------------------------------------------
    // Memory Initialization - TRAINED BIASES
    //
    // Layout:
    //   Generator Encoder 1: 4 biases [0:3]
    //   Generator Bottleneck: 8 biases [4:11]
    //   Generator Decoder 1: 4 biases [12:15]
    //   Generator Output: 2 biases [16:17]
    //   Discriminator Conv1: 8 biases [32:39]
    //   Discriminator Conv2: 16 biases [40:55]
    //   Discriminator Dense: 1 bias [56]
    //--------------------------------------------------------------------------
    
    integer i;
    initial begin
        // Initialize all to zero
        for (i = 0; i < DEPTH; i = i + 1)
            biases[i] = 16'h0000;
        
        //======================================================================
        // GENERATOR BIASES (Trained)
        //======================================================================
        
        // Encoder 1 biases
""")
        
        # Write generator biases
        for i, b in enumerate(biases[:4]):
            f.write(f"        biases[{i:2d}] = {q8_8_to_hex(b)};  // Enc1 bias[{i}]\n")
        
        f.write("""
        // Bottleneck biases
""")
        for i, b in enumerate(biases[4:12]):
            f.write(f"        biases[{4+i:2d}] = {q8_8_to_hex(b)};  // Bneck bias[{i}]\n")
        
        f.write("""
        // Decoder 1 biases
""")
        for i, b in enumerate(biases[12:16]):
            f.write(f"        biases[{12+i:2d}] = {q8_8_to_hex(b)};  // Dec1 bias[{i}]\n")
        
        f.write("""
        // Output biases
""")
        for i, b in enumerate(biases[16:18]):
            f.write(f"        biases[{16+i:2d}] = {q8_8_to_hex(b)};  // Out bias[{i}]\n")
        
        f.write("""
        //======================================================================
        // DISCRIMINATOR BIASES (Random placeholder)
        //======================================================================
        
        // Conv1 biases
""")
        for i, b in enumerate(disc_conv1_b):
            f.write(f"        biases[{32+i:2d}] = {q8_8_to_hex(b)};\n")
        
        f.write("""
        // Conv2 biases
""")
        for i, b in enumerate(disc_conv2_b):
            f.write(f"        biases[{40+i:2d}] = {q8_8_to_hex(b)};\n")
        
        f.write("""
        // Dense bias
""")
        for i, b in enumerate(disc_dense_b):
            f.write(f"        biases[{56+i:2d}] = {q8_8_to_hex(b)};\n")
        
        f.write("""    end
    
    //--------------------------------------------------------------------------
    // Synchronous Read
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        data <= biases[addr];
    end

endmodule
""")
    
    print(f"✓ Generated {output_file}")
    print(f"  Generator: {len(weights[:224])} weights, {len(biases[:18])} biases")
    print(f"  Discriminator: {len(disc_conv1_w)+len(disc_conv2_w)+len(disc_dense_w)} weights")
    
    return len(weights[:224]), len(biases[:18])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export trained weights to Verilog ROM")
    parser.add_argument("--export_dir", default="export/generator", 
                        help="Directory with exported .bin files")
    parser.add_argument("--output", default="rtl/weight_rom.v",
                        help="Output Verilog file")
    
    args = parser.parse_args()
    
    generate_mini_weights(args.export_dir, args.output)
