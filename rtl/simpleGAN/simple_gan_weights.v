//==============================================================================
// Simple GAN - Weight ROM Module
//
// Contains initial weights from MATLAB training initialization:
//
// Generator Weights (from MATLAB):
//   Wg2 (3x2): Hidden layer weights
//   bg2 (3):   Hidden layer biases
//   Wg3 (9x3): Output layer weights
//   bg3 (9):   Output layer biases
//
// Discriminator Weights (from MATLAB):
//   Wd2 (3x9): Hidden layer weights
//   bd2 (3):   Hidden layer biases
//   Wd3 (1x3): Output layer weights
//   bd3 (1):   Output layer bias
//
// Fixed-Point Format:
//   - Weights: Q1.7 (8-bit signed, range [-1, +0.992])
//   - Biases: Q8.8 (16-bit signed)
//==============================================================================

`timescale 1ns / 1ps

module simple_gan_weights (
    input  wire                clk,
    
    //----------------------------------------------------------------------
    // Generator Weight Interface
    //----------------------------------------------------------------------
    // Layer 1: 2 -> 3 (6 weights, 3 biases)
    input  wire [3:0]          g_w1_addr,    // Weight address (0-5)
    output reg signed [7:0]    g_w1_data,    // Q1.7 weight
    input  wire [1:0]          g_b1_addr,    // Bias address (0-2)
    output reg signed [15:0]   g_b1_data,    // Q8.8 bias
    
    // Layer 2: 3 -> 9 (27 weights, 9 biases)
    input  wire [4:0]          g_w2_addr,    // Weight address (0-26)
    output reg signed [7:0]    g_w2_data,    // Q1.7 weight
    input  wire [3:0]          g_b2_addr,    // Bias address (0-8)
    output reg signed [15:0]   g_b2_data,    // Q8.8 bias
    
    //----------------------------------------------------------------------
    // Discriminator Weight Interface
    //----------------------------------------------------------------------
    // Layer 1: 9 -> 3 (27 weights, 3 biases)
    input  wire [4:0]          d_w1_addr,    // Weight address (0-26)
    output reg signed [7:0]    d_w1_data,    // Q1.7 weight
    input  wire [1:0]          d_b1_addr,    // Bias address (0-2)
    output reg signed [15:0]   d_b1_data,    // Q8.8 bias
    
    // Layer 2: 3 -> 1 (3 weights, 1 bias)
    input  wire [1:0]          d_w2_addr,    // Weight address (0-2)
    output reg signed [7:0]    d_w2_data,    // Q1.7 weight
    output reg signed [15:0]   d_b2_data     // Q8.8 bias (single value)
);

    //==========================================================================
    // Generator Weights (from MATLAB initialization)
    //==========================================================================
    
    // Wg2: 3x2 matrix (row-major: [0,0], [0,1], [1,0], [1,1], [2,0], [2,1])
    // Values from MATLAB, quantized to Q1.7 (multiply by 128)
    reg signed [7:0] gen_w1_rom [0:5];
    initial begin
        // W^G2 from MATLAB epooch 30.000:
        gen_w1_rom[0] = 8'sd4;    
        gen_w1_rom[1] = 8'sd8;   
        gen_w1_rom[2] = 8'sd22;   
        gen_w1_rom[3] = 8'sd4;    
        gen_w1_rom[4] = -8'sd28;  
        gen_w1_rom[5] = -8'sd16;  
    end
    
    // bg2: 3 biases (from MATLAB model)
    reg signed [15:0] gen_b1_rom [0:2];
    initial begin
        gen_b1_rom[0] = 16'sd113;
        gen_b1_rom[1] = 16'sd51;
        gen_b1_rom[2] = -16'sd28;
    end
    
    // Wg3: 9x3 matrix (row-major order)
    // 27 weights total
    reg signed [7:0] gen_w2_rom [0:26];
    initial begin
        // W^G3 from MATLAB (each row is one output neuron):
        
        // Row 0 (output[0]):
        gen_w2_rom[0]  = -8'sd6;   
        gen_w2_rom[1]  = -8'sd3;   
        gen_w2_rom[2]  = 8'sd6;    
        // Row 1 (output[1]):
        gen_w2_rom[3]  = 8'sd22;    
        gen_w2_rom[4]  = 8'sd6;   
        gen_w2_rom[5]  = 8'sd9;   
        // Row 2 (output[2]):
        gen_w2_rom[6]  = 8'sd44;   
        gen_w2_rom[7]  = 8'sd18;   
        gen_w2_rom[8]  = 8'sd10;    
        // Row 3 (output[3]):
        gen_w2_rom[9]  = 8'sd50;   
        gen_w2_rom[10] = 8'sd24;   
        gen_w2_rom[11] = -8'sd7;   
        // Row 4 (output[4]):
        gen_w2_rom[12] = -8'sd18;  
        gen_w2_rom[13] = 8'sd18;   
        gen_w2_rom[14] = 8'sd4;    
        // Row 5 (output[5]):
        gen_w2_rom[15] = 8'sd54;   
        gen_w2_rom[16] = 8'sd15;   
        gen_w2_rom[17] = -8'sd13;  
        // Row 6 (output[6]):
        gen_w2_rom[18] = 8'sd10;   
        gen_w2_rom[19] = -8'sd15;  
        gen_w2_rom[20] = 8'sd11;   
        // Row 7 (output[7]):
        gen_w2_rom[21] = 8'sd17;   
        gen_w2_rom[22] = 8'sd17;   
        gen_w2_rom[23] = -8'sd18;  
        // Row 8 (output[8]):
        gen_w2_rom[24] = 8'sd9;    
        gen_w2_rom[25] = 8'sd21;   
        gen_w2_rom[26] = -8'sd14;  
    end
    
    // bg3: 9 biases (from MATLAB model)
    reg signed [15:0] gen_b2_rom [0:8];
    initial begin
        gen_b2_rom[0] = -16'sd30;
        gen_b2_rom[1] = 16'sd146;
        gen_b2_rom[2] = 16'sd5;
        gen_b2_rom[3] = 16'sd143;
        gen_b2_rom[4] = -16'sd16;
        gen_b2_rom[5] = 16'sd150;
        gen_b2_rom[6] = -16'sd6;
        gen_b2_rom[7] = 16'sd178;
        gen_b2_rom[8] = -16'sd1;
    end
    
    //==========================================================================
    // Discriminator Weights (from MATLAB initialization)
    //==========================================================================
    
    // Wd2: 3x9 matrix (row-major: 27 weights)
    reg signed [7:0] disc_w1_rom [0:26];
    initial begin
        // W^D2 from MATLAB epoch 30.000:
        
        // Row 0 (hidden[0]):
        disc_w1_rom[0]  = -8'sd19;  
        disc_w1_rom[1]  = -8'sd38;    
        disc_w1_rom[2]  = -8'sd6;  
        disc_w1_rom[3]  = -8'sd29;    
        disc_w1_rom[4]  = -8'sd10;    
        disc_w1_rom[5]  = -8'sd17;   
        disc_w1_rom[6]  = 8'sd1;    
        disc_w1_rom[7]  = -8'sd31;   
        disc_w1_rom[8]  = 8'sd13;    
        // Row 1 (hidden[1]):
        disc_w1_rom[9]  = 8'sd9;  
        disc_w1_rom[10] = 8'sd59;  
        disc_w1_rom[11] = -8'sd20; 
        disc_w1_rom[12] = 8'sd50;  
        disc_w1_rom[13] = -8'sd1;  
        disc_w1_rom[14] = 8'sd54;  
        disc_w1_rom[15] = 8'sd8;  
        disc_w1_rom[16] = 8'sd59; 
        disc_w1_rom[17] = 8'sd0;  
        // Row 2 (hidden[2]):
        disc_w1_rom[18] = 8'sd18; 
        disc_w1_rom[19] = 8'sd20; 
        disc_w1_rom[20] = -8'sd3; 
        disc_w1_rom[21] = -8'sd9; 
        disc_w1_rom[22] = 8'sd8;  
        disc_w1_rom[23] = -8'sd9; 
        disc_w1_rom[24] = -8'sd14;
        disc_w1_rom[25] = -8'sd8; 
        disc_w1_rom[26] = 8'sd14; 
    end
    
    // bd2: 3 biases (from MATLAB model)
    reg signed [15:0] disc_b1_rom [0:2];
    initial begin
        disc_b1_rom[0] = 16'sd82;
        disc_b1_rom[1] = -16'sd306;
        disc_b1_rom[2] = 16'sd0;
    end
    
    // Wd3: 1x3 matrix (3 weights)
    reg signed [7:0] disc_w2_rom [0:2];
    initial begin
        // W^D3 from MATLAB epoch 30.000:
        disc_w2_rom[0] = -8'sd85;  
        disc_w2_rom[1] = 8'sd127;  
        disc_w2_rom[2] = 8'sd2;    
    end
    
    // bd3: 1 bias (from MATLAB model)
    reg signed [15:0] disc_b2;
    initial begin
        disc_b2 = -16'sd120;
    end
    
    //==========================================================================
    // ROM Read Logic (synchronous)
    //==========================================================================
    
    always @(posedge clk) begin
        // Generator Layer 1
        g_w1_data <= gen_w1_rom[g_w1_addr];
        g_b1_data <= gen_b1_rom[g_b1_addr];
        
        // Generator Layer 2
        g_w2_data <= gen_w2_rom[g_w2_addr];
        g_b2_data <= gen_b2_rom[g_b2_addr];
        
        // Discriminator Layer 1
        d_w1_data <= disc_w1_rom[d_w1_addr];
        d_b1_data <= disc_b1_rom[d_b1_addr];
        
        // Discriminator Layer 2
        d_w2_data <= disc_w2_rom[d_w2_addr];
        d_b2_data <= disc_b2;  // Single bias, no address
    end

endmodule
