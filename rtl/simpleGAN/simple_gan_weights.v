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
        // W^G2 from MATLAB:
        //  0.0538    0.0862
        //  0.1834    0.0319
        // -0.2259   -0.1308
        gen_w1_rom[0] = 8'sd7;    //  0.0538 * 128 =  6.9
        gen_w1_rom[1] = 8'sd11;   //  0.0862 * 128 = 11.0
        gen_w1_rom[2] = 8'sd23;   //  0.1834 * 128 = 23.5
        gen_w1_rom[3] = 8'sd4;    //  0.0319 * 128 =  4.1
        gen_w1_rom[4] = -8'sd29;  // -0.2259 * 128 = -28.9
        gen_w1_rom[5] = -8'sd17;  // -0.1308 * 128 = -16.7
    end
    
    // bg2: 3 biases (initialized to zero)
    reg signed [15:0] gen_b1_rom [0:2];
    initial begin
        gen_b1_rom[0] = 16'sd0;
        gen_b1_rom[1] = 16'sd0;
        gen_b1_rom[2] = 16'sd0;
    end
    
    // Wg3: 9x3 matrix (row-major order)
    // 27 weights total
    reg signed [7:0] gen_w2_rom [0:26];
    initial begin
        // W^G3 from MATLAB (each row is one output neuron):
        //  -0.0434   -0.0205    0.0489
        //   0.0343   -0.0124    0.1035
        //   0.3578    0.1490    0.0727
        //   0.2769    0.1409   -0.0303
        //  -0.1350    0.1417    0.0294
        //   0.3035    0.0671   -0.0787
        //   0.0725   -0.1207    0.0888
        //  -0.0063    0.0717   -0.1147
        //   0.0715    0.1630   -0.1069
        
        // Row 0 (output[0]):
        gen_w2_rom[0]  = -8'sd6;   // -0.0434 * 128 = -5.6
        gen_w2_rom[1]  = -8'sd3;   // -0.0205 * 128 = -2.6
        gen_w2_rom[2]  = 8'sd6;    //  0.0489 * 128 =  6.3
        // Row 1 (output[1]):
        gen_w2_rom[3]  = 8'sd4;    //  0.0343 * 128 =  4.4
        gen_w2_rom[4]  = -8'sd2;   // -0.0124 * 128 = -1.6
        gen_w2_rom[5]  = 8'sd13;   //  0.1035 * 128 = 13.2
        // Row 2 (output[2]):
        gen_w2_rom[6]  = 8'sd46;   //  0.3578 * 128 = 45.8
        gen_w2_rom[7]  = 8'sd19;   //  0.1490 * 128 = 19.1
        gen_w2_rom[8]  = 8'sd9;    //  0.0727 * 128 =  9.3
        // Row 3 (output[3]):
        gen_w2_rom[9]  = 8'sd35;   //  0.2769 * 128 = 35.4
        gen_w2_rom[10] = 8'sd18;   //  0.1409 * 128 = 18.0
        gen_w2_rom[11] = -8'sd4;   // -0.0303 * 128 = -3.9
        // Row 4 (output[4]):
        gen_w2_rom[12] = -8'sd17;  // -0.1350 * 128 = -17.3
        gen_w2_rom[13] = 8'sd18;   //  0.1417 * 128 = 18.1
        gen_w2_rom[14] = 8'sd4;    //  0.0294 * 128 =  3.8
        // Row 5 (output[5]):
        gen_w2_rom[15] = 8'sd39;   //  0.3035 * 128 = 38.8
        gen_w2_rom[16] = 8'sd9;    //  0.0671 * 128 =  8.6
        gen_w2_rom[17] = -8'sd10;  // -0.0787 * 128 = -10.1
        // Row 6 (output[6]):
        gen_w2_rom[18] = 8'sd9;    //  0.0725 * 128 =  9.3
        gen_w2_rom[19] = -8'sd15;  // -0.1207 * 128 = -15.4
        gen_w2_rom[20] = 8'sd11;   //  0.0888 * 128 = 11.4
        // Row 7 (output[7]):
        gen_w2_rom[21] = -8'sd1;   // -0.0063 * 128 = -0.8
        gen_w2_rom[22] = 8'sd9;    //  0.0717 * 128 =  9.2
        gen_w2_rom[23] = -8'sd15;  // -0.1147 * 128 = -14.7
        // Row 8 (output[8]):
        gen_w2_rom[24] = 8'sd9;    //  0.0715 * 128 =  9.2
        gen_w2_rom[25] = 8'sd21;   //  0.1630 * 128 = 20.9
        gen_w2_rom[26] = -8'sd14;  // -0.1069 * 128 = -13.7
    end
    
    // bg3: 9 biases (initialized to zero)
    reg signed [15:0] gen_b2_rom [0:8];
    initial begin
        gen_b2_rom[0] = 16'sd0;
        gen_b2_rom[1] = 16'sd0;
        gen_b2_rom[2] = 16'sd0;
        gen_b2_rom[3] = 16'sd0;
        gen_b2_rom[4] = 16'sd0;
        gen_b2_rom[5] = 16'sd0;
        gen_b2_rom[6] = 16'sd0;
        gen_b2_rom[7] = 16'sd0;
        gen_b2_rom[8] = 16'sd0;
    end
    
    //==========================================================================
    // Discriminator Weights (from MATLAB initialization)
    //==========================================================================
    
    // Wd2: 3x9 matrix (row-major: 27 weights)
    reg signed [7:0] disc_w1_rom [0:26];
    initial begin
        // W^D2 from MATLAB:
        // -0.0809  0.0325 -0.1712  0.0319 -0.0030  0.1093  0.0077 -0.0007  0.0371
        // -0.2944 -0.0755 -0.0102  0.0313 -0.0165  0.1109 -0.1214  0.1533 -0.0226
        //  0.1438  0.1370 -0.0241 -0.0865  0.0628 -0.0864 -0.1114 -0.0770  0.1117
        
        // Row 0 (hidden[0]):
        disc_w1_rom[0]  = -8'sd10;  // -0.0809 * 128 = -10.4
        disc_w1_rom[1]  = 8'sd4;    //  0.0325 * 128 =   4.2
        disc_w1_rom[2]  = -8'sd22;  // -0.1712 * 128 = -21.9
        disc_w1_rom[3]  = 8'sd4;    //  0.0319 * 128 =   4.1
        disc_w1_rom[4]  = 8'sd0;    // -0.0030 * 128 =  -0.4 (rounds to 0)
        disc_w1_rom[5]  = 8'sd14;   //  0.1093 * 128 =  14.0
        disc_w1_rom[6]  = 8'sd1;    //  0.0077 * 128 =   1.0
        disc_w1_rom[7]  = 8'sd0;    // -0.0007 * 128 =  -0.1 (rounds to 0)
        disc_w1_rom[8]  = 8'sd5;    //  0.0371 * 128 =   4.7
        // Row 1 (hidden[1]):
        disc_w1_rom[9]  = -8'sd38;  // -0.2944 * 128 = -37.7
        disc_w1_rom[10] = -8'sd10;  // -0.0755 * 128 =  -9.7
        disc_w1_rom[11] = -8'sd1;   // -0.0102 * 128 =  -1.3
        disc_w1_rom[12] = 8'sd4;    //  0.0313 * 128 =   4.0
        disc_w1_rom[13] = -8'sd2;   // -0.0165 * 128 =  -2.1
        disc_w1_rom[14] = 8'sd14;   //  0.1109 * 128 =  14.2
        disc_w1_rom[15] = -8'sd16;  // -0.1214 * 128 = -15.5
        disc_w1_rom[16] = 8'sd20;   //  0.1533 * 128 =  19.6
        disc_w1_rom[17] = -8'sd3;   // -0.0226 * 128 =  -2.9
        // Row 2 (hidden[2]):
        disc_w1_rom[18] = 8'sd18;   //  0.1438 * 128 =  18.4
        disc_w1_rom[19] = 8'sd18;   //  0.1370 * 128 =  17.5
        disc_w1_rom[20] = -8'sd3;   // -0.0241 * 128 =  -3.1
        disc_w1_rom[21] = -8'sd11;  // -0.0865 * 128 = -11.1
        disc_w1_rom[22] = 8'sd8;    //  0.0628 * 128 =   8.0
        disc_w1_rom[23] = -8'sd11;  // -0.0864 * 128 = -11.1
        disc_w1_rom[24] = -8'sd14;  // -0.1114 * 128 = -14.3
        disc_w1_rom[25] = -8'sd10;  // -0.0770 * 128 =  -9.9
        disc_w1_rom[26] = 8'sd14;   //  0.1117 * 128 =  14.3
    end
    
    // bd2: 3 biases (initialized to zero)
    reg signed [15:0] disc_b1_rom [0:2];
    initial begin
        disc_b1_rom[0] = 16'sd0;
        disc_b1_rom[1] = 16'sd0;
        disc_b1_rom[2] = 16'sd0;
    end
    
    // Wd3: 1x3 matrix (3 weights)
    reg signed [7:0] disc_w2_rom [0:2];
    initial begin
        // W^D3 from MATLAB: -0.1089    0.0033    0.0553
        disc_w2_rom[0] = -8'sd14;  // -0.1089 * 128 = -13.9
        disc_w2_rom[1] = 8'sd0;    //  0.0033 * 128 =   0.4 (rounds to 0)
        disc_w2_rom[2] = 8'sd7;    //  0.0553 * 128 =   7.1
    end
    
    // bd3: 1 bias (initialized to zero)
    reg signed [15:0] disc_b2;
    initial begin
        disc_b2 = 16'sd0;
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
