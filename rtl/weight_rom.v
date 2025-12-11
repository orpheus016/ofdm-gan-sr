//==============================================================================
// Weight ROM Module
//
// Stores pre-trained weights for Generator and Discriminator
// Organized by layer with base address offsets
//
// Fixed-Point: Q1.7 (8-bit signed weights)
//==============================================================================

module weight_rom #(
    parameter WEIGHT_WIDTH = 8,            // Q1.7 format
    parameter DEPTH        = 2048,         // Total weight storage
    parameter ADDR_WIDTH   = 11            // ceil(log2(DEPTH))
)(
    input  wire                     clk,
    input  wire [ADDR_WIDTH-1:0]    addr,
    output reg  [WEIGHT_WIDTH-1:0]  data
);

    //--------------------------------------------------------------------------
    // Weight Memory
    //--------------------------------------------------------------------------
    reg [WEIGHT_WIDTH-1:0] weights [0:DEPTH-1];
    
    //--------------------------------------------------------------------------
    // Memory Initialization
    // 
    // Layout for Mini Generator (2ch->4ch->8ch->4ch->2ch):
    //   Encoder Conv1: 2*4*3 = 24 weights  [0:23]
    //   Bottleneck:    4*8*3 = 96 weights  [24:119]
    //   Decoder Conv1: 8*4*3 = 96 weights  [120:215]
    //   Output Conv:   4*2*1 = 8 weights   [216:223]
    //   
    // Layout for Mini Discriminator (4ch->8ch->16ch):
    //   Conv1: 4*8*3  = 96 weights  [256:351]
    //   Conv2: 8*16*3 = 384 weights [352:735]
    //   Dense: 16*1   = 16 weights  [736:751]
    //--------------------------------------------------------------------------
    
    initial begin
        // Initialize with pseudo-random values in [0, 1) range
        // In Q1.7: 0x00 to 0x7F represents 0.0 to 0.992
        // These would be replaced with trained weights
        
        //----------------------------------------------------------------------
        // Generator Weights (addresses 0-255)
        //----------------------------------------------------------------------
        
        // Encoder Conv1: in_ch=2, out_ch=4, kernel=3
        // Weight order: [out_ch][in_ch][kernel]
        weights[0]  = 8'h20;  // 0.25
        weights[1]  = 8'h40;  // 0.50
        weights[2]  = 8'h20;  // 0.25
        weights[3]  = 8'h18;  // 0.19
        weights[4]  = 8'h48;  // 0.56
        weights[5]  = 8'h18;  // 0.19
        weights[6]  = 8'h30;  // 0.38
        weights[7]  = 8'h38;  // 0.44
        weights[8]  = 8'h30;  // 0.38
        weights[9]  = 8'h28;  // 0.31
        weights[10] = 8'h50;  // 0.63
        weights[11] = 8'h28;  // 0.31
        weights[12] = 8'h10;  // 0.13
        weights[13] = 8'h60;  // 0.75
        weights[14] = 8'h10;  // 0.13
        weights[15] = 8'h38;  // 0.44
        weights[16] = 8'h30;  // 0.38
        weights[17] = 8'h38;  // 0.44
        weights[18] = 8'h20;  // 0.25
        weights[19] = 8'h58;  // 0.69
        weights[20] = 8'h20;  // 0.25
        weights[21] = 8'h48;  // 0.56
        weights[22] = 8'h28;  // 0.31
        weights[23] = 8'h48;  // 0.56
        
        // Bottleneck: in_ch=4, out_ch=8, kernel=3
        // 96 weights [24:119]
        weights[24]  = 8'h35;
        weights[25]  = 8'h45;
        weights[26]  = 8'h35;
        weights[27]  = 8'h2A;
        weights[28]  = 8'h4A;
        weights[29]  = 8'h2A;
        weights[30]  = 8'h3F;
        weights[31]  = 8'h3F;
        // ... continue pattern (abbreviated for readability)
        // Fill remaining bottleneck weights
        
        // Decoder Conv1: in_ch=8, out_ch=4, kernel=3
        // 96 weights [120:215]
        weights[120] = 8'h42;
        weights[121] = 8'h3E;
        weights[122] = 8'h42;
        weights[123] = 8'h36;
        weights[124] = 8'h4C;
        weights[125] = 8'h36;
        // ... continue pattern
        
        // Output Conv: in_ch=4, out_ch=2, kernel=1
        // 8 weights [216:223]
        weights[216] = 8'h50;
        weights[217] = 8'h40;
        weights[218] = 8'h48;
        weights[219] = 8'h38;
        weights[220] = 8'h58;
        weights[221] = 8'h30;
        weights[222] = 8'h44;
        weights[223] = 8'h3C;
        
        //----------------------------------------------------------------------
        // Discriminator Weights (addresses 256-1023)
        //----------------------------------------------------------------------
        
        // Conv1: in_ch=4, out_ch=8, kernel=3, stride=2
        // 96 weights [256:351]
        weights[256] = 8'h3A;
        weights[257] = 8'h46;
        weights[258] = 8'h3A;
        weights[259] = 8'h2E;
        weights[260] = 8'h52;
        weights[261] = 8'h2E;
        // ... continue pattern
        
        // Conv2: in_ch=8, out_ch=16, kernel=3, stride=2
        // 384 weights [352:735]
        weights[352] = 8'h40;
        weights[353] = 8'h40;
        weights[354] = 8'h40;
        // ... continue pattern
        
        // Dense: in=16, out=1
        // 16 weights [736:751]
        weights[736] = 8'h32;
        weights[737] = 8'h4E;
        weights[738] = 8'h3C;
        weights[739] = 8'h44;
        weights[740] = 8'h28;
        weights[741] = 8'h58;
        weights[742] = 8'h38;
        weights[743] = 8'h48;
        weights[744] = 8'h30;
        weights[745] = 8'h50;
        weights[746] = 8'h34;
        weights[747] = 8'h4C;
        weights[748] = 8'h2C;
        weights[749] = 8'h54;
        weights[750] = 8'h36;
        weights[751] = 8'h4A;
        
        // Initialize remaining memory to zero
        // (In actual implementation, use $readmemh for full table)
    end
    
    //--------------------------------------------------------------------------
    // Synchronous Read
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        data <= weights[addr];
    end

endmodule


//==============================================================================
// Bias ROM Module
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
    output reg  [DATA_WIDTH-1:0]    data
);

    reg [DATA_WIDTH-1:0] biases [0:DEPTH-1];
    
    initial begin
        //----------------------------------------------------------------------
        // Generator Biases
        //----------------------------------------------------------------------
        // Encoder Conv1: 4 biases [0:3]
        biases[0]  = 16'h0010;  // 0.0625
        biases[1]  = 16'hFFF0;  // -0.0625
        biases[2]  = 16'h0008;  // 0.03125
        biases[3]  = 16'hFFF8;  // -0.03125
        
        // Bottleneck: 8 biases [4:11]
        biases[4]  = 16'h000C;
        biases[5]  = 16'hFFF4;
        biases[6]  = 16'h0014;
        biases[7]  = 16'hFFEC;
        biases[8]  = 16'h0018;
        biases[9]  = 16'hFFE8;
        biases[10] = 16'h0004;
        biases[11] = 16'hFFFC;
        
        // Decoder Conv1: 4 biases [12:15]
        biases[12] = 16'h0012;
        biases[13] = 16'hFFEE;
        biases[14] = 16'h000A;
        biases[15] = 16'hFFF6;
        
        // Output Conv: 2 biases [16:17]
        biases[16] = 16'h0000;  // Zero bias for output
        biases[17] = 16'h0000;
        
        //----------------------------------------------------------------------
        // Discriminator Biases
        //----------------------------------------------------------------------
        // Conv1: 8 biases [32:39]
        biases[32] = 16'h0008;
        biases[33] = 16'hFFF8;
        biases[34] = 16'h0010;
        biases[35] = 16'hFFF0;
        biases[36] = 16'h000C;
        biases[37] = 16'hFFF4;
        biases[38] = 16'h0014;
        biases[39] = 16'hFFEC;
        
        // Conv2: 16 biases [40:55]
        biases[40] = 16'h0006;
        biases[41] = 16'hFFFA;
        biases[42] = 16'h000E;
        biases[43] = 16'hFFF2;
        biases[44] = 16'h0004;
        biases[45] = 16'hFFFC;
        biases[46] = 16'h0012;
        biases[47] = 16'hFFEE;
        biases[48] = 16'h0008;
        biases[49] = 16'hFFF8;
        biases[50] = 16'h000A;
        biases[51] = 16'hFFF6;
        biases[52] = 16'h0010;
        biases[53] = 16'hFFF0;
        biases[54] = 16'h0002;
        biases[55] = 16'hFFFE;
        
        // Dense: 1 bias [56]
        biases[56] = 16'h0000;
    end
    
    always @(posedge clk) begin
        data <= biases[addr];
    end

endmodule
