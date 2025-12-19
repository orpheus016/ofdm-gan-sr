//==============================================================================
// Sigmoid Activation Module
//
// Implements sigmoid(x) = 1 / (1 + exp(-x))
// Using LUT-based approximation
//
// Properties:
//   - sigmoid(0) = 0.5
//   - sigmoid(x) -> 1 as x -> +inf
//   - sigmoid(x) -> 0 as x -> -inf
//   - sigmoid(-x) = 1 - sigmoid(x)
//
// Fixed-Point: Q8.8 (16-bit signed)
// Output range: [0, 1] in Q8.8 = [0x0000, 0x0100]
//==============================================================================

`timescale 1ns / 1ps

module activation_sigmoid #(
    parameter DATA_WIDTH = 16,             // Q8.8 format
    parameter LUT_DEPTH  = 256,            // LUT entries (8-bit address)
    parameter PIPELINED  = 1
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output reg signed [DATA_WIDTH-1:0]  data_out,
    output reg                          valid_out
);

    //--------------------------------------------------------------------------
    // LUT for sigmoid (positive values only)
    // For negative x: sigmoid(-x) = 1 - sigmoid(x)
    //
    // LUT stores sigmoid(x) for x in [0, 8) with 256 entries
    // Address = x[10:3] for range [0, 8)
    // Values stored in Q8.8 (256 = 1.0)
    //--------------------------------------------------------------------------
    
    reg [DATA_WIDTH-1:0] sigmoid_lut [0:LUT_DEPTH-1];
    
    // Initialize LUT: sigmoid(i * 8/256) = sigmoid(i/32)
    initial begin
        // sigmoid(x) in Q8.8: 256 = 1.0, 128 = 0.5
        sigmoid_lut[0]   = 16'h0080;  // sigmoid(0.000) = 0.500
        sigmoid_lut[1]   = 16'h0082;  // sigmoid(0.031) = 0.508
        sigmoid_lut[2]   = 16'h0084;  // sigmoid(0.063) = 0.516
        sigmoid_lut[3]   = 16'h0086;  // sigmoid(0.094) = 0.523
        sigmoid_lut[4]   = 16'h0088;  // sigmoid(0.125) = 0.531
        sigmoid_lut[5]   = 16'h008A;  // sigmoid(0.156) = 0.539
        sigmoid_lut[6]   = 16'h008C;  // sigmoid(0.188) = 0.547
        sigmoid_lut[7]   = 16'h008E;  // sigmoid(0.219) = 0.555
        sigmoid_lut[8]   = 16'h0090;  // sigmoid(0.250) = 0.562
        sigmoid_lut[9]   = 16'h0092;  // sigmoid(0.281) = 0.570
        sigmoid_lut[10]  = 16'h0094;  // sigmoid(0.313) = 0.578
        sigmoid_lut[11]  = 16'h0096;  // sigmoid(0.344) = 0.585
        sigmoid_lut[12]  = 16'h0098;  // sigmoid(0.375) = 0.593
        sigmoid_lut[13]  = 16'h009A;  // sigmoid(0.406) = 0.600
        sigmoid_lut[14]  = 16'h009C;  // sigmoid(0.438) = 0.608
        sigmoid_lut[15]  = 16'h009E;  // sigmoid(0.469) = 0.615
        sigmoid_lut[16]  = 16'h00A0;  // sigmoid(0.500) = 0.622
        sigmoid_lut[17]  = 16'h00A2;  // sigmoid(0.531) = 0.630
        sigmoid_lut[18]  = 16'h00A4;  // sigmoid(0.563) = 0.637
        sigmoid_lut[19]  = 16'h00A5;  // sigmoid(0.594) = 0.644
        sigmoid_lut[20]  = 16'h00A7;  // sigmoid(0.625) = 0.651
        sigmoid_lut[21]  = 16'h00A9;  // sigmoid(0.656) = 0.658
        sigmoid_lut[22]  = 16'h00AB;  // sigmoid(0.688) = 0.665
        sigmoid_lut[23]  = 16'h00AC;  // sigmoid(0.719) = 0.672
        sigmoid_lut[24]  = 16'h00AE;  // sigmoid(0.750) = 0.679
        sigmoid_lut[25]  = 16'h00B0;  // sigmoid(0.781) = 0.686
        sigmoid_lut[26]  = 16'h00B1;  // sigmoid(0.813) = 0.693
        sigmoid_lut[27]  = 16'h00B3;  // sigmoid(0.844) = 0.699
        sigmoid_lut[28]  = 16'h00B5;  // sigmoid(0.875) = 0.706
        sigmoid_lut[29]  = 16'h00B6;  // sigmoid(0.906) = 0.712
        sigmoid_lut[30]  = 16'h00B8;  // sigmoid(0.938) = 0.718
        sigmoid_lut[31]  = 16'h00B9;  // sigmoid(0.969) = 0.725
        sigmoid_lut[32]  = 16'h00BB;  // sigmoid(1.000) = 0.731
        sigmoid_lut[33]  = 16'h00BC;  // sigmoid(1.031) = 0.737
        sigmoid_lut[34]  = 16'h00BE;  // sigmoid(1.063) = 0.743
        sigmoid_lut[35]  = 16'h00BF;  // sigmoid(1.094) = 0.749
        sigmoid_lut[36]  = 16'h00C0;  // sigmoid(1.125) = 0.755
        sigmoid_lut[37]  = 16'h00C2;  // sigmoid(1.156) = 0.760
        sigmoid_lut[38]  = 16'h00C3;  // sigmoid(1.188) = 0.766
        sigmoid_lut[39]  = 16'h00C4;  // sigmoid(1.219) = 0.772
        sigmoid_lut[40]  = 16'h00C6;  // sigmoid(1.250) = 0.777
        sigmoid_lut[41]  = 16'h00C7;  // sigmoid(1.281) = 0.783
        sigmoid_lut[42]  = 16'h00C8;  // sigmoid(1.313) = 0.788
        sigmoid_lut[43]  = 16'h00C9;  // sigmoid(1.344) = 0.793
        sigmoid_lut[44]  = 16'h00CA;  // sigmoid(1.375) = 0.798
        sigmoid_lut[45]  = 16'h00CB;  // sigmoid(1.406) = 0.803
        sigmoid_lut[46]  = 16'h00CC;  // sigmoid(1.438) = 0.808
        sigmoid_lut[47]  = 16'h00CD;  // sigmoid(1.469) = 0.813
        sigmoid_lut[48]  = 16'h00CE;  // sigmoid(1.500) = 0.818
        sigmoid_lut[49]  = 16'h00CF;  // sigmoid(1.531) = 0.822
        sigmoid_lut[50]  = 16'h00D0;  // sigmoid(1.563) = 0.827
        sigmoid_lut[51]  = 16'h00D1;  // sigmoid(1.594) = 0.831
        sigmoid_lut[52]  = 16'h00D2;  // sigmoid(1.625) = 0.836
        sigmoid_lut[53]  = 16'h00D3;  // sigmoid(1.656) = 0.840
        sigmoid_lut[54]  = 16'h00D4;  // sigmoid(1.688) = 0.844
        sigmoid_lut[55]  = 16'h00D5;  // sigmoid(1.719) = 0.848
        sigmoid_lut[56]  = 16'h00D5;  // sigmoid(1.750) = 0.852
        sigmoid_lut[57]  = 16'h00D6;  // sigmoid(1.781) = 0.856
        sigmoid_lut[58]  = 16'h00D7;  // sigmoid(1.813) = 0.860
        sigmoid_lut[59]  = 16'h00D8;  // sigmoid(1.844) = 0.863
        sigmoid_lut[60]  = 16'h00D8;  // sigmoid(1.875) = 0.867
        sigmoid_lut[61]  = 16'h00D9;  // sigmoid(1.906) = 0.870
        sigmoid_lut[62]  = 16'h00DA;  // sigmoid(1.938) = 0.874
        sigmoid_lut[63]  = 16'h00DA;  // sigmoid(1.969) = 0.877
        sigmoid_lut[64]  = 16'h00DB;  // sigmoid(2.000) = 0.881
        sigmoid_lut[65]  = 16'h00DC;  // sigmoid(2.031) = 0.884
        sigmoid_lut[66]  = 16'h00DC;  // sigmoid(2.063) = 0.887
        sigmoid_lut[67]  = 16'h00DD;  // sigmoid(2.094) = 0.890
        sigmoid_lut[68]  = 16'h00DD;  // sigmoid(2.125) = 0.893
        sigmoid_lut[69]  = 16'h00DE;  // sigmoid(2.156) = 0.896
        sigmoid_lut[70]  = 16'h00DE;  // sigmoid(2.188) = 0.899
        sigmoid_lut[71]  = 16'h00DF;  // sigmoid(2.219) = 0.902
        sigmoid_lut[72]  = 16'h00DF;  // sigmoid(2.250) = 0.905
        sigmoid_lut[73]  = 16'h00E0;  // sigmoid(2.281) = 0.907
        sigmoid_lut[74]  = 16'h00E0;  // sigmoid(2.313) = 0.910
        sigmoid_lut[75]  = 16'h00E1;  // sigmoid(2.344) = 0.912
        sigmoid_lut[76]  = 16'h00E1;  // sigmoid(2.375) = 0.915
        sigmoid_lut[77]  = 16'h00E2;  // sigmoid(2.406) = 0.917
        sigmoid_lut[78]  = 16'h00E2;  // sigmoid(2.438) = 0.919
        sigmoid_lut[79]  = 16'h00E3;  // sigmoid(2.469) = 0.922
        sigmoid_lut[80]  = 16'h00E3;  // sigmoid(2.500) = 0.924
        sigmoid_lut[81]  = 16'h00E4;  // sigmoid(2.531) = 0.926
        sigmoid_lut[82]  = 16'h00E4;  // sigmoid(2.563) = 0.928
        sigmoid_lut[83]  = 16'h00E4;  // sigmoid(2.594) = 0.930
        sigmoid_lut[84]  = 16'h00E5;  // sigmoid(2.625) = 0.932
        sigmoid_lut[85]  = 16'h00E5;  // sigmoid(2.656) = 0.934
        sigmoid_lut[86]  = 16'h00E5;  // sigmoid(2.688) = 0.936
        sigmoid_lut[87]  = 16'h00E6;  // sigmoid(2.719) = 0.938
        sigmoid_lut[88]  = 16'h00E6;  // sigmoid(2.750) = 0.940
        sigmoid_lut[89]  = 16'h00E6;  // sigmoid(2.781) = 0.942
        sigmoid_lut[90]  = 16'h00E7;  // sigmoid(2.813) = 0.943
        sigmoid_lut[91]  = 16'h00E7;  // sigmoid(2.844) = 0.945
        sigmoid_lut[92]  = 16'h00E7;  // sigmoid(2.875) = 0.947
        sigmoid_lut[93]  = 16'h00E8;  // sigmoid(2.906) = 0.948
        sigmoid_lut[94]  = 16'h00E8;  // sigmoid(2.938) = 0.950
        sigmoid_lut[95]  = 16'h00E8;  // sigmoid(2.969) = 0.951
        sigmoid_lut[96]  = 16'h00E9;  // sigmoid(3.000) = 0.953
        sigmoid_lut[97]  = 16'h00E9;  // sigmoid(3.031) = 0.954
        sigmoid_lut[98]  = 16'h00E9;  // sigmoid(3.063) = 0.955
        sigmoid_lut[99]  = 16'h00E9;  // sigmoid(3.094) = 0.957
        sigmoid_lut[100] = 16'h00EA;  // sigmoid(3.125) = 0.958
        sigmoid_lut[101] = 16'h00EA;  // sigmoid(3.156) = 0.959
        sigmoid_lut[102] = 16'h00EA;  // sigmoid(3.188) = 0.960
        sigmoid_lut[103] = 16'h00EA;  // sigmoid(3.219) = 0.962
        sigmoid_lut[104] = 16'h00EB;  // sigmoid(3.250) = 0.963
        sigmoid_lut[105] = 16'h00EB;  // sigmoid(3.281) = 0.964
        sigmoid_lut[106] = 16'h00EB;  // sigmoid(3.313) = 0.965
        sigmoid_lut[107] = 16'h00EB;  // sigmoid(3.344) = 0.966
        sigmoid_lut[108] = 16'h00EB;  // sigmoid(3.375) = 0.967
        sigmoid_lut[109] = 16'h00EC;  // sigmoid(3.406) = 0.968
        sigmoid_lut[110] = 16'h00EC;  // sigmoid(3.438) = 0.969
        sigmoid_lut[111] = 16'h00EC;  // sigmoid(3.469) = 0.970
        sigmoid_lut[112] = 16'h00EC;  // sigmoid(3.500) = 0.971
        sigmoid_lut[113] = 16'h00EC;  // sigmoid(3.531) = 0.971
        sigmoid_lut[114] = 16'h00ED;  // sigmoid(3.563) = 0.972
        sigmoid_lut[115] = 16'h00ED;  // sigmoid(3.594) = 0.973
        sigmoid_lut[116] = 16'h00ED;  // sigmoid(3.625) = 0.974
        sigmoid_lut[117] = 16'h00ED;  // sigmoid(3.656) = 0.975
        sigmoid_lut[118] = 16'h00ED;  // sigmoid(3.688) = 0.975
        sigmoid_lut[119] = 16'h00ED;  // sigmoid(3.719) = 0.976
        sigmoid_lut[120] = 16'h00EE;  // sigmoid(3.750) = 0.977
        sigmoid_lut[121] = 16'h00EE;  // sigmoid(3.781) = 0.978
        sigmoid_lut[122] = 16'h00EE;  // sigmoid(3.813) = 0.978
        sigmoid_lut[123] = 16'h00EE;  // sigmoid(3.844) = 0.979
        sigmoid_lut[124] = 16'h00EE;  // sigmoid(3.875) = 0.979
        sigmoid_lut[125] = 16'h00EE;  // sigmoid(3.906) = 0.980
        sigmoid_lut[126] = 16'h00EE;  // sigmoid(3.938) = 0.981
        sigmoid_lut[127] = 16'h00EF;  // sigmoid(3.969) = 0.981
        // Values 128-255: near saturation (0.982 - 0.999)
        sigmoid_lut[128] = 16'h00EF;  // sigmoid(4.000) = 0.982
        sigmoid_lut[129] = 16'h00EF;
        sigmoid_lut[130] = 16'h00EF;
        sigmoid_lut[131] = 16'h00EF;
        sigmoid_lut[132] = 16'h00EF;
        sigmoid_lut[133] = 16'h00F0;
        sigmoid_lut[134] = 16'h00F0;
        sigmoid_lut[135] = 16'h00F0;
        sigmoid_lut[136] = 16'h00F0;
        sigmoid_lut[137] = 16'h00F0;
        sigmoid_lut[138] = 16'h00F0;
        sigmoid_lut[139] = 16'h00F0;
        sigmoid_lut[140] = 16'h00F1;
        sigmoid_lut[141] = 16'h00F1;
        sigmoid_lut[142] = 16'h00F1;
        sigmoid_lut[143] = 16'h00F1;
        sigmoid_lut[144] = 16'h00F1;
        sigmoid_lut[145] = 16'h00F1;
        sigmoid_lut[146] = 16'h00F1;
        sigmoid_lut[147] = 16'h00F2;
        sigmoid_lut[148] = 16'h00F2;
        sigmoid_lut[149] = 16'h00F2;
        sigmoid_lut[150] = 16'h00F2;
        sigmoid_lut[151] = 16'h00F2;
        sigmoid_lut[152] = 16'h00F2;
        sigmoid_lut[153] = 16'h00F2;
        sigmoid_lut[154] = 16'h00F3;
        sigmoid_lut[155] = 16'h00F3;
        sigmoid_lut[156] = 16'h00F3;
        sigmoid_lut[157] = 16'h00F3;
        sigmoid_lut[158] = 16'h00F3;
        sigmoid_lut[159] = 16'h00F3;
        sigmoid_lut[160] = 16'h00F3;
        sigmoid_lut[161] = 16'h00F4;
        sigmoid_lut[162] = 16'h00F4;
        sigmoid_lut[163] = 16'h00F4;
        sigmoid_lut[164] = 16'h00F4;
        sigmoid_lut[165] = 16'h00F4;
        sigmoid_lut[166] = 16'h00F4;
        sigmoid_lut[167] = 16'h00F4;
        sigmoid_lut[168] = 16'h00F5;
        sigmoid_lut[169] = 16'h00F5;
        sigmoid_lut[170] = 16'h00F5;
        sigmoid_lut[171] = 16'h00F5;
        sigmoid_lut[172] = 16'h00F5;
        sigmoid_lut[173] = 16'h00F5;
        sigmoid_lut[174] = 16'h00F5;
        sigmoid_lut[175] = 16'h00F5;
        sigmoid_lut[176] = 16'h00F6;
        sigmoid_lut[177] = 16'h00F6;
        sigmoid_lut[178] = 16'h00F6;
        sigmoid_lut[179] = 16'h00F6;
        sigmoid_lut[180] = 16'h00F6;
        sigmoid_lut[181] = 16'h00F6;
        sigmoid_lut[182] = 16'h00F6;
        sigmoid_lut[183] = 16'h00F6;
        sigmoid_lut[184] = 16'h00F7;
        sigmoid_lut[185] = 16'h00F7;
        sigmoid_lut[186] = 16'h00F7;
        sigmoid_lut[187] = 16'h00F7;
        sigmoid_lut[188] = 16'h00F7;
        sigmoid_lut[189] = 16'h00F7;
        sigmoid_lut[190] = 16'h00F7;
        sigmoid_lut[191] = 16'h00F7;
        sigmoid_lut[192] = 16'h00F8;
        sigmoid_lut[193] = 16'h00F8;
        sigmoid_lut[194] = 16'h00F8;
        sigmoid_lut[195] = 16'h00F8;
        sigmoid_lut[196] = 16'h00F8;
        sigmoid_lut[197] = 16'h00F8;
        sigmoid_lut[198] = 16'h00F8;
        sigmoid_lut[199] = 16'h00F8;
        sigmoid_lut[200] = 16'h00F8;
        sigmoid_lut[201] = 16'h00F9;
        sigmoid_lut[202] = 16'h00F9;
        sigmoid_lut[203] = 16'h00F9;
        sigmoid_lut[204] = 16'h00F9;
        sigmoid_lut[205] = 16'h00F9;
        sigmoid_lut[206] = 16'h00F9;
        sigmoid_lut[207] = 16'h00F9;
        sigmoid_lut[208] = 16'h00F9;
        sigmoid_lut[209] = 16'h00F9;
        sigmoid_lut[210] = 16'h00FA;
        sigmoid_lut[211] = 16'h00FA;
        sigmoid_lut[212] = 16'h00FA;
        sigmoid_lut[213] = 16'h00FA;
        sigmoid_lut[214] = 16'h00FA;
        sigmoid_lut[215] = 16'h00FA;
        sigmoid_lut[216] = 16'h00FA;
        sigmoid_lut[217] = 16'h00FA;
        sigmoid_lut[218] = 16'h00FA;
        sigmoid_lut[219] = 16'h00FA;
        sigmoid_lut[220] = 16'h00FB;
        sigmoid_lut[221] = 16'h00FB;
        sigmoid_lut[222] = 16'h00FB;
        sigmoid_lut[223] = 16'h00FB;
        sigmoid_lut[224] = 16'h00FB;
        sigmoid_lut[225] = 16'h00FB;
        sigmoid_lut[226] = 16'h00FB;
        sigmoid_lut[227] = 16'h00FB;
        sigmoid_lut[228] = 16'h00FB;
        sigmoid_lut[229] = 16'h00FB;
        sigmoid_lut[230] = 16'h00FC;
        sigmoid_lut[231] = 16'h00FC;
        sigmoid_lut[232] = 16'h00FC;
        sigmoid_lut[233] = 16'h00FC;
        sigmoid_lut[234] = 16'h00FC;
        sigmoid_lut[235] = 16'h00FC;
        sigmoid_lut[236] = 16'h00FC;
        sigmoid_lut[237] = 16'h00FC;
        sigmoid_lut[238] = 16'h00FC;
        sigmoid_lut[239] = 16'h00FC;
        sigmoid_lut[240] = 16'h00FD;
        sigmoid_lut[241] = 16'h00FD;
        sigmoid_lut[242] = 16'h00FD;
        sigmoid_lut[243] = 16'h00FD;
        sigmoid_lut[244] = 16'h00FD;
        sigmoid_lut[245] = 16'h00FD;
        sigmoid_lut[246] = 16'h00FD;
        sigmoid_lut[247] = 16'h00FD;
        sigmoid_lut[248] = 16'h00FE;
        sigmoid_lut[249] = 16'h00FE;
        sigmoid_lut[250] = 16'h00FE;
        sigmoid_lut[251] = 16'h00FE;
        sigmoid_lut[252] = 16'h00FE;
        sigmoid_lut[253] = 16'h00FE;
        sigmoid_lut[254] = 16'h00FF;
        sigmoid_lut[255] = 16'h00FF;  // sigmoid(8.0) ≈ 0.9997
    end

    //--------------------------------------------------------------------------
    // Sigmoid Computation
    //--------------------------------------------------------------------------
    
    wire is_negative = data_in[DATA_WIDTH-1];
    wire [DATA_WIDTH-1:0] abs_input = is_negative ? -data_in : data_in;
    
    // LUT address: use upper bits of absolute value
    // For range [0, 8), use bits [10:3]
    wire [7:0] lut_addr = (abs_input > 16'h0800) ? 8'hFF : abs_input[10:3];
    
    wire [DATA_WIDTH-1:0] lut_value = sigmoid_lut[lut_addr];
    
    // For negative input: sigmoid(-x) = 1 - sigmoid(x)
    // 1.0 in Q8.8 = 0x0100 = 256
    wire [DATA_WIDTH-1:0] result = is_negative ? (16'h0100 - lut_value) : lut_value;
    
    //--------------------------------------------------------------------------
    // Pipeline Register (optional)
    //--------------------------------------------------------------------------
    
    generate
        if (PIPELINED) begin : gen_pipe
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    data_out <= 0;
                    valid_out <= 0;
                end else begin
                    data_out <= result;
                    valid_out <= valid_in;
                end
            end
        end else begin : gen_comb
            always @(*) begin
                data_out = result;
                valid_out = valid_in;
            end
        end
    endgenerate

endmodule
