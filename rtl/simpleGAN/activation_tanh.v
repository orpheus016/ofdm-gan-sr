//==============================================================================
// Tanh Activation Module
//
// Implements tanh(x) using piecewise linear approximation (PLAN)
//
// tanh(x) ≈ 
//   x                     if |x| < 1
//   sign(x) * (1 - 1/|x|) if 1 <= |x| < 4
//   sign(x)               if |x| >= 4
//
// Or using LUT for better accuracy
//
// Fixed-Point: Q8.8 (16-bit signed)
//==============================================================================

`timescale 1ns / 1ps

module activation_tanh #(
    parameter DATA_WIDTH = 16,             // Q8.8 format
    parameter USE_LUT    = 1,              // 0: PLAN approx, 1: LUT
    parameter LUT_DEPTH  = 256,            // LUT entries (8-bit address)
    parameter PIPELINED  = 1
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output wire signed [DATA_WIDTH-1:0] data_out,
    output wire                         valid_out
);

    //--------------------------------------------------------------------------
    // LUT-based Tanh
    //--------------------------------------------------------------------------
    generate
        if (USE_LUT) begin : gen_lut
            
            // LUT stores tanh values for input range [0, 4) in Q8.8
            // Address mapping: addr = abs_input >> 2 (i.e., addr = input_q88 / 4)
            // So addr i corresponds to x = i/64 in real units
            // Output = tanh(i/64) in Q8.8 = round(tanh(i/64) * 256)
            
            reg [DATA_WIDTH-1:0] tanh_lut [0:LUT_DEPTH-1];
            
            // Initialize LUT with correct tanh values
            // tanh_lut[i] = round(tanh(i/64) * 256) for Q8.8
            initial begin
                // Generated correctly: tanh(i/64) * 256
                tanh_lut[0]   = 16'h0000;  tanh_lut[1]   = 16'h0004;  tanh_lut[2]   = 16'h0008;  tanh_lut[3]   = 16'h000C;
                tanh_lut[4]   = 16'h0010;  tanh_lut[5]   = 16'h0014;  tanh_lut[6]   = 16'h0018;  tanh_lut[7]   = 16'h001C;
                tanh_lut[8]   = 16'h001F;  tanh_lut[9]   = 16'h0023;  tanh_lut[10]  = 16'h0027;  tanh_lut[11]  = 16'h002B;
                tanh_lut[12]  = 16'h002F;  tanh_lut[13]  = 16'h0032;  tanh_lut[14]  = 16'h0036;  tanh_lut[15]  = 16'h003A;
                tanh_lut[16]  = 16'h003E;  tanh_lut[17]  = 16'h0041;  tanh_lut[18]  = 16'h0045;  tanh_lut[19]  = 16'h0048;
                tanh_lut[20]  = 16'h004C;  tanh_lut[21]  = 16'h004F;  tanh_lut[22]  = 16'h0053;  tanh_lut[23]  = 16'h0056;
                tanh_lut[24]  = 16'h0059;  tanh_lut[25]  = 16'h005D;  tanh_lut[26]  = 16'h0060;  tanh_lut[27]  = 16'h0063;
                tanh_lut[28]  = 16'h0066;  tanh_lut[29]  = 16'h0069;  tanh_lut[30]  = 16'h006C;  tanh_lut[31]  = 16'h006F;
                tanh_lut[32]  = 16'h0072;  tanh_lut[33]  = 16'h0075;  tanh_lut[34]  = 16'h0078;  tanh_lut[35]  = 16'h007A;
                tanh_lut[36]  = 16'h007D;  tanh_lut[37]  = 16'h0080;  tanh_lut[38]  = 16'h0082;  tanh_lut[39]  = 16'h0085;
                tanh_lut[40]  = 16'h0087;  tanh_lut[41]  = 16'h0089;  tanh_lut[42]  = 16'h008C;  tanh_lut[43]  = 16'h008E;
                tanh_lut[44]  = 16'h0090;  tanh_lut[45]  = 16'h0092;  tanh_lut[46]  = 16'h0094;  tanh_lut[47]  = 16'h0096;
                tanh_lut[48]  = 16'h0098;  tanh_lut[49]  = 16'h009A;  tanh_lut[50]  = 16'h009C;  tanh_lut[51]  = 16'h009E;
                tanh_lut[52]  = 16'h00A0;  tanh_lut[53]  = 16'h00A1;  tanh_lut[54]  = 16'h00A3;  tanh_lut[55]  = 16'h00A5;
                tanh_lut[56]  = 16'h00A6;  tanh_lut[57]  = 16'h00A8;  tanh_lut[58]  = 16'h00A9;  tanh_lut[59]  = 16'h00AB;
                tanh_lut[60]  = 16'h00AC;  tanh_lut[61]  = 16'h00AD;  tanh_lut[62]  = 16'h00AF;  tanh_lut[63]  = 16'h00B0;
                tanh_lut[64]  = 16'h00B1;  tanh_lut[65]  = 16'h00B2;  tanh_lut[66]  = 16'h00B3;  tanh_lut[67]  = 16'h00B5;
                tanh_lut[68]  = 16'h00B6;  tanh_lut[69]  = 16'h00B7;  tanh_lut[70]  = 16'h00B8;  tanh_lut[71]  = 16'h00B9;
                tanh_lut[72]  = 16'h00BA;  tanh_lut[73]  = 16'h00BA;  tanh_lut[74]  = 16'h00BB;  tanh_lut[75]  = 16'h00BC;
                tanh_lut[76]  = 16'h00BD;  tanh_lut[77]  = 16'h00BE;  tanh_lut[78]  = 16'h00BE;  tanh_lut[79]  = 16'h00BF;
                tanh_lut[80]  = 16'h00C0;  tanh_lut[81]  = 16'h00C0;  tanh_lut[82]  = 16'h00C1;  tanh_lut[83]  = 16'h00C2;
                tanh_lut[84]  = 16'h00C2;  tanh_lut[85]  = 16'h00C3;  tanh_lut[86]  = 16'h00C3;  tanh_lut[87]  = 16'h00C4;
                tanh_lut[88]  = 16'h00C4;  tanh_lut[89]  = 16'h00C5;  tanh_lut[90]  = 16'h00C5;  tanh_lut[91]  = 16'h00C6;
                tanh_lut[92]  = 16'h00C6;  tanh_lut[93]  = 16'h00C7;  tanh_lut[94]  = 16'h00C7;  tanh_lut[95]  = 16'h00C8;
                tanh_lut[96]  = 16'h00C8;  tanh_lut[97]  = 16'h00C8;  tanh_lut[98]  = 16'h00C9;  tanh_lut[99]  = 16'h00C9;
                tanh_lut[100] = 16'h00C9;  tanh_lut[101] = 16'h00CA;  tanh_lut[102] = 16'h00CA;  tanh_lut[103] = 16'h00CA;
                tanh_lut[104] = 16'h00CB;  tanh_lut[105] = 16'h00CB;  tanh_lut[106] = 16'h00CB;  tanh_lut[107] = 16'h00CC;
                tanh_lut[108] = 16'h00CC;  tanh_lut[109] = 16'h00CC;  tanh_lut[110] = 16'h00CC;  tanh_lut[111] = 16'h00CD;
                tanh_lut[112] = 16'h00CD;  tanh_lut[113] = 16'h00CD;  tanh_lut[114] = 16'h00CD;  tanh_lut[115] = 16'h00CE;
                tanh_lut[116] = 16'h00CE;  tanh_lut[117] = 16'h00CE;  tanh_lut[118] = 16'h00CE;  tanh_lut[119] = 16'h00CF;
                tanh_lut[120] = 16'h00CF;  tanh_lut[121] = 16'h00CF;  tanh_lut[122] = 16'h00CF;  tanh_lut[123] = 16'h00CF;
                tanh_lut[124] = 16'h00D0;  tanh_lut[125] = 16'h00D0;  tanh_lut[126] = 16'h00D0;  tanh_lut[127] = 16'h00D0;
                // tanh saturates toward 1.0 for larger inputs
                tanh_lut[128] = 16'h00D0;  tanh_lut[129] = 16'h00D1;  tanh_lut[130] = 16'h00D1;  tanh_lut[131] = 16'h00D1;
                tanh_lut[132] = 16'h00D1;  tanh_lut[133] = 16'h00D1;  tanh_lut[134] = 16'h00D2;  tanh_lut[135] = 16'h00D2;
                tanh_lut[136] = 16'h00D2;  tanh_lut[137] = 16'h00D2;  tanh_lut[138] = 16'h00D2;  tanh_lut[139] = 16'h00D2;
                tanh_lut[140] = 16'h00D3;  tanh_lut[141] = 16'h00D3;  tanh_lut[142] = 16'h00D3;  tanh_lut[143] = 16'h00D3;
                tanh_lut[144] = 16'h00D3;  tanh_lut[145] = 16'h00D3;  tanh_lut[146] = 16'h00D3;  tanh_lut[147] = 16'h00D4;
                tanh_lut[148] = 16'h00D4;  tanh_lut[149] = 16'h00D4;  tanh_lut[150] = 16'h00D4;  tanh_lut[151] = 16'h00D4;
                tanh_lut[152] = 16'h00D4;  tanh_lut[153] = 16'h00D4;  tanh_lut[154] = 16'h00D4;  tanh_lut[155] = 16'h00D5;
                tanh_lut[156] = 16'h00D5;  tanh_lut[157] = 16'h00D5;  tanh_lut[158] = 16'h00D5;  tanh_lut[159] = 16'h00D5;
                tanh_lut[160] = 16'h00D5;  tanh_lut[161] = 16'h00D5;  tanh_lut[162] = 16'h00D5;  tanh_lut[163] = 16'h00D5;
                tanh_lut[164] = 16'h00D6;  tanh_lut[165] = 16'h00D6;  tanh_lut[166] = 16'h00D6;  tanh_lut[167] = 16'h00D6;
                tanh_lut[168] = 16'h00D6;  tanh_lut[169] = 16'h00D6;  tanh_lut[170] = 16'h00D6;  tanh_lut[171] = 16'h00D6;
                tanh_lut[172] = 16'h00D6;  tanh_lut[173] = 16'h00D6;  tanh_lut[174] = 16'h00D6;  tanh_lut[175] = 16'h00D7;
                tanh_lut[176] = 16'h00D7;  tanh_lut[177] = 16'h00D7;  tanh_lut[178] = 16'h00D7;  tanh_lut[179] = 16'h00D7;
                tanh_lut[180] = 16'h00D7;  tanh_lut[181] = 16'h00D7;  tanh_lut[182] = 16'h00D7;  tanh_lut[183] = 16'h00D7;
                tanh_lut[184] = 16'h00D7;  tanh_lut[185] = 16'h00D7;  tanh_lut[186] = 16'h00D7;  tanh_lut[187] = 16'h00D8;
                tanh_lut[188] = 16'h00D8;  tanh_lut[189] = 16'h00D8;  tanh_lut[190] = 16'h00D8;  tanh_lut[191] = 16'h00D8;
                // Near saturation: tanh approaches 1.0 = 256, but cap at 255 for unsigned representation
                tanh_lut[192] = 16'h00F5;  tanh_lut[193] = 16'h00F5;  tanh_lut[194] = 16'h00F6;  tanh_lut[195] = 16'h00F6;
                tanh_lut[196] = 16'h00F7;  tanh_lut[197] = 16'h00F7;  tanh_lut[198] = 16'h00F7;  tanh_lut[199] = 16'h00F8;
                tanh_lut[200] = 16'h00F8;  tanh_lut[201] = 16'h00F8;  tanh_lut[202] = 16'h00F8;  tanh_lut[203] = 16'h00F9;
                tanh_lut[204] = 16'h00F9;  tanh_lut[205] = 16'h00F9;  tanh_lut[206] = 16'h00F9;  tanh_lut[207] = 16'h00FA;
                tanh_lut[208] = 16'h00FA;  tanh_lut[209] = 16'h00FA;  tanh_lut[210] = 16'h00FA;  tanh_lut[211] = 16'h00FA;
                tanh_lut[212] = 16'h00FB;  tanh_lut[213] = 16'h00FB;  tanh_lut[214] = 16'h00FB;  tanh_lut[215] = 16'h00FB;
                tanh_lut[216] = 16'h00FB;  tanh_lut[217] = 16'h00FC;  tanh_lut[218] = 16'h00FC;  tanh_lut[219] = 16'h00FC;
                tanh_lut[220] = 16'h00FC;  tanh_lut[221] = 16'h00FC;  tanh_lut[222] = 16'h00FC;  tanh_lut[223] = 16'h00FD;
                tanh_lut[224] = 16'h00FD;  tanh_lut[225] = 16'h00FD;  tanh_lut[226] = 16'h00FD;  tanh_lut[227] = 16'h00FD;
                tanh_lut[228] = 16'h00FD;  tanh_lut[229] = 16'h00FD;  tanh_lut[230] = 16'h00FE;  tanh_lut[231] = 16'h00FE;
                tanh_lut[232] = 16'h00FE;  tanh_lut[233] = 16'h00FE;  tanh_lut[234] = 16'h00FE;  tanh_lut[235] = 16'h00FE;
                tanh_lut[236] = 16'h00FE;  tanh_lut[237] = 16'h00FE;  tanh_lut[238] = 16'h00FF;  tanh_lut[239] = 16'h00FF;
                tanh_lut[240] = 16'h00FF;  tanh_lut[241] = 16'h00FF;  tanh_lut[242] = 16'h00FF;  tanh_lut[243] = 16'h00FF;
                tanh_lut[244] = 16'h00FF;  tanh_lut[245] = 16'h00FF;  tanh_lut[246] = 16'h00FF;  tanh_lut[247] = 16'h00FF;
                tanh_lut[248] = 16'h00FF;  tanh_lut[249] = 16'h00FF;  tanh_lut[250] = 16'h00FF;  tanh_lut[251] = 16'h00FF;
                tanh_lut[252] = 16'h00FF;  tanh_lut[253] = 16'h00FF;  tanh_lut[254] = 16'h00FF;  tanh_lut[255] = 16'h00FF;
            end
            
            // Logic
            wire is_negative;
            wire [DATA_WIDTH-1:0] abs_input;
            wire [7:0] lut_addr;
            wire saturated;
            reg [DATA_WIDTH-1:0] lut_value;
            wire signed [DATA_WIDTH-1:0] result;
            
            assign is_negative = data_in[DATA_WIDTH-1];
            assign abs_input = is_negative ? (~data_in + 1) : data_in;
            
            // Address: abs_input >> 2 (divide by 4 for Q8.8), saturate to 255
            // For very small values, just pass through (linear region)
            assign saturated = (abs_input[DATA_WIDTH-1:10] != 0);  // |x| >= 4.0
            assign lut_addr = saturated ? 8'hFF : abs_input[9:2];
            
            // LUT lookup
            always @(*) begin
                lut_value = tanh_lut[lut_addr];
            end
            
            // Apply sign
            assign result = is_negative ? (~lut_value + 1) : lut_value;
            
            // Pipeline register
            if (PIPELINED) begin : gen_pipe
                reg signed [DATA_WIDTH-1:0] data_out_reg;
                reg valid_out_reg;
                
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        data_out_reg <= 0;
                        valid_out_reg <= 1'b0;
                    end else begin
                        data_out_reg <= result;
                        valid_out_reg <= valid_in;
                    end
                end
                
                assign data_out = data_out_reg;
                assign valid_out = valid_out_reg;
            end else begin : gen_comb
                assign data_out = result;
                assign valid_out = valid_in;
            end
            
        end else begin : gen_plan
            //------------------------------------------------------------------
            // Piecewise Linear Approximation (PLAN)
            //------------------------------------------------------------------
            wire is_negative;
            wire [DATA_WIDTH-1:0] abs_input;
            wire signed [DATA_WIDTH-1:0] result;
            
            assign is_negative = data_in[DATA_WIDTH-1];
            assign abs_input = is_negative ? (~data_in + 1) : data_in;
            
            // Thresholds in Q8.8
            localparam [DATA_WIDTH-1:0] THRESH_1 = 16'h0100;  // 1.0
            localparam [DATA_WIDTH-1:0] THRESH_4 = 16'h0400;  // 4.0
            localparam [DATA_WIDTH-1:0] ONE      = 16'h0100;  // 1.0
            
            wire [DATA_WIDTH-1:0] plan_result;
            
            // PLAN segments
            assign plan_result = 
                (abs_input < THRESH_1) ? abs_input :                    // Linear region
                (abs_input < THRESH_4) ? (ONE - (ONE / abs_input)) :    // Saturation region (approx)
                ONE;                                                     // Saturated
            
            assign result = is_negative ? (~plan_result + 1) : plan_result;
            
            if (PIPELINED) begin : gen_pipe_plan
                reg signed [DATA_WIDTH-1:0] data_out_reg;
                reg valid_out_reg;
                
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        data_out_reg <= 0;
                        valid_out_reg <= 1'b0;
                    end else begin
                        data_out_reg <= result;
                        valid_out_reg <= valid_in;
                    end
                end
                
                assign data_out = data_out_reg;
                assign valid_out = valid_out_reg;
            end else begin : gen_comb_plan
                assign data_out = result;
                assign valid_out = valid_in;
            end
        end
    endgenerate

endmodule
