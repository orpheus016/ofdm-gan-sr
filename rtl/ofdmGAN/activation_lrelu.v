//==============================================================================
// LeakyReLU Activation Module
//
// Implements LeakyReLU(x) = x if x > 0, else alpha * x
// where alpha = 0.2 (in fixed-point: 0.2 ≈ 26/128 ≈ 0x1A)
//
// Fixed-Point: Q8.8 (16-bit signed)
//
// Architecture: Purely combinational with pipelined option
//==============================================================================

`timescale 1ns / 1ps

module activation_lrelu #(
    parameter DATA_WIDTH = 16,             // Q8.8 format
    parameter ALPHA_BITS = 8,              // Alpha precision
    parameter ALPHA      = 8'h1A,          // 0.2 in Q0.8 (26/128)
    parameter PIPELINED  = 1               // 0: combinational, 1: registered output
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output wire signed [DATA_WIDTH-1:0] data_out,
    output wire                         valid_out
);

    //--------------------------------------------------------------------------
    // LeakyReLU Logic
    //--------------------------------------------------------------------------
    wire is_negative;
    wire signed [DATA_WIDTH+ALPHA_BITS-1:0] scaled;
    wire signed [DATA_WIDTH-1:0] leaky_result;
    wire signed [DATA_WIDTH-1:0] result;
    
    // Check sign bit
    assign is_negative = data_in[DATA_WIDTH-1];
    
    // Multiply by alpha (shift right by 8 to maintain Q8.8)
    // data_in * ALPHA = Q8.8 * Q0.8 = Q8.16, shift right 8 = Q8.8
    assign scaled = $signed(data_in) * $signed({1'b0, ALPHA});
    assign leaky_result = scaled[DATA_WIDTH+ALPHA_BITS-1:ALPHA_BITS];
    
    // Select between x and alpha*x
    assign result = is_negative ? leaky_result : data_in;
    
    //--------------------------------------------------------------------------
    // Output (pipelined or combinational)
    //--------------------------------------------------------------------------
    generate
        if (PIPELINED) begin : gen_pipelined
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
        end else begin : gen_combinational
            assign data_out = result;
            assign valid_out = valid_in;
        end
    endgenerate

endmodule


//==============================================================================
// Batch of LeakyReLU - Parallel Processing
//
// Processes multiple values in parallel for increased throughput
//==============================================================================

module activation_lrelu_batch #(
    parameter DATA_WIDTH  = 16,
    parameter BATCH_SIZE  = 4,             // Number of parallel units
    parameter ALPHA_BITS  = 8,
    parameter ALPHA       = 8'h1A          // 0.2
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    input  wire signed [BATCH_SIZE*DATA_WIDTH-1:0]  data_in,
    input  wire                                     valid_in,
    output wire signed [BATCH_SIZE*DATA_WIDTH-1:0]  data_out,
    output wire                                     valid_out
);

    genvar i;
    wire [BATCH_SIZE-1:0] valid_out_arr;
    
    generate
        for (i = 0; i < BATCH_SIZE; i = i + 1) begin : gen_lrelu
            activation_lrelu #(
                .DATA_WIDTH(DATA_WIDTH),
                .ALPHA_BITS(ALPHA_BITS),
                .ALPHA(ALPHA),
                .PIPELINED(1)
            ) u_lrelu (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(data_in[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH]),
                .valid_in(valid_in),
                .data_out(data_out[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH]),
                .valid_out(valid_out_arr[i])
            );
        end
    endgenerate
    
    assign valid_out = valid_out_arr[0];  // All have same latency

endmodule
