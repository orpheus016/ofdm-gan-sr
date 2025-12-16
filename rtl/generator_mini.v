//==============================================================================
// Mini U-Net Generator for OFDM Signal Reconstruction
//
// PARALLEL PIPELINED VERSION
//
// Architecture:
//   Input [2×16] → Enc1 [4×8] → Bottleneck [8×4] → Dec1 [4×8] → Output [2×16]
//                 ↓                                    ↑
//                 └──────── Skip Connection ───────────┘
//
// Parallel/Pipeline Features:
//   - PARALLEL KERNEL: 3 multipliers for k=0,1,2 computed in single cycle
//   - PIPELINED MAC: 3-stage pipeline (fetch → multiply → accumulate)
//   - Throughput: ~6x improvement over sequential version
//
// Fixed-point: Q8.8 activations, Q1.7 weights, Q16.16 accumulator
//
// Resource Estimates (FPGA):
//   DSPs: 6 (3 kernel MACs × 2 pipeline stages)
//   BRAMs: 2 (weight + bias ROM)
//   FFs: ~2500
//   LUTs: ~3500
//==============================================================================

`timescale 1ns / 1ps

module generator_mini #(
    parameter DATA_WIDTH   = 16,           // Q8.8 activations
    parameter WEIGHT_WIDTH = 8,            // Q1.7 weights
    parameter ACC_WIDTH    = 32,           // Accumulator width
    parameter FRAME_LEN    = 16,           // Input frame length
    parameter IN_CH        = 2,            // Input channels (I, Q)
    parameter OUT_CH       = 2             // Output channels (I, Q)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    
    // Input: degraded OFDM signal
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Condition input (for CWGAN)
    input  wire signed [DATA_WIDTH-1:0] cond_in,
    input  wire                         cond_valid,
    
    // Output: reconstructed OFDM signal
    output reg  signed [DATA_WIDTH-1:0] data_out,
    output reg                          valid_out,
    input  wire                         ready_out,
    
    // Status
    output wire                         busy,
    output wire                         done
);

    //--------------------------------------------------------------------------
    // Local Parameters
    //--------------------------------------------------------------------------
    localparam ENC1_OUT_CH   = 4;
    localparam ENC1_OUT_LEN  = 8;   // 16/2
    localparam BNECK_OUT_CH  = 8;
    localparam BNECK_OUT_LEN = 4;   // 8/2
    localparam DEC1_OUT_CH   = 4;
    localparam DEC1_OUT_LEN  = 8;   // 4*2
    localparam UP1_LEN       = 8;   // BNECK*2
    
    // Weight ROM base addresses
    localparam WADDR_ENC1   = 0;    // 2*4*3 = 24 weights
    localparam WADDR_BNECK  = 24;   // 4*8*3 = 96 weights  
    localparam WADDR_DEC1   = 120;  // 8*4*3 = 96 weights
    localparam WADDR_OUT    = 216;  // 4*2*1 = 8 weights
    
    // Bias ROM base addresses  
    localparam BADDR_ENC1   = 0;
    localparam BADDR_BNECK  = 4;
    localparam BADDR_DEC1   = 12;
    localparam BADDR_OUT    = 16;

    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    localparam ST_IDLE      = 4'd0;
    localparam ST_LOAD_IN   = 4'd1;
    localparam ST_ENC1      = 4'd2;
    localparam ST_BNECK     = 4'd3;
    localparam ST_UPSAMPLE1 = 4'd4;
    localparam ST_DEC1      = 4'd5;
    localparam ST_SKIP_ADD  = 4'd6;
    localparam ST_UPSAMPLE2 = 4'd7;
    localparam ST_OUT_CONV  = 4'd8;
    localparam ST_TANH      = 4'd9;
    localparam ST_OUTPUT    = 4'd10;
    localparam ST_DONE      = 4'd11;
    
    reg [3:0] state, next_state;
    
    //--------------------------------------------------------------------------
    // Buffers - with padding space for convolutions
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_buf [0:IN_CH-1][0:FRAME_LEN+1];
    reg signed [DATA_WIDTH-1:0] skip_buf  [0:ENC1_OUT_CH-1][0:ENC1_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] enc1_buf  [0:ENC1_OUT_CH-1][0:ENC1_OUT_LEN+1];
    reg signed [DATA_WIDTH-1:0] bneck_buf [0:BNECK_OUT_CH-1][0:BNECK_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] up1_buf   [0:BNECK_OUT_CH-1][0:UP1_LEN+1];
    reg signed [DATA_WIDTH-1:0] dec1_buf  [0:DEC1_OUT_CH-1][0:DEC1_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] up2_buf   [0:DEC1_OUT_CH-1][0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] out_buf   [0:OUT_CH-1][0:FRAME_LEN-1];
    
    //--------------------------------------------------------------------------
    // PARALLEL Weight ROM Interface (3 weights for kernel positions k=0,1,2)
    //--------------------------------------------------------------------------
    reg  [10:0] weight_addr_base;
    wire [10:0] weight_addr_k0 = weight_addr_base;
    wire [10:0] weight_addr_k1 = weight_addr_base + 1;
    wire [10:0] weight_addr_k2 = weight_addr_base + 2;
    
    wire signed [WEIGHT_WIDTH-1:0] weight_k0, weight_k1, weight_k2;
    
    // 3 parallel weight ROM instances for simultaneous kernel access
    weight_rom #(.WEIGHT_WIDTH(WEIGHT_WIDTH), .DEPTH(2048), .ADDR_WIDTH(11))
        u_wrom_k0 (.clk(clk), .addr(weight_addr_k0), .data(weight_k0));
    weight_rom #(.WEIGHT_WIDTH(WEIGHT_WIDTH), .DEPTH(2048), .ADDR_WIDTH(11))
        u_wrom_k1 (.clk(clk), .addr(weight_addr_k1), .data(weight_k1));
    weight_rom #(.WEIGHT_WIDTH(WEIGHT_WIDTH), .DEPTH(2048), .ADDR_WIDTH(11))
        u_wrom_k2 (.clk(clk), .addr(weight_addr_k2), .data(weight_k2));
    
    reg  [5:0] bias_addr;
    wire signed [DATA_WIDTH-1:0] bias_data;
    
    bias_rom #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(64), .ADDR_WIDTH(6))
        u_bias_rom (.clk(clk), .addr(bias_addr), .data(bias_data));
    
    //--------------------------------------------------------------------------
    // PARALLEL MAC: 3 multipliers for kernel window
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] data_k0, data_k1, data_k2;
    
    // Parallel multipliers (synthesize to DSP48 on Xilinx)
    wire signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_k0 = data_k0 * $signed(weight_k0);
    wire signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_k1 = data_k1 * $signed(weight_k1);
    wire signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_k2 = data_k2 * $signed(weight_k2);
    
    // Single-cycle kernel sum (Q8.8 × Q1.7 = Q9.15, shift by 7 → Q8.8)
    wire signed [ACC_WIDTH-1:0] kernel_sum = (mult_k0 >>> 7) + (mult_k1 >>> 7) + (mult_k2 >>> 7);
    
    //--------------------------------------------------------------------------
    // Pipeline Registers (3-stage: Fetch → MAC → Accumulate)
    //--------------------------------------------------------------------------
    reg        pipe_s2_valid;
    reg [3:0]  pipe_s2_out_ch;
    reg [4:0]  pipe_s2_out_pos;
    reg        pipe_s2_last_in_ch;
    
    reg        pipe_s3_valid;
    reg [3:0]  pipe_s3_out_ch;
    reg [4:0]  pipe_s3_out_pos;
    reg        pipe_s3_last_in_ch;
    reg signed [ACC_WIDTH-1:0] pipe_s3_ksum;
    
    // Accumulator bank (per output channel)
    reg signed [ACC_WIDTH-1:0] accum [0:15];
    
    //--------------------------------------------------------------------------
    // Processing Counters
    //--------------------------------------------------------------------------
    reg [2:0] in_ch_cnt;
    reg [4:0] in_pos_cnt;
    reg [3:0] out_ch_cnt;
    reg [4:0] out_pos_cnt;
    reg [3:0] in_ch_iter;
    reg [2:0] pipe_flush;
    
    integer i, j;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= ST_IDLE;
        else
            state <= next_state;
    end
    
    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start)
                    next_state = ST_LOAD_IN;
            end
            ST_LOAD_IN: begin
                if (in_ch_cnt == IN_CH-1 && in_pos_cnt == FRAME_LEN-1 && valid_in)
                    next_state = ST_ENC1;
            end
            ST_ENC1: begin
                if (out_ch_cnt == ENC1_OUT_CH-1 && out_pos_cnt == ENC1_OUT_LEN-1 &&
                    in_ch_iter == IN_CH-1 && pipe_flush == 2)
                    next_state = ST_BNECK;
            end
            ST_BNECK: begin
                if (out_ch_cnt == BNECK_OUT_CH-1 && out_pos_cnt == BNECK_OUT_LEN-1 &&
                    in_ch_iter == ENC1_OUT_CH-1 && pipe_flush == 2)
                    next_state = ST_UPSAMPLE1;
            end
            ST_UPSAMPLE1: begin
                if (out_ch_cnt == BNECK_OUT_CH-1 && out_pos_cnt == BNECK_OUT_LEN-1)
                    next_state = ST_DEC1;
            end
            ST_DEC1: begin
                if (out_ch_cnt == DEC1_OUT_CH-1 && out_pos_cnt == DEC1_OUT_LEN-1 &&
                    in_ch_iter == BNECK_OUT_CH-1 && pipe_flush == 2)
                    next_state = ST_SKIP_ADD;
            end
            ST_SKIP_ADD: begin
                if (out_ch_cnt == DEC1_OUT_CH-1 && out_pos_cnt == DEC1_OUT_LEN-1)
                    next_state = ST_UPSAMPLE2;
            end
            ST_UPSAMPLE2: begin
                if (out_ch_cnt == DEC1_OUT_CH-1 && out_pos_cnt == DEC1_OUT_LEN-1)
                    next_state = ST_OUT_CONV;
            end
            ST_OUT_CONV: begin
                if (out_ch_cnt == OUT_CH-1 && out_pos_cnt == FRAME_LEN-1 &&
                    in_ch_iter == DEC1_OUT_CH-1 && pipe_flush == 2)
                    next_state = ST_TANH;
            end
            ST_TANH: begin
                if (out_ch_cnt == OUT_CH-1 && out_pos_cnt == FRAME_LEN-1)
                    next_state = ST_OUTPUT;
            end
            ST_OUTPUT: begin
                if (in_ch_cnt == OUT_CH-1 && in_pos_cnt == FRAME_LEN-1 && ready_out)
                    next_state = ST_DONE;
            end
            ST_DONE: next_state = ST_IDLE;
            default: next_state = ST_IDLE;
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Input Loading
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_ch_cnt <= 0;
            in_pos_cnt <= 0;
            for (i = 0; i < IN_CH; i = i + 1)
                for (j = 0; j < FRAME_LEN+2; j = j + 1)
                    input_buf[i][j] <= 0;
        end else if (state == ST_IDLE && start) begin
            in_ch_cnt <= 0;
            in_pos_cnt <= 0;
            for (i = 0; i < IN_CH; i = i + 1)
                for (j = 0; j < FRAME_LEN+2; j = j + 1)
                    input_buf[i][j] <= 0;
        end else if (state == ST_LOAD_IN && valid_in) begin
            input_buf[in_ch_cnt][in_pos_cnt + 1] <= data_in;
            
            if (in_pos_cnt == FRAME_LEN-1) begin
                in_pos_cnt <= 0;
                in_ch_cnt <= in_ch_cnt + 1;
            end else begin
                in_pos_cnt <= in_pos_cnt + 1;
            end
        end else if (state == ST_OUTPUT && ready_out) begin
            if (in_pos_cnt == FRAME_LEN-1) begin
                in_pos_cnt <= 0;
                in_ch_cnt <= in_ch_cnt + 1;
            end else begin
                in_pos_cnt <= in_pos_cnt + 1;
            end
        end
    end
    
    assign ready_in = (state == ST_LOAD_IN);
    
    //--------------------------------------------------------------------------
    // Pipelined Convolution Processing
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_ch_cnt <= 0;
            out_pos_cnt <= 0;
            in_ch_iter <= 0;
            pipe_flush <= 0;
            weight_addr_base <= 0;
            bias_addr <= 0;
            
            data_k0 <= 0; data_k1 <= 0; data_k2 <= 0;
            pipe_s2_valid <= 0;
            pipe_s3_valid <= 0;
            
            for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
            for (i = 0; i < ENC1_OUT_CH; i = i + 1) begin
                for (j = 0; j < ENC1_OUT_LEN+2; j = j + 1) enc1_buf[i][j] <= 0;
                for (j = 0; j < ENC1_OUT_LEN; j = j + 1) skip_buf[i][j] <= 0;
            end
            for (i = 0; i < BNECK_OUT_CH; i = i + 1) begin
                for (j = 0; j < BNECK_OUT_LEN; j = j + 1) bneck_buf[i][j] <= 0;
                for (j = 0; j < UP1_LEN+2; j = j + 1) up1_buf[i][j] <= 0;
            end
            for (i = 0; i < DEC1_OUT_CH; i = i + 1) begin
                for (j = 0; j < DEC1_OUT_LEN; j = j + 1) dec1_buf[i][j] <= 0;
                for (j = 0; j < FRAME_LEN; j = j + 1) up2_buf[i][j] <= 0;
            end
            for (i = 0; i < OUT_CH; i = i + 1)
                for (j = 0; j < FRAME_LEN; j = j + 1) out_buf[i][j] <= 0;
        end else begin
            case (state)
                ST_IDLE, ST_LOAD_IN: begin
                    out_ch_cnt <= 0;
                    out_pos_cnt <= 0;
                    in_ch_iter <= 0;
                    pipe_flush <= 0;
                    pipe_s2_valid <= 0;
                    pipe_s3_valid <= 0;
                    for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
                end
                
                //==============================================================
                // ENCODER 1: Pipelined Conv(2→4, k=3, s=2) + LeakyReLU
                //==============================================================
                ST_ENC1: begin
                    // Stage 1: Address + parallel data fetch
                    weight_addr_base <= WADDR_ENC1 + out_ch_cnt * (IN_CH * 3) + in_ch_iter * 3;
                    bias_addr <= BADDR_ENC1 + out_ch_cnt;
                    
                    // PARALLEL: Fetch 3 kernel positions at once
                    data_k0 <= input_buf[in_ch_iter][out_pos_cnt * 2 + 0];
                    data_k1 <= input_buf[in_ch_iter][out_pos_cnt * 2 + 1];
                    data_k2 <= input_buf[in_ch_iter][out_pos_cnt * 2 + 2];
                    
                    pipe_s2_valid <= 1'b1;
                    pipe_s2_out_ch <= out_ch_cnt;
                    pipe_s2_out_pos <= out_pos_cnt;
                    pipe_s2_last_in_ch <= (in_ch_iter == IN_CH-1);
                    
                    // Stage 2→3
                    pipe_s3_valid <= pipe_s2_valid;
                    pipe_s3_out_ch <= pipe_s2_out_ch;
                    pipe_s3_out_pos <= pipe_s2_out_pos;
                    pipe_s3_last_in_ch <= pipe_s2_last_in_ch;
                    pipe_s3_ksum <= kernel_sum;
                    
                    // Stage 3: Accumulate + store
                    if (pipe_s3_valid) begin
                        if (pipe_s3_last_in_ch) begin
                            begin : enc1_store
                                reg signed [ACC_WIDTH-1:0] sum;
                                reg signed [DATA_WIDTH-1:0] result;
                                sum = accum[pipe_s3_out_ch] + pipe_s3_ksum + 
                                      {{16{bias_data[15]}}, bias_data};
                                if (sum > 32'sh00007FFF) result = 16'sh7FFF;
                                else if (sum < 32'shFFFF8000) result = 16'sh8000;
                                else result = sum[15:0];
                                if (result[15])
                                    result = (result >>> 2) + (result >>> 4);
                                enc1_buf[pipe_s3_out_ch][pipe_s3_out_pos + 1] <= result;
                                skip_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= result;
                            end
                            accum[pipe_s3_out_ch] <= 0;
                        end else begin
                            accum[pipe_s3_out_ch] <= accum[pipe_s3_out_ch] + pipe_s3_ksum;
                        end
                    end
                    
                    // Counter advancement
                    if (in_ch_iter == IN_CH-1) begin
                        in_ch_iter <= 0;
                        if (out_pos_cnt == ENC1_OUT_LEN-1) begin
                            out_pos_cnt <= 0;
                            if (out_ch_cnt == ENC1_OUT_CH-1)
                                pipe_flush <= pipe_flush + 1;
                            else
                                out_ch_cnt <= out_ch_cnt + 1;
                        end else begin
                            out_pos_cnt <= out_pos_cnt + 1;
                        end
                    end else begin
                        in_ch_iter <= in_ch_iter + 1;
                    end
                end
                
                //==============================================================
                // BOTTLENECK: Pipelined Conv(4→8, k=3, s=2) + LeakyReLU
                //==============================================================
                ST_BNECK: begin
                    if (out_ch_cnt == 0 && out_pos_cnt == 0 && in_ch_iter == 0 && pipe_flush == 0) begin
                        pipe_s2_valid <= 0; pipe_s3_valid <= 0;
                        for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
                    end
                    
                    weight_addr_base <= WADDR_BNECK + out_ch_cnt * (ENC1_OUT_CH * 3) + in_ch_iter * 3;
                    bias_addr <= BADDR_BNECK + out_ch_cnt;
                    
                    data_k0 <= enc1_buf[in_ch_iter][out_pos_cnt * 2 + 0];
                    data_k1 <= enc1_buf[in_ch_iter][out_pos_cnt * 2 + 1];
                    data_k2 <= enc1_buf[in_ch_iter][out_pos_cnt * 2 + 2];
                    
                    pipe_s2_valid <= 1'b1;
                    pipe_s2_out_ch <= out_ch_cnt;
                    pipe_s2_out_pos <= out_pos_cnt;
                    pipe_s2_last_in_ch <= (in_ch_iter == ENC1_OUT_CH-1);
                    
                    pipe_s3_valid <= pipe_s2_valid;
                    pipe_s3_out_ch <= pipe_s2_out_ch;
                    pipe_s3_out_pos <= pipe_s2_out_pos;
                    pipe_s3_last_in_ch <= pipe_s2_last_in_ch;
                    pipe_s3_ksum <= kernel_sum;
                    
                    if (pipe_s3_valid) begin
                        if (pipe_s3_last_in_ch) begin
                            begin : bneck_store
                                reg signed [ACC_WIDTH-1:0] sum;
                                reg signed [DATA_WIDTH-1:0] result;
                                sum = accum[pipe_s3_out_ch] + pipe_s3_ksum + 
                                      {{16{bias_data[15]}}, bias_data};
                                if (sum > 32'sh00007FFF) result = 16'sh7FFF;
                                else if (sum < 32'shFFFF8000) result = 16'sh8000;
                                else result = sum[15:0];
                                if (result[15])
                                    result = (result >>> 2) + (result >>> 4);
                                bneck_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= result;
                            end
                            accum[pipe_s3_out_ch] <= 0;
                        end else begin
                            accum[pipe_s3_out_ch] <= accum[pipe_s3_out_ch] + pipe_s3_ksum;
                        end
                    end
                    
                    if (in_ch_iter == ENC1_OUT_CH-1) begin
                        in_ch_iter <= 0;
                        if (out_pos_cnt == BNECK_OUT_LEN-1) begin
                            out_pos_cnt <= 0;
                            if (out_ch_cnt == BNECK_OUT_CH-1)
                                pipe_flush <= pipe_flush + 1;
                            else
                                out_ch_cnt <= out_ch_cnt + 1;
                        end else begin
                            out_pos_cnt <= out_pos_cnt + 1;
                        end
                    end else begin
                        in_ch_iter <= in_ch_iter + 1;
                    end
                end
                
                //==============================================================
                // UPSAMPLE 1: Nearest neighbor 2x
                //==============================================================
                ST_UPSAMPLE1: begin
                    pipe_s2_valid <= 0; pipe_s3_valid <= 0; pipe_flush <= 0;
                    for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
                    
                    up1_buf[out_ch_cnt][out_pos_cnt*2 + 1] <= bneck_buf[out_ch_cnt][out_pos_cnt];
                    up1_buf[out_ch_cnt][out_pos_cnt*2 + 2] <= bneck_buf[out_ch_cnt][out_pos_cnt];
                    
                    if (out_pos_cnt == BNECK_OUT_LEN-1) begin
                        out_pos_cnt <= 0;
                        if (out_ch_cnt == BNECK_OUT_CH-1) out_ch_cnt <= 0;
                        else out_ch_cnt <= out_ch_cnt + 1;
                    end else begin
                        out_pos_cnt <= out_pos_cnt + 1;
                    end
                end
                
                //==============================================================
                // DECODER 1: Pipelined Conv(8→4, k=3, s=1) + LeakyReLU
                //==============================================================
                ST_DEC1: begin
                    weight_addr_base <= WADDR_DEC1 + out_ch_cnt * (BNECK_OUT_CH * 3) + in_ch_iter * 3;
                    bias_addr <= BADDR_DEC1 + out_ch_cnt;
                    
                    data_k0 <= up1_buf[in_ch_iter][out_pos_cnt + 0];
                    data_k1 <= up1_buf[in_ch_iter][out_pos_cnt + 1];
                    data_k2 <= up1_buf[in_ch_iter][out_pos_cnt + 2];
                    
                    pipe_s2_valid <= 1'b1;
                    pipe_s2_out_ch <= out_ch_cnt;
                    pipe_s2_out_pos <= out_pos_cnt;
                    pipe_s2_last_in_ch <= (in_ch_iter == BNECK_OUT_CH-1);
                    
                    pipe_s3_valid <= pipe_s2_valid;
                    pipe_s3_out_ch <= pipe_s2_out_ch;
                    pipe_s3_out_pos <= pipe_s2_out_pos;
                    pipe_s3_last_in_ch <= pipe_s2_last_in_ch;
                    pipe_s3_ksum <= kernel_sum;
                    
                    if (pipe_s3_valid) begin
                        if (pipe_s3_last_in_ch) begin
                            begin : dec1_store
                                reg signed [ACC_WIDTH-1:0] sum;
                                reg signed [DATA_WIDTH-1:0] result;
                                sum = accum[pipe_s3_out_ch] + pipe_s3_ksum + 
                                      {{16{bias_data[15]}}, bias_data};
                                if (sum > 32'sh00007FFF) result = 16'sh7FFF;
                                else if (sum < 32'shFFFF8000) result = 16'sh8000;
                                else result = sum[15:0];
                                if (result[15])
                                    result = (result >>> 2) + (result >>> 4);
                                dec1_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= result;
                            end
                            accum[pipe_s3_out_ch] <= 0;
                        end else begin
                            accum[pipe_s3_out_ch] <= accum[pipe_s3_out_ch] + pipe_s3_ksum;
                        end
                    end
                    
                    if (in_ch_iter == BNECK_OUT_CH-1) begin
                        in_ch_iter <= 0;
                        if (out_pos_cnt == DEC1_OUT_LEN-1) begin
                            out_pos_cnt <= 0;
                            if (out_ch_cnt == DEC1_OUT_CH-1)
                                pipe_flush <= pipe_flush + 1;
                            else
                                out_ch_cnt <= out_ch_cnt + 1;
                        end else begin
                            out_pos_cnt <= out_pos_cnt + 1;
                        end
                    end else begin
                        in_ch_iter <= in_ch_iter + 1;
                    end
                end
                
                //==============================================================
                // SKIP ADD: Element-wise addition
                //==============================================================
                ST_SKIP_ADD: begin
                    pipe_s2_valid <= 0; pipe_s3_valid <= 0; pipe_flush <= 0;
                    for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
                    
                    begin : skip_add_blk
                        reg signed [ACC_WIDTH-1:0] sum;
                        sum = {{16{dec1_buf[out_ch_cnt][out_pos_cnt][15]}}, 
                               dec1_buf[out_ch_cnt][out_pos_cnt]} + 
                              {{16{skip_buf[out_ch_cnt][out_pos_cnt][15]}}, 
                               skip_buf[out_ch_cnt][out_pos_cnt]};
                        if (sum > 32'sh00007FFF)
                            dec1_buf[out_ch_cnt][out_pos_cnt] <= 16'sh7FFF;
                        else if (sum < 32'shFFFF8000)
                            dec1_buf[out_ch_cnt][out_pos_cnt] <= 16'sh8000;
                        else
                            dec1_buf[out_ch_cnt][out_pos_cnt] <= sum[15:0];
                    end
                    
                    if (out_pos_cnt == DEC1_OUT_LEN-1) begin
                        out_pos_cnt <= 0;
                        if (out_ch_cnt == DEC1_OUT_CH-1) out_ch_cnt <= 0;
                        else out_ch_cnt <= out_ch_cnt + 1;
                    end else begin
                        out_pos_cnt <= out_pos_cnt + 1;
                    end
                end
                
                //==============================================================
                // UPSAMPLE 2: Nearest neighbor 2x
                //==============================================================
                ST_UPSAMPLE2: begin
                    up2_buf[out_ch_cnt][out_pos_cnt*2]     <= dec1_buf[out_ch_cnt][out_pos_cnt];
                    up2_buf[out_ch_cnt][out_pos_cnt*2 + 1] <= dec1_buf[out_ch_cnt][out_pos_cnt];
                    
                    if (out_pos_cnt == DEC1_OUT_LEN-1) begin
                        out_pos_cnt <= 0;
                        if (out_ch_cnt == DEC1_OUT_CH-1) out_ch_cnt <= 0;
                        else out_ch_cnt <= out_ch_cnt + 1;
                    end else begin
                        out_pos_cnt <= out_pos_cnt + 1;
                    end
                end
                
                //==============================================================
                // OUTPUT CONV: Conv(4→2, k=1, s=1)
                //==============================================================
                ST_OUT_CONV: begin
                    weight_addr_base <= WADDR_OUT + out_ch_cnt * DEC1_OUT_CH + in_ch_iter;
                    bias_addr <= BADDR_OUT + out_ch_cnt;
                    data_k0 <= up2_buf[in_ch_iter][out_pos_cnt];
                    
                    pipe_s2_valid <= 1'b1;
                    pipe_s2_out_ch <= out_ch_cnt;
                    pipe_s2_out_pos <= out_pos_cnt;
                    pipe_s2_last_in_ch <= (in_ch_iter == DEC1_OUT_CH-1);
                    
                    pipe_s3_valid <= pipe_s2_valid;
                    pipe_s3_out_ch <= pipe_s2_out_ch;
                    pipe_s3_out_pos <= pipe_s2_out_pos;
                    pipe_s3_last_in_ch <= pipe_s2_last_in_ch;
                    pipe_s3_ksum <= mult_k0 >>> 7;
                    
                    if (pipe_s3_valid) begin
                        if (pipe_s3_last_in_ch) begin
                            begin : out_store
                                reg signed [ACC_WIDTH-1:0] sum;
                                sum = accum[pipe_s3_out_ch] + pipe_s3_ksum + 
                                      {{16{bias_data[15]}}, bias_data};
                                if (sum > 32'sh00007FFF)
                                    out_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= 16'sh7FFF;
                                else if (sum < 32'shFFFF8000)
                                    out_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= 16'sh8000;
                                else
                                    out_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= sum[15:0];
                            end
                            accum[pipe_s3_out_ch] <= 0;
                        end else begin
                            accum[pipe_s3_out_ch] <= accum[pipe_s3_out_ch] + pipe_s3_ksum;
                        end
                    end
                    
                    if (in_ch_iter == DEC1_OUT_CH-1) begin
                        in_ch_iter <= 0;
                        if (out_pos_cnt == FRAME_LEN-1) begin
                            out_pos_cnt <= 0;
                            if (out_ch_cnt == OUT_CH-1)
                                pipe_flush <= pipe_flush + 1;
                            else
                                out_ch_cnt <= out_ch_cnt + 1;
                        end else begin
                            out_pos_cnt <= out_pos_cnt + 1;
                        end
                    end else begin
                        in_ch_iter <= in_ch_iter + 1;
                    end
                end
                
                //==============================================================
                // TANH: Simplified saturation
                //==============================================================
                ST_TANH: begin
                    pipe_s2_valid <= 0; pipe_s3_valid <= 0; pipe_flush <= 0;
                    
                    if (out_buf[out_ch_cnt][out_pos_cnt] > 16'sh0100)
                        out_buf[out_ch_cnt][out_pos_cnt] <= 16'sh00FF;
                    else if (out_buf[out_ch_cnt][out_pos_cnt] < -16'sh0100)
                        out_buf[out_ch_cnt][out_pos_cnt] <= 16'shFF01;
                    
                    if (out_pos_cnt == FRAME_LEN-1) begin
                        out_pos_cnt <= 0;
                        out_ch_cnt <= out_ch_cnt + 1;
                    end else begin
                        out_pos_cnt <= out_pos_cnt + 1;
                    end
                end
                
                default: ;
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Output
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
            valid_out <= 1'b0;
        end else if (state == ST_OUTPUT) begin
            data_out <= out_buf[in_ch_cnt][in_pos_cnt];
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
    
    //--------------------------------------------------------------------------
    // Status
    //--------------------------------------------------------------------------
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule
