//==============================================================================
// Mini Discriminator (Critic) for CWGAN-GP
//
// PARALLEL PIPELINED VERSION
//
// Architecture:
//   Input [4×16] → Conv1 [8×8] → Conv2 [16×4] → SumPool [16] → Dense → Score [1]
//
// Input: Concatenation of candidate signal (2ch) and condition signal (2ch)
//
// Parallel/Pipeline Features:
//   - PARALLEL KERNEL: 3 multipliers for k=0,1,2 computed in single cycle
//   - PIPELINED MAC: 3-stage pipeline (fetch → multiply → accumulate)
//   - PARALLEL POOLING: 4 channels summed per cycle
//   - LeakyReLU activations (no batch norm per WGAN-GP)
//
// Fixed-point: Q8.8 activations, Q1.7 weights, Q16.16 accumulator
//
// Resource Estimates (FPGA):
//   DSPs: 6 (3 kernel MACs × 2 stages)
//   BRAMs: 2 (weight + bias ROM)
//   FFs: ~2000
//   LUTs: ~3000
//==============================================================================

`timescale 1ns / 1ps

module discriminator_mini #(
    parameter DATA_WIDTH   = 16,           // Q8.8 activations
    parameter WEIGHT_WIDTH = 8,            // Q1.7 weights
    parameter ACC_WIDTH    = 32,           // Accumulator width
    parameter FRAME_LEN    = 16,           // Input frame length
    parameter IN_CH        = 4             // Input channels (2 candidate + 2 condition)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    
    // Candidate input (generated or real)
    input  wire signed [DATA_WIDTH-1:0] cand_in,
    input  wire                         cand_valid,
    
    // Condition input
    input  wire signed [DATA_WIDTH-1:0] cond_in,
    input  wire                         cond_valid,
    
    output wire                         ready_in,
    
    // Output: critic score (higher = more real)
    output reg  signed [DATA_WIDTH-1:0] score_out,
    output reg                          score_valid,
    
    // Status
    output wire                         busy,
    output wire                         done
);

    //--------------------------------------------------------------------------
    // Local Parameters
    //--------------------------------------------------------------------------
    localparam CONV1_OUT_CH  = 8;
    localparam CONV1_OUT_LEN = 8;    // 16/2
    localparam CONV2_OUT_CH  = 16;
    localparam CONV2_OUT_LEN = 4;    // 8/2
    
    // Weight ROM addresses (discriminator starts at 256)
    localparam WADDR_CONV1  = 256;   // 4*8*3 = 96 weights
    localparam WADDR_CONV2  = 352;   // 8*16*3 = 384 weights
    localparam WADDR_DENSE  = 736;   // 16 weights
    
    // Bias ROM addresses
    localparam BADDR_CONV1  = 32;
    localparam BADDR_CONV2  = 40;
    localparam BADDR_DENSE  = 56;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    localparam ST_IDLE      = 4'd0;
    localparam ST_LOAD_CAND = 4'd1;
    localparam ST_LOAD_COND = 4'd2;
    localparam ST_CONV1     = 4'd3;
    localparam ST_CONV2     = 4'd4;
    localparam ST_POOL      = 4'd5;
    localparam ST_DENSE     = 4'd6;
    localparam ST_OUTPUT    = 4'd7;
    localparam ST_DONE      = 4'd8;
    
    reg [3:0] state, next_state;
    
    //--------------------------------------------------------------------------
    // Buffers
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_buf [0:IN_CH-1][0:FRAME_LEN+1];
    reg signed [DATA_WIDTH-1:0] conv1_buf [0:CONV1_OUT_CH-1][0:CONV1_OUT_LEN+1];
    reg signed [DATA_WIDTH-1:0] conv2_buf [0:CONV2_OUT_CH-1][0:CONV2_OUT_LEN-1];
    reg signed [ACC_WIDTH-1:0]  pool_buf  [0:CONV2_OUT_CH-1];
    
    //--------------------------------------------------------------------------
    // PARALLEL Weight ROM Interface (3 weights for kernel)
    //--------------------------------------------------------------------------
    reg  [10:0] weight_addr_base;
    wire [10:0] weight_addr_k0 = weight_addr_base;
    wire [10:0] weight_addr_k1 = weight_addr_base + 1;
    wire [10:0] weight_addr_k2 = weight_addr_base + 2;
    
    wire signed [WEIGHT_WIDTH-1:0] weight_k0, weight_k1, weight_k2;
    
    // 3 parallel weight ROMs
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
    
    wire signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_k0 = data_k0 * $signed(weight_k0);
    wire signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_k1 = data_k1 * $signed(weight_k1);
    wire signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_k2 = data_k2 * $signed(weight_k2);
    
    wire signed [ACC_WIDTH-1:0] kernel_sum = (mult_k0 >>> 7) + (mult_k1 >>> 7) + (mult_k2 >>> 7);
    
    //--------------------------------------------------------------------------
    // Pipeline Registers
    //--------------------------------------------------------------------------
    reg        pipe_s2_valid;
    reg [4:0]  pipe_s2_out_ch;
    reg [4:0]  pipe_s2_out_pos;
    reg        pipe_s2_last_in_ch;
    
    reg        pipe_s3_valid;
    reg [4:0]  pipe_s3_out_ch;
    reg [4:0]  pipe_s3_out_pos;
    reg        pipe_s3_last_in_ch;
    reg signed [ACC_WIDTH-1:0] pipe_s3_ksum;
    
    // Accumulator bank
    reg signed [ACC_WIDTH-1:0] accum [0:15];
    
    // Dense accumulator
    reg signed [ACC_WIDTH-1:0] dense_acc;
    
    //--------------------------------------------------------------------------
    // Processing Counters
    //--------------------------------------------------------------------------
    reg [1:0]  load_ch_cnt;
    reg [4:0]  load_pos_cnt;
    reg [4:0]  out_ch_cnt;
    reg [4:0]  out_pos_cnt;
    reg [4:0]  in_ch_iter;
    reg [2:0]  pipe_flush;
    
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
                    next_state = ST_LOAD_CAND;
            end
            ST_LOAD_CAND: begin
                if (load_ch_cnt == 1 && load_pos_cnt == FRAME_LEN-1 && cand_valid)
                    next_state = ST_LOAD_COND;
            end
            ST_LOAD_COND: begin
                if (load_ch_cnt == 1 && load_pos_cnt == FRAME_LEN-1 && cond_valid)
                    next_state = ST_CONV1;
            end
            ST_CONV1: begin
                if (out_ch_cnt == CONV1_OUT_CH-1 && out_pos_cnt == CONV1_OUT_LEN-1 &&
                    in_ch_iter == IN_CH-1 && pipe_flush == 2)
                    next_state = ST_CONV2;
            end
            ST_CONV2: begin
                if (out_ch_cnt == CONV2_OUT_CH-1 && out_pos_cnt == CONV2_OUT_LEN-1 &&
                    in_ch_iter == CONV1_OUT_CH-1 && pipe_flush == 2)
                    next_state = ST_POOL;
            end
            ST_POOL: begin
                if (out_ch_cnt == CONV2_OUT_CH-1 && out_pos_cnt == CONV2_OUT_LEN-1)
                    next_state = ST_DENSE;
            end
            ST_DENSE: begin
                if (out_ch_cnt == CONV2_OUT_CH-1 && pipe_flush == 2)
                    next_state = ST_OUTPUT;
            end
            ST_OUTPUT: begin
                next_state = ST_DONE;
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Input Loading
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_ch_cnt <= 0;
            load_pos_cnt <= 0;
            for (i = 0; i < IN_CH; i = i + 1)
                for (j = 0; j < FRAME_LEN+2; j = j + 1)
                    input_buf[i][j] <= 0;
        end else if (state == ST_IDLE && start) begin
            load_ch_cnt <= 0;
            load_pos_cnt <= 0;
            for (i = 0; i < IN_CH; i = i + 1)
                for (j = 0; j < FRAME_LEN+2; j = j + 1)
                    input_buf[i][j] <= 0;
        end else if (state == ST_LOAD_CAND && cand_valid) begin
            input_buf[load_ch_cnt][load_pos_cnt + 1] <= cand_in;
            
            if (load_pos_cnt == FRAME_LEN-1) begin
                load_pos_cnt <= 0;
                if (load_ch_cnt == 1)
                    load_ch_cnt <= 0;
                else
                    load_ch_cnt <= load_ch_cnt + 1;
            end else begin
                load_pos_cnt <= load_pos_cnt + 1;
            end
        end else if (state == ST_LOAD_COND && cond_valid) begin
            input_buf[load_ch_cnt + 2][load_pos_cnt + 1] <= cond_in;
            
            if (load_pos_cnt == FRAME_LEN-1) begin
                load_pos_cnt <= 0;
                load_ch_cnt <= load_ch_cnt + 1;
            end else begin
                load_pos_cnt <= load_pos_cnt + 1;
            end
        end
    end
    
    assign ready_in = (state == ST_LOAD_CAND) || (state == ST_LOAD_COND);
    
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
            dense_acc <= 0;
            
            data_k0 <= 0; data_k1 <= 0; data_k2 <= 0;
            pipe_s2_valid <= 0;
            pipe_s3_valid <= 0;
            
            for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
            for (i = 0; i < CONV1_OUT_CH; i = i + 1)
                for (j = 0; j < CONV1_OUT_LEN+2; j = j + 1) conv1_buf[i][j] <= 0;
            for (i = 0; i < CONV2_OUT_CH; i = i + 1) begin
                for (j = 0; j < CONV2_OUT_LEN; j = j + 1) conv2_buf[i][j] <= 0;
                pool_buf[i] <= 0;
            end
        end else begin
            case (state)
                ST_IDLE, ST_LOAD_CAND, ST_LOAD_COND: begin
                    out_ch_cnt <= 0;
                    out_pos_cnt <= 0;
                    in_ch_iter <= 0;
                    pipe_flush <= 0;
                    pipe_s2_valid <= 0;
                    pipe_s3_valid <= 0;
                    dense_acc <= 0;
                    for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
                    for (i = 0; i < CONV2_OUT_CH; i = i + 1) pool_buf[i] <= 0;
                end
                
                //==============================================================
                // CONV1: Pipelined Conv(4→8, k=3, s=2) + LeakyReLU
                //==============================================================
                ST_CONV1: begin
                    // Stage 1: Address + data fetch
                    weight_addr_base <= WADDR_CONV1 + out_ch_cnt * (IN_CH * 3) + in_ch_iter * 3;
                    bias_addr <= BADDR_CONV1 + out_ch_cnt;
                    
                    // PARALLEL: 3 kernel positions
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
                            begin : conv1_store
                                reg signed [ACC_WIDTH-1:0] sum;
                                reg signed [DATA_WIDTH-1:0] result;
                                sum = accum[pipe_s3_out_ch] + pipe_s3_ksum + 
                                      {{16{bias_data[15]}}, bias_data};
                                if (sum > 32'sh00007FFF) result = 16'sh7FFF;
                                else if (sum < 32'shFFFF8000) result = 16'sh8000;
                                else result = sum[15:0];
                                // LeakyReLU
                                if (result[15])
                                    result = (result >>> 2) + (result >>> 4);
                                conv1_buf[pipe_s3_out_ch][pipe_s3_out_pos + 1] <= result;
                            end
                            accum[pipe_s3_out_ch] <= 0;
                        end else begin
                            accum[pipe_s3_out_ch] <= accum[pipe_s3_out_ch] + pipe_s3_ksum;
                        end
                    end
                    
                    // Counter advancement
                    if (in_ch_iter == IN_CH-1) begin
                        in_ch_iter <= 0;
                        if (out_pos_cnt == CONV1_OUT_LEN-1) begin
                            out_pos_cnt <= 0;
                            if (out_ch_cnt == CONV1_OUT_CH-1)
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
                // CONV2: Pipelined Conv(8→16, k=3, s=2) + LeakyReLU
                //==============================================================
                ST_CONV2: begin
                    if (out_ch_cnt == 0 && out_pos_cnt == 0 && in_ch_iter == 0 && pipe_flush == 0) begin
                        pipe_s2_valid <= 0; pipe_s3_valid <= 0;
                        for (i = 0; i < 16; i = i + 1) accum[i] <= 0;
                    end
                    
                    weight_addr_base <= WADDR_CONV2 + out_ch_cnt * (CONV1_OUT_CH * 3) + in_ch_iter * 3;
                    bias_addr <= BADDR_CONV2 + out_ch_cnt;
                    
                    data_k0 <= conv1_buf[in_ch_iter][out_pos_cnt * 2 + 0];
                    data_k1 <= conv1_buf[in_ch_iter][out_pos_cnt * 2 + 1];
                    data_k2 <= conv1_buf[in_ch_iter][out_pos_cnt * 2 + 2];
                    
                    pipe_s2_valid <= 1'b1;
                    pipe_s2_out_ch <= out_ch_cnt;
                    pipe_s2_out_pos <= out_pos_cnt;
                    pipe_s2_last_in_ch <= (in_ch_iter == CONV1_OUT_CH-1);
                    
                    pipe_s3_valid <= pipe_s2_valid;
                    pipe_s3_out_ch <= pipe_s2_out_ch;
                    pipe_s3_out_pos <= pipe_s2_out_pos;
                    pipe_s3_last_in_ch <= pipe_s2_last_in_ch;
                    pipe_s3_ksum <= kernel_sum;
                    
                    if (pipe_s3_valid) begin
                        if (pipe_s3_last_in_ch) begin
                            begin : conv2_store
                                reg signed [ACC_WIDTH-1:0] sum;
                                reg signed [DATA_WIDTH-1:0] result;
                                sum = accum[pipe_s3_out_ch] + pipe_s3_ksum + 
                                      {{16{bias_data[15]}}, bias_data};
                                if (sum > 32'sh00007FFF) result = 16'sh7FFF;
                                else if (sum < 32'shFFFF8000) result = 16'sh8000;
                                else result = sum[15:0];
                                if (result[15])
                                    result = (result >>> 2) + (result >>> 4);
                                conv2_buf[pipe_s3_out_ch][pipe_s3_out_pos] <= result;
                            end
                            accum[pipe_s3_out_ch] <= 0;
                        end else begin
                            accum[pipe_s3_out_ch] <= accum[pipe_s3_out_ch] + pipe_s3_ksum;
                        end
                    end
                    
                    if (in_ch_iter == CONV1_OUT_CH-1) begin
                        in_ch_iter <= 0;
                        if (out_pos_cnt == CONV2_OUT_LEN-1) begin
                            out_pos_cnt <= 0;
                            if (out_ch_cnt == CONV2_OUT_CH-1)
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
                // GLOBAL SUM POOLING: Parallel accumulation
                //==============================================================
                ST_POOL: begin
                    pipe_s2_valid <= 0; pipe_s3_valid <= 0; pipe_flush <= 0;
                    
                    // Accumulate spatial positions
                    pool_buf[out_ch_cnt] <= pool_buf[out_ch_cnt] + 
                        {{16{conv2_buf[out_ch_cnt][out_pos_cnt][15]}}, 
                         conv2_buf[out_ch_cnt][out_pos_cnt]};
                    
                    if (out_pos_cnt == CONV2_OUT_LEN-1) begin
                        out_pos_cnt <= 0;
                        if (out_ch_cnt == CONV2_OUT_CH-1)
                            out_ch_cnt <= 0;
                        else
                            out_ch_cnt <= out_ch_cnt + 1;
                    end else begin
                        out_pos_cnt <= out_pos_cnt + 1;
                    end
                end
                
                //==============================================================
                // DENSE: Pipelined Linear(16→1)
                //==============================================================
                ST_DENSE: begin
                    weight_addr_base <= WADDR_DENSE + out_ch_cnt;
                    bias_addr <= BADDR_DENSE;
                    
                    // Fetch pooled value
                    data_k0 <= pool_buf[out_ch_cnt][15:0];
                    
                    pipe_s2_valid <= 1'b1;
                    pipe_s2_out_ch <= out_ch_cnt;
                    pipe_s2_last_in_ch <= (out_ch_cnt == CONV2_OUT_CH-1);
                    
                    pipe_s3_valid <= pipe_s2_valid;
                    pipe_s3_out_ch <= pipe_s2_out_ch;
                    pipe_s3_last_in_ch <= pipe_s2_last_in_ch;
                    pipe_s3_ksum <= mult_k0 >>> 7;
                    
                    if (pipe_s3_valid) begin
                        dense_acc <= dense_acc + pipe_s3_ksum;
                        if (pipe_s3_last_in_ch) begin
                            // Add bias on last
                            dense_acc <= dense_acc + pipe_s3_ksum + 
                                        {{16{bias_data[15]}}, bias_data};
                        end
                    end
                    
                    if (out_ch_cnt == CONV2_OUT_CH-1)
                        pipe_flush <= pipe_flush + 1;
                    else
                        out_ch_cnt <= out_ch_cnt + 1;
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
            score_out <= 0;
            score_valid <= 1'b0;
        end else if (state == ST_OUTPUT) begin
            // Saturate dense_acc to 16-bit
            if (dense_acc > $signed(32'h00007FFF))
                score_out <= 16'h7FFF;
            else if (dense_acc < $signed(32'hFFFF8000))
                score_out <= 16'h8000;
            else
                score_out <= dense_acc[DATA_WIDTH-1:0];
            score_valid <= 1'b1;
        end else begin
            score_valid <= 1'b0;
        end
    end
    
    //--------------------------------------------------------------------------
    // Status
    //--------------------------------------------------------------------------
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule
