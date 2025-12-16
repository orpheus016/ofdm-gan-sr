//==============================================================================
// CWGAN-GP for OFDM Signal Reconstruction - Conv1D Engine
//
// Pipelined 1D Convolution with Parallel MACs
//
// Architecture: Fully pipelined with configurable parallelism
// - Supports variable kernel size (1, 3, 5, 7)
// - Configurable input/output channels
// - Optional stride (1 or 2)
// - Bias addition
//
// Fixed-Point: Q8.8 activations, Q1.7 weights, Q16.16 accumulator
//==============================================================================

`timescale 1ns / 1ps

module conv1d_engine#(
    parameter DATA_WIDTH    = 16,          // Activation bits (Q8.8)
    parameter WEIGHT_WIDTH  = 8,           // Weight bits (Q1.7)
    parameter ACC_WIDTH     = 32,          // Accumulator bits (Q16.16)
    parameter FRAME_LEN     = 16,          // Input frame length
    parameter IN_CH         = 2,           // Input channels
    parameter OUT_CH        = 4,           // Output channels
    parameter KERNEL_SIZE   = 3,           // Convolution kernel size
    parameter STRIDE        = 1,           // Stride (1 or 2)
    parameter PADDING       = 1,           // Zero padding on each side
    parameter NUM_MACS      = 4            // Parallel MAC units
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    
    // Input data interface (stream)
    input  wire [DATA_WIDTH-1:0]    data_in,
    input  wire                     data_valid,
    output wire                     data_ready,
    
    // Weight ROM interface
    output wire [$clog2(IN_CH*OUT_CH*KERNEL_SIZE)-1:0] weight_addr,
    input  wire [WEIGHT_WIDTH-1:0]  weight_data,
    
    // Bias ROM interface
    output wire [$clog2(OUT_CH)-1:0] bias_addr,
    input  wire [DATA_WIDTH-1:0]    bias_data,
    
    // Output data interface
    output reg  [DATA_WIDTH-1:0]    data_out,
    output reg                      data_out_valid,
    input  wire                     data_out_ready,
    
    // Status
    output wire                     busy,
    output wire                     done
);

    //--------------------------------------------------------------------------
    // Local Parameters
    //--------------------------------------------------------------------------
    localparam OUT_LEN = (FRAME_LEN + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    localparam WEIGHT_COUNT = IN_CH * OUT_CH * KERNEL_SIZE;
    localparam TOTAL_OPS = OUT_CH * OUT_LEN * IN_CH * KERNEL_SIZE;
    
    // Pipeline stages
    localparam PIPE_STAGES = 4;  // data_fetch, weight_fetch, multiply, accumulate
    
    //--------------------------------------------------------------------------
    // Internal Signals
    //--------------------------------------------------------------------------
    
    // Input buffer (holds padded frame)
    reg signed [DATA_WIDTH-1:0] input_buffer [0:IN_CH-1][0:FRAME_LEN+2*PADDING-1];
    
    // Output buffer
    reg signed [DATA_WIDTH-1:0] output_buffer [0:OUT_CH-1][0:OUT_LEN-1];
    
    // State machine
    localparam ST_IDLE   = 3'd0;
    localparam ST_LOAD   = 3'd1;
    localparam ST_CONV   = 3'd2;
    localparam ST_BIAS   = 3'd3;
    localparam ST_OUTPUT = 3'd4;
    localparam ST_DONE   = 3'd5;
    
    reg [2:0] state, next_state;
    
    // Counters
    reg [$clog2(IN_CH)-1:0]        in_ch_cnt;
    reg [$clog2(OUT_CH)-1:0]       out_ch_cnt;
    reg [$clog2(FRAME_LEN+1)-1:0]  pos_cnt;
    reg [$clog2(KERNEL_SIZE)-1:0]  kern_cnt;
    reg [$clog2(OUT_LEN)-1:0]      out_pos_cnt;
    
    // Loading counters
    reg [$clog2(IN_CH)-1:0]        load_ch_cnt;
    reg [$clog2(FRAME_LEN)-1:0]    load_pos_cnt;
    
    // MAC pipeline
    reg signed [DATA_WIDTH-1:0]   pipe_data   [0:PIPE_STAGES-1];
    reg signed [WEIGHT_WIDTH-1:0] pipe_weight [0:PIPE_STAGES-1];
    reg signed [2*DATA_WIDTH-1:0] pipe_mult   [0:PIPE_STAGES-1];
    reg        [PIPE_STAGES-1:0]  pipe_valid;
    
    // Accumulator
    reg signed [ACC_WIDTH-1:0] accumulator [0:OUT_CH-1];
    
    // Weight address calculation
    wire [$clog2(WEIGHT_COUNT)-1:0] current_weight_addr;
    assign current_weight_addr = out_ch_cnt * (IN_CH * KERNEL_SIZE) + 
                                 in_ch_cnt * KERNEL_SIZE + 
                                 kern_cnt;
    assign weight_addr = current_weight_addr;
    assign bias_addr = out_ch_cnt;
    
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
                    next_state = ST_LOAD;
            end
            ST_LOAD: begin
                if (load_ch_cnt == IN_CH-1 && load_pos_cnt == FRAME_LEN-1 && data_valid)
                    next_state = ST_CONV;
            end
            ST_CONV: begin
                if (out_ch_cnt == OUT_CH-1 && 
                    out_pos_cnt == OUT_LEN-1 && 
                    in_ch_cnt == IN_CH-1 && 
                    kern_cnt == KERNEL_SIZE-1)
                    next_state = ST_BIAS;
            end
            ST_BIAS: begin
                if (out_ch_cnt == OUT_CH-1)
                    next_state = ST_OUTPUT;
            end
            ST_OUTPUT: begin
                if (out_ch_cnt == OUT_CH-1 && out_pos_cnt == OUT_LEN-1 && data_out_ready)
                    next_state = ST_DONE;
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Data Loading
    //--------------------------------------------------------------------------
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_ch_cnt <= 0;
            load_pos_cnt <= 0;
            // Initialize input buffer with zeros (padding)
            for (i = 0; i < IN_CH; i = i + 1)
                for (j = 0; j < FRAME_LEN + 2*PADDING; j = j + 1)
                    input_buffer[i][j] <= 0;
        end else if (state == ST_IDLE && start) begin
            load_ch_cnt <= 0;
            load_pos_cnt <= 0;
        end else if (state == ST_LOAD && data_valid) begin
            // Store data with padding offset
            input_buffer[load_ch_cnt][load_pos_cnt + PADDING] <= data_in;
            
            if (load_pos_cnt == FRAME_LEN-1) begin
                load_pos_cnt <= 0;
                load_ch_cnt <= load_ch_cnt + 1;
            end else begin
                load_pos_cnt <= load_pos_cnt + 1;
            end
        end
    end
    
    assign data_ready = (state == ST_LOAD);
    
    //--------------------------------------------------------------------------
    // Convolution Engine - Nested Loop Iteration
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_ch_cnt  <= 0;
            out_pos_cnt <= 0;
            in_ch_cnt   <= 0;
            kern_cnt    <= 0;
        end else if (state == ST_IDLE) begin
            out_ch_cnt  <= 0;
            out_pos_cnt <= 0;
            in_ch_cnt   <= 0;
            kern_cnt    <= 0;
        end else if (state == ST_CONV) begin
            // Iterate: out_ch -> out_pos -> in_ch -> kernel
            if (kern_cnt == KERNEL_SIZE-1) begin
                kern_cnt <= 0;
                if (in_ch_cnt == IN_CH-1) begin
                    in_ch_cnt <= 0;
                    if (out_pos_cnt == OUT_LEN-1) begin
                        out_pos_cnt <= 0;
                        if (out_ch_cnt < OUT_CH-1)
                            out_ch_cnt <= out_ch_cnt + 1;
                    end else begin
                        out_pos_cnt <= out_pos_cnt + 1;
                    end
                end else begin
                    in_ch_cnt <= in_ch_cnt + 1;
                end
            end else begin
                kern_cnt <= kern_cnt + 1;
            end
        end else if (state == ST_BIAS) begin
            if (out_ch_cnt < OUT_CH-1)
                out_ch_cnt <= out_ch_cnt + 1;
            else
                out_ch_cnt <= 0;
        end else if (state == ST_OUTPUT) begin
            if (data_out_ready) begin
                if (out_pos_cnt == OUT_LEN-1) begin
                    out_pos_cnt <= 0;
                    out_ch_cnt <= out_ch_cnt + 1;
                end else begin
                    out_pos_cnt <= out_pos_cnt + 1;
                end
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // MAC Pipeline
    //--------------------------------------------------------------------------
    // Stage 0: Fetch data from input buffer
    wire [$clog2(FRAME_LEN+2*PADDING)-1:0] data_idx;
    assign data_idx = out_pos_cnt * STRIDE + kern_cnt;
    
    always @(posedge clk) begin
        if (state == ST_CONV) begin
            pipe_data[0] <= input_buffer[in_ch_cnt][data_idx];
            pipe_valid[0] <= 1'b1;
        end else begin
            pipe_valid[0] <= 1'b0;
        end
    end
    
    // Stage 1: Fetch weight (arrives from ROM)
    always @(posedge clk) begin
        pipe_data[1] <= pipe_data[0];
        pipe_weight[1] <= weight_data;
        pipe_valid[1] <= pipe_valid[0];
    end
    
    // Stage 2: Multiply
    always @(posedge clk) begin
        if (pipe_valid[1])
            pipe_mult[2] <= $signed(pipe_data[1]) * $signed(pipe_weight[1]);
        else
            pipe_mult[2] <= 0;
        pipe_valid[2] <= pipe_valid[1];
    end
    
    // Stage 3: Accumulate (extend to 32-bit)
    wire signed [ACC_WIDTH-1:0] mult_extended;
    assign mult_extended = {{(ACC_WIDTH-2*DATA_WIDTH){pipe_mult[2][2*DATA_WIDTH-1]}}, pipe_mult[2]};
    
    // Determine when to reset accumulator (new output position)
    reg new_output_pos;
    always @(posedge clk) begin
        new_output_pos <= (state == ST_CONV && in_ch_cnt == 0 && kern_cnt == 0);
    end
    
    // Accumulator array - one per output channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < OUT_CH; i = i + 1)
                accumulator[i] <= 0;
        end else if (state == ST_CONV && pipe_valid[2]) begin
            // Shift right by 7 (weight is Q1.7) to keep Q8.8
            if (new_output_pos)
                accumulator[out_ch_cnt] <= mult_extended >>> 7;
            else
                accumulator[out_ch_cnt] <= accumulator[out_ch_cnt] + (mult_extended >>> 7);
        end
    end
    
    //--------------------------------------------------------------------------
    // Bias Addition & Output Buffer Write
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (state == ST_BIAS) begin
            // Write to output buffer with bias (bias is Q8.8)
            for (i = 0; i < OUT_LEN; i = i + 1) begin
                output_buffer[out_ch_cnt][i] <= 
                    // Saturate to 16-bit
                    (accumulator[out_ch_cnt] + $signed(bias_data) > 32'sh00007FFF) ? 16'sh7FFF :
                    (accumulator[out_ch_cnt] + $signed(bias_data) < 32'sh8000FFFF) ? 16'sh8000 :
                    accumulator[out_ch_cnt][DATA_WIDTH-1:0] + bias_data;
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Interface
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
            data_out_valid <= 1'b0;
        end else if (state == ST_OUTPUT) begin
            data_out <= output_buffer[out_ch_cnt][out_pos_cnt];
            data_out_valid <= 1'b1;
        end else begin
            data_out_valid <= 1'b0;
        end
    end
    
    //--------------------------------------------------------------------------
    // Status Signals
    //--------------------------------------------------------------------------
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule
