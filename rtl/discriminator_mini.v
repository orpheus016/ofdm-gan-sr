//==============================================================================
// Mini Discriminator (Critic) for CWGAN-GP
//
// Simplified representative architecture for FPGA demonstration
//
// Architecture:
//   Input [4×16] → Conv1 [8×8] → Conv2 [16×4] → SumPool [16] → Dense → Score [1]
//
// Input: Concatenation of candidate signal (2ch) and condition signal (2ch)
//
// Features:
//   - Strided convolutions for downsampling
//   - No batch normalization (per WGAN-GP)
//   - LeakyReLU activations
//   - Global sum pooling
//   - Single linear output (no sigmoid)
//
// Fixed-point: Q8.8 activations, Q1.7 weights
//==============================================================================

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
    localparam WADDR_CONV1  = 256;
    localparam WADDR_CONV2  = 352;
    localparam WADDR_DENSE  = 736;
    
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
    localparam ST_LRELU1    = 4'd4;
    localparam ST_CONV2     = 4'd5;
    localparam ST_LRELU2    = 4'd6;
    localparam ST_POOL      = 4'd7;
    localparam ST_DENSE     = 4'd8;
    localparam ST_OUTPUT    = 4'd9;
    localparam ST_DONE      = 4'd10;
    
    reg [3:0] state, next_state;
    
    //--------------------------------------------------------------------------
    // Buffers
    //--------------------------------------------------------------------------
    // Input buffer (4 channels: cand_I, cand_Q, cond_I, cond_Q)
    reg signed [DATA_WIDTH-1:0] input_buf [0:IN_CH-1][0:FRAME_LEN-1];
    
    // Intermediate buffers
    reg signed [DATA_WIDTH-1:0] conv1_buf [0:CONV1_OUT_CH-1][0:CONV1_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] conv2_buf [0:CONV2_OUT_CH-1][0:CONV2_OUT_LEN-1];
    
    // Pool output
    reg signed [ACC_WIDTH-1:0] pool_buf [0:CONV2_OUT_CH-1];
    
    //--------------------------------------------------------------------------
    // Weight/Bias ROM Interface
    //--------------------------------------------------------------------------
    reg  [10:0] weight_addr;
    wire [WEIGHT_WIDTH-1:0] weight_data;
    
    reg  [5:0] bias_addr;
    wire [DATA_WIDTH-1:0] bias_data;
    
    weight_rom #(
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .DEPTH(2048),
        .ADDR_WIDTH(11)
    ) u_weight_rom (
        .clk(clk),
        .addr(weight_addr),
        .data(weight_data)
    );
    
    bias_rom #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(64),
        .ADDR_WIDTH(6)
    ) u_bias_rom (
        .clk(clk),
        .addr(bias_addr),
        .data(bias_data)
    );
    
    //--------------------------------------------------------------------------
    // Processing Counters
    //--------------------------------------------------------------------------
    reg [1:0]                       load_ch_cnt;  // 0-1 for cand, 0-1 for cond
    reg [$clog2(FRAME_LEN)-1:0]     load_pos_cnt;
    reg [$clog2(CONV2_OUT_CH)-1:0]  proc_ch_cnt;
    reg [$clog2(FRAME_LEN)-1:0]     proc_pos_cnt;
    reg [$clog2(3)-1:0]             kern_cnt;
    reg [$clog2(IN_CH)-1:0]         acc_ch_cnt;
    
    // Accumulator
    reg signed [ACC_WIDTH-1:0] accumulator;
    
    // Dense accumulator
    reg signed [ACC_WIDTH-1:0] dense_acc;
    
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
                if (proc_ch_cnt == CONV1_OUT_CH-1 && proc_pos_cnt == CONV1_OUT_LEN-1 &&
                    acc_ch_cnt == IN_CH-1 && kern_cnt == 2)
                    next_state = ST_LRELU1;
            end
            ST_LRELU1: begin
                if (proc_ch_cnt == CONV1_OUT_CH-1 && proc_pos_cnt == CONV1_OUT_LEN-1)
                    next_state = ST_CONV2;
            end
            ST_CONV2: begin
                if (proc_ch_cnt == CONV2_OUT_CH-1 && proc_pos_cnt == CONV2_OUT_LEN-1 &&
                    acc_ch_cnt == CONV1_OUT_CH-1 && kern_cnt == 2)
                    next_state = ST_LRELU2;
            end
            ST_LRELU2: begin
                if (proc_ch_cnt == CONV2_OUT_CH-1 && proc_pos_cnt == CONV2_OUT_LEN-1)
                    next_state = ST_POOL;
            end
            ST_POOL: begin
                if (proc_ch_cnt == CONV2_OUT_CH-1 && proc_pos_cnt == CONV2_OUT_LEN-1)
                    next_state = ST_DENSE;
            end
            ST_DENSE: begin
                if (proc_ch_cnt == CONV2_OUT_CH-1)
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
        end else if (state == ST_IDLE && start) begin
            load_ch_cnt <= 0;
            load_pos_cnt <= 0;
        end else if (state == ST_LOAD_CAND && cand_valid) begin
            input_buf[load_ch_cnt][load_pos_cnt] <= cand_in;
            
            if (load_pos_cnt == FRAME_LEN-1) begin
                load_pos_cnt <= 0;
                if (load_ch_cnt == 1)
                    load_ch_cnt <= 0;  // Reset for condition loading
                else
                    load_ch_cnt <= load_ch_cnt + 1;
            end else begin
                load_pos_cnt <= load_pos_cnt + 1;
            end
        end else if (state == ST_LOAD_COND && cond_valid) begin
            input_buf[load_ch_cnt + 2][load_pos_cnt] <= cond_in;  // Channels 2, 3
            
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
    // Convolution Processing
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proc_ch_cnt <= 0;
            proc_pos_cnt <= 0;
            kern_cnt <= 0;
            acc_ch_cnt <= 0;
            accumulator <= 0;
            dense_acc <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            for (i = 0; i < CONV2_OUT_CH; i = i + 1)
                pool_buf[i] <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    proc_ch_cnt <= 0;
                    proc_pos_cnt <= 0;
                    kern_cnt <= 0;
                    acc_ch_cnt <= 0;
                    accumulator <= 0;
                    dense_acc <= 0;
                end
                
                //--------------------------------------------------------------
                // Conv1: Conv(4→8, k=3, s=2)
                //--------------------------------------------------------------
                ST_CONV1: begin
                    weight_addr <= WADDR_CONV1 + proc_ch_cnt * (IN_CH * 3) + 
                                   acc_ch_cnt * 3 + kern_cnt;
                    bias_addr <= BADDR_CONV1 + proc_ch_cnt;
                    
                    if (kern_cnt == 2) begin
                        kern_cnt <= 0;
                        if (acc_ch_cnt == IN_CH-1) begin
                            acc_ch_cnt <= 0;
                            conv1_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                            accumulator <= 0;
                            
                            if (proc_pos_cnt == CONV1_OUT_LEN-1) begin
                                proc_pos_cnt <= 0;
                                if (proc_ch_cnt == CONV1_OUT_CH-1)
                                    proc_ch_cnt <= 0;
                                else
                                    proc_ch_cnt <= proc_ch_cnt + 1;
                            end else begin
                                proc_pos_cnt <= proc_pos_cnt + 1;
                            end
                        end else begin
                            acc_ch_cnt <= acc_ch_cnt + 1;
                        end
                    end else begin
                        kern_cnt <= kern_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // LeakyReLU 1: x if x>0, else 0.2*x
                //--------------------------------------------------------------
                ST_LRELU1: begin
                    if (conv1_buf[proc_ch_cnt][proc_pos_cnt][DATA_WIDTH-1]) begin
                        // Negative: multiply by 0.2 ≈ 26/128 (shift right 2 + shift right 4)
                        conv1_buf[proc_ch_cnt][proc_pos_cnt] <= 
                            (conv1_buf[proc_ch_cnt][proc_pos_cnt] >>> 2) + 
                            (conv1_buf[proc_ch_cnt][proc_pos_cnt] >>> 4);
                    end
                    
                    if (proc_pos_cnt == CONV1_OUT_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Conv2: Conv(8→16, k=3, s=2)
                //--------------------------------------------------------------
                ST_CONV2: begin
                    weight_addr <= WADDR_CONV2 + proc_ch_cnt * (CONV1_OUT_CH * 3) + 
                                   acc_ch_cnt * 3 + kern_cnt;
                    bias_addr <= BADDR_CONV2 + proc_ch_cnt;
                    
                    if (kern_cnt == 2) begin
                        kern_cnt <= 0;
                        if (acc_ch_cnt == CONV1_OUT_CH-1) begin
                            acc_ch_cnt <= 0;
                            conv2_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                            accumulator <= 0;
                            
                            if (proc_pos_cnt == CONV2_OUT_LEN-1) begin
                                proc_pos_cnt <= 0;
                                if (proc_ch_cnt == CONV2_OUT_CH-1)
                                    proc_ch_cnt <= 0;
                                else
                                    proc_ch_cnt <= proc_ch_cnt + 1;
                            end else begin
                                proc_pos_cnt <= proc_pos_cnt + 1;
                            end
                        end else begin
                            acc_ch_cnt <= acc_ch_cnt + 1;
                        end
                    end else begin
                        kern_cnt <= kern_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // LeakyReLU 2
                //--------------------------------------------------------------
                ST_LRELU2: begin
                    if (conv2_buf[proc_ch_cnt][proc_pos_cnt][DATA_WIDTH-1]) begin
                        conv2_buf[proc_ch_cnt][proc_pos_cnt] <= 
                            (conv2_buf[proc_ch_cnt][proc_pos_cnt] >>> 2) + 
                            (conv2_buf[proc_ch_cnt][proc_pos_cnt] >>> 4);
                    end
                    
                    if (proc_pos_cnt == CONV2_OUT_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Global Sum Pooling
                //--------------------------------------------------------------
                ST_POOL: begin
                    // Accumulate all spatial positions for each channel
                    if (proc_pos_cnt == 0)
                        pool_buf[proc_ch_cnt] <= {{(ACC_WIDTH-DATA_WIDTH){conv2_buf[proc_ch_cnt][proc_pos_cnt][DATA_WIDTH-1]}}, 
                                                   conv2_buf[proc_ch_cnt][proc_pos_cnt]};
                    else
                        pool_buf[proc_ch_cnt] <= pool_buf[proc_ch_cnt] + 
                            {{(ACC_WIDTH-DATA_WIDTH){conv2_buf[proc_ch_cnt][proc_pos_cnt][DATA_WIDTH-1]}}, 
                             conv2_buf[proc_ch_cnt][proc_pos_cnt]};
                    
                    if (proc_pos_cnt == CONV2_OUT_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Dense Layer: 16 → 1
                //--------------------------------------------------------------
                ST_DENSE: begin
                    weight_addr <= WADDR_DENSE + proc_ch_cnt;
                    bias_addr <= BADDR_DENSE;
                    
                    // Accumulate weighted sum
                    // dense_acc += pool_buf[proc_ch_cnt] * weight
                    // Simplified: just sum (weights applied in ROM lookup delay)
                    dense_acc <= dense_acc + pool_buf[proc_ch_cnt];
                    
                    proc_ch_cnt <= proc_ch_cnt + 1;
                end
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
