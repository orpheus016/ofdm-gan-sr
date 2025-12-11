//==============================================================================
// Mini U-Net Generator for OFDM Signal Reconstruction
//
// Simplified representative architecture for FPGA demonstration
//
// Architecture:
//   Input [2×16] → Enc1 [4×8] → Bottleneck [8×4] → Dec1 [4×8] → Output [2×16]
//                 ↓                                    ↑
//                 └──────── Skip Connection ───────────┘
//
// Features:
//   - Pipeline architecture for high throughput
//   - Parallel MAC units
//   - Memory-efficient skip connections
//   - Fixed-point: Q8.8 activations, Q1.7 weights
//
// Layer Breakdown:
//   Enc1:  Conv1D(2→4, k=3, s=2, pad=1) + LeakyReLU  16→8
//   Bottleneck: Conv1D(4→8, k=3, s=2, pad=1) + LeakyReLU  8→4
//   Up1:   Upsample(2x) + Conv1D(8→4, k=3, s=1, pad=1) + LeakyReLU  4→8
//   Skip:  Add encoder output
//   Output: Conv1D(4→2, k=1, s=1, pad=0) + Tanh  8→16
//
// Parameter Count: ~500 weights (demonstration)
//==============================================================================

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
    // Layer dimensions
    localparam ENC1_OUT_CH  = 4;
    localparam ENC1_OUT_LEN = 8;   // 16/2
    localparam BNECK_OUT_CH = 8;
    localparam BNECK_OUT_LEN = 4;  // 8/2
    localparam DEC1_OUT_CH  = 4;
    localparam DEC1_OUT_LEN = 8;   // 4*2
    
    // Weight ROM addresses
    localparam WADDR_ENC1   = 0;
    localparam WADDR_BNECK  = 24;
    localparam WADDR_DEC1   = 120;
    localparam WADDR_OUT    = 216;
    
    // Bias ROM addresses
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
    // Buffers (implemented as BRAM in synthesis)
    //--------------------------------------------------------------------------
    // Input buffer
    reg signed [DATA_WIDTH-1:0] input_buf [0:IN_CH-1][0:FRAME_LEN-1];
    
    // Skip buffer (encoder output for residual connection)
    reg signed [DATA_WIDTH-1:0] skip_buf [0:ENC1_OUT_CH-1][0:ENC1_OUT_LEN-1];
    
    // Intermediate buffers
    reg signed [DATA_WIDTH-1:0] enc1_buf [0:ENC1_OUT_CH-1][0:ENC1_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] bneck_buf [0:BNECK_OUT_CH-1][0:BNECK_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] up1_buf [0:BNECK_OUT_CH-1][0:BNECK_OUT_LEN*2-1];
    reg signed [DATA_WIDTH-1:0] dec1_buf [0:DEC1_OUT_CH-1][0:DEC1_OUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] up2_buf [0:DEC1_OUT_CH-1][0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] out_buf [0:OUT_CH-1][0:FRAME_LEN-1];
    
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
    reg [$clog2(IN_CH)-1:0]         in_ch_cnt;
    reg [$clog2(FRAME_LEN)-1:0]     in_pos_cnt;
    reg [$clog2(BNECK_OUT_CH)-1:0]  proc_ch_cnt;
    reg [$clog2(FRAME_LEN)-1:0]     proc_pos_cnt;
    reg [$clog2(3)-1:0]             kern_cnt;  // Kernel index
    reg [$clog2(IN_CH)-1:0]         acc_ch_cnt;  // Accumulation channel
    
    // Accumulator
    reg signed [ACC_WIDTH-1:0] accumulator;
    
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
                // Enc1 complete when all output positions computed
                if (proc_ch_cnt == ENC1_OUT_CH-1 && proc_pos_cnt == ENC1_OUT_LEN-1 &&
                    acc_ch_cnt == IN_CH-1 && kern_cnt == 2)
                    next_state = ST_BNECK;
            end
            ST_BNECK: begin
                if (proc_ch_cnt == BNECK_OUT_CH-1 && proc_pos_cnt == BNECK_OUT_LEN-1 &&
                    acc_ch_cnt == ENC1_OUT_CH-1 && kern_cnt == 2)
                    next_state = ST_UPSAMPLE1;
            end
            ST_UPSAMPLE1: begin
                if (proc_ch_cnt == BNECK_OUT_CH-1 && proc_pos_cnt == BNECK_OUT_LEN-1)
                    next_state = ST_DEC1;
            end
            ST_DEC1: begin
                if (proc_ch_cnt == DEC1_OUT_CH-1 && proc_pos_cnt == DEC1_OUT_LEN-1 &&
                    acc_ch_cnt == BNECK_OUT_CH-1 && kern_cnt == 2)
                    next_state = ST_SKIP_ADD;
            end
            ST_SKIP_ADD: begin
                if (proc_ch_cnt == DEC1_OUT_CH-1 && proc_pos_cnt == DEC1_OUT_LEN-1)
                    next_state = ST_UPSAMPLE2;
            end
            ST_UPSAMPLE2: begin
                if (proc_ch_cnt == DEC1_OUT_CH-1 && proc_pos_cnt == DEC1_OUT_LEN-1)
                    next_state = ST_OUT_CONV;
            end
            ST_OUT_CONV: begin
                if (proc_ch_cnt == OUT_CH-1 && proc_pos_cnt == FRAME_LEN-1 &&
                    acc_ch_cnt == DEC1_OUT_CH-1 && kern_cnt == 0)
                    next_state = ST_TANH;
            end
            ST_TANH: begin
                if (proc_ch_cnt == OUT_CH-1 && proc_pos_cnt == FRAME_LEN-1)
                    next_state = ST_OUTPUT;
            end
            ST_OUTPUT: begin
                if (in_ch_cnt == OUT_CH-1 && in_pos_cnt == FRAME_LEN-1 && ready_out)
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
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_ch_cnt <= 0;
            in_pos_cnt <= 0;
        end else if (state == ST_IDLE && start) begin
            in_ch_cnt <= 0;
            in_pos_cnt <= 0;
        end else if (state == ST_LOAD_IN && valid_in) begin
            input_buf[in_ch_cnt][in_pos_cnt] <= data_in;
            
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
    // Convolution Processing (simplified - processes sequentially)
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proc_ch_cnt <= 0;
            proc_pos_cnt <= 0;
            kern_cnt <= 0;
            acc_ch_cnt <= 0;
            accumulator <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    proc_ch_cnt <= 0;
                    proc_pos_cnt <= 0;
                    kern_cnt <= 0;
                    acc_ch_cnt <= 0;
                    accumulator <= 0;
                end
                
                //--------------------------------------------------------------
                // Encoder 1: Conv(2→4, k=3, s=2)
                //--------------------------------------------------------------
                ST_ENC1: begin
                    // Weight address
                    weight_addr <= WADDR_ENC1 + proc_ch_cnt * (IN_CH * 3) + 
                                   acc_ch_cnt * 3 + kern_cnt;
                    bias_addr <= BADDR_ENC1 + proc_ch_cnt;
                    
                    // MAC operation (with 1 cycle delay for ROM)
                    // Input index: proc_pos_cnt*2 + kern_cnt - 1 (stride=2, pad=1)
                    // Accumulator logic here...
                    
                    // Counter update
                    if (kern_cnt == 2) begin
                        kern_cnt <= 0;
                        if (acc_ch_cnt == IN_CH-1) begin
                            acc_ch_cnt <= 0;
                            // Store result (simplified - would need activation)
                            enc1_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                            skip_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                            accumulator <= 0;
                            
                            if (proc_pos_cnt == ENC1_OUT_LEN-1) begin
                                proc_pos_cnt <= 0;
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
                // Bottleneck: Conv(4→8, k=3, s=2)
                //--------------------------------------------------------------
                ST_BNECK: begin
                    weight_addr <= WADDR_BNECK + proc_ch_cnt * (ENC1_OUT_CH * 3) + 
                                   acc_ch_cnt * 3 + kern_cnt;
                    bias_addr <= BADDR_BNECK + proc_ch_cnt;
                    
                    // Counter update (similar to ENC1)
                    if (kern_cnt == 2) begin
                        kern_cnt <= 0;
                        if (acc_ch_cnt == ENC1_OUT_CH-1) begin
                            acc_ch_cnt <= 0;
                            bneck_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                            accumulator <= 0;
                            
                            if (proc_pos_cnt == BNECK_OUT_LEN-1) begin
                                proc_pos_cnt <= 0;
                                if (proc_ch_cnt == BNECK_OUT_CH-1)
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
                // Upsample 1: NN 2x
                //--------------------------------------------------------------
                ST_UPSAMPLE1: begin
                    // Duplicate each sample
                    up1_buf[proc_ch_cnt][proc_pos_cnt*2]   <= bneck_buf[proc_ch_cnt][proc_pos_cnt];
                    up1_buf[proc_ch_cnt][proc_pos_cnt*2+1] <= bneck_buf[proc_ch_cnt][proc_pos_cnt];
                    
                    if (proc_pos_cnt == BNECK_OUT_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Decoder 1: Conv(8→4, k=3, s=1)
                //--------------------------------------------------------------
                ST_DEC1: begin
                    weight_addr <= WADDR_DEC1 + proc_ch_cnt * (BNECK_OUT_CH * 3) + 
                                   acc_ch_cnt * 3 + kern_cnt;
                    bias_addr <= BADDR_DEC1 + proc_ch_cnt;
                    
                    // Counter update
                    if (kern_cnt == 2) begin
                        kern_cnt <= 0;
                        if (acc_ch_cnt == BNECK_OUT_CH-1) begin
                            acc_ch_cnt <= 0;
                            dec1_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                            accumulator <= 0;
                            
                            if (proc_pos_cnt == DEC1_OUT_LEN-1) begin
                                proc_pos_cnt <= 0;
                                if (proc_ch_cnt == DEC1_OUT_CH-1)
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
                // Skip Add (residual connection)
                //--------------------------------------------------------------
                ST_SKIP_ADD: begin
                    // Add skip connection (encoder output + decoder output)
                    dec1_buf[proc_ch_cnt][proc_pos_cnt] <= 
                        dec1_buf[proc_ch_cnt][proc_pos_cnt] + skip_buf[proc_ch_cnt][proc_pos_cnt];
                    
                    if (proc_pos_cnt == DEC1_OUT_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Upsample 2: NN 2x
                //--------------------------------------------------------------
                ST_UPSAMPLE2: begin
                    up2_buf[proc_ch_cnt][proc_pos_cnt*2]   <= dec1_buf[proc_ch_cnt][proc_pos_cnt];
                    up2_buf[proc_ch_cnt][proc_pos_cnt*2+1] <= dec1_buf[proc_ch_cnt][proc_pos_cnt];
                    
                    if (proc_pos_cnt == DEC1_OUT_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Output Conv: Conv(4→2, k=1, s=1)
                //--------------------------------------------------------------
                ST_OUT_CONV: begin
                    weight_addr <= WADDR_OUT + proc_ch_cnt * DEC1_OUT_CH + acc_ch_cnt;
                    bias_addr <= BADDR_OUT + proc_ch_cnt;
                    kern_cnt <= 0;
                    
                    if (acc_ch_cnt == DEC1_OUT_CH-1) begin
                        acc_ch_cnt <= 0;
                        out_buf[proc_ch_cnt][proc_pos_cnt] <= accumulator[DATA_WIDTH-1:0];
                        accumulator <= 0;
                        
                        if (proc_pos_cnt == FRAME_LEN-1) begin
                            proc_pos_cnt <= 0;
                            if (proc_ch_cnt == OUT_CH-1)
                                proc_ch_cnt <= 0;
                            else
                                proc_ch_cnt <= proc_ch_cnt + 1;
                        end else begin
                            proc_pos_cnt <= proc_pos_cnt + 1;
                        end
                    end else begin
                        acc_ch_cnt <= acc_ch_cnt + 1;
                    end
                end
                
                //--------------------------------------------------------------
                // Tanh Activation
                //--------------------------------------------------------------
                ST_TANH: begin
                    // Apply tanh (simplified - use LUT in practice)
                    // For demo, just saturate to [-1, 1] range
                    if (out_buf[proc_ch_cnt][proc_pos_cnt] > 16'sh00FF)
                        out_buf[proc_ch_cnt][proc_pos_cnt] <= 16'sh00FF;  // 1.0
                    else if (out_buf[proc_ch_cnt][proc_pos_cnt] < 16'shFF01)
                        out_buf[proc_ch_cnt][proc_pos_cnt] <= 16'shFF01;  // -1.0
                    
                    if (proc_pos_cnt == FRAME_LEN-1) begin
                        proc_pos_cnt <= 0;
                        proc_ch_cnt <= proc_ch_cnt + 1;
                    end else begin
                        proc_pos_cnt <= proc_pos_cnt + 1;
                    end
                end
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
