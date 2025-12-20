//==============================================================================
// Global Sum Pooling Module
//
// Computes sum of all elements across spatial dimension for each channel
// Output is one value per channel
//
// Fixed-Point: Q8.8 input, Q16.16 accumulator, Q8.8 output
//
// Architecture: Streaming accumulator
//==============================================================================

`timescale 1ns / 1ps

module sum_pool #(
    parameter DATA_WIDTH = 16,             // Q8.8 format
    parameter ACC_WIDTH  = 32,             // Q16.16 for accumulation
    parameter CHANNELS   = 16,             // Number of channels
    parameter FRAME_LEN  = 4               // Spatial dimension
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    
    // Input interface (stream: ch0[0], ch0[1], ..., ch0[N-1], ch1[0], ...)
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Output interface (stream: sum_ch0, sum_ch1, ...)
    output reg  signed [DATA_WIDTH-1:0] data_out,
    output reg                          valid_out,
    input  wire                         ready_out,
    
    // Status
    output wire                         busy,
    output wire                         done
);

    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    localparam ST_IDLE   = 2'd0;
    localparam ST_ACC    = 2'd1;  // Accumulating
    localparam ST_OUTPUT = 2'd2;  // Outputting results
    localparam ST_DONE   = 2'd3;
    
    reg [1:0] state, next_state;
    
    // Counters
    reg [$clog2(CHANNELS)-1:0]  ch_cnt;
    reg [$clog2(FRAME_LEN)-1:0] pos_cnt;
    reg [$clog2(CHANNELS)-1:0]  out_cnt;
    
    // Accumulators (one per channel)
    reg signed [ACC_WIDTH-1:0] accumulators [0:CHANNELS-1];
    
    integer i;
    
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
                    next_state = ST_ACC;
            end
            ST_ACC: begin
                if (valid_in && ch_cnt == CHANNELS-1 && pos_cnt == FRAME_LEN-1)
                    next_state = ST_OUTPUT;
            end
            ST_OUTPUT: begin
                if (ready_out && out_cnt == CHANNELS-1)
                    next_state = ST_DONE;
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Accumulation Logic
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch_cnt <= 0;
            pos_cnt <= 0;
            for (i = 0; i < CHANNELS; i = i + 1)
                accumulators[i] <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    ch_cnt <= 0;
                    pos_cnt <= 0;
                    if (start) begin
                        for (i = 0; i < CHANNELS; i = i + 1)
                            accumulators[i] <= 0;
                    end
                end
                ST_ACC: begin
                    if (valid_in) begin
                        // Accumulate (sign extend to 32-bit)
                        accumulators[ch_cnt] <= accumulators[ch_cnt] + 
                            {{(ACC_WIDTH-DATA_WIDTH){data_in[DATA_WIDTH-1]}}, data_in};
                        
                        // Update counters (spatial first, then channel)
                        if (pos_cnt == FRAME_LEN-1) begin
                            pos_cnt <= 0;
                            ch_cnt <= ch_cnt + 1;
                        end else begin
                            pos_cnt <= pos_cnt + 1;
                        end
                    end
                end
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Logic
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_cnt <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    out_cnt <= 0;
                end
                ST_OUTPUT: begin
                    if (ready_out)
                        out_cnt <= out_cnt + 1;
                end
            endcase
        end
    end
    
    // Saturate accumulator to 16-bit output
    wire signed [ACC_WIDTH-1:0] current_acc;
    wire signed [DATA_WIDTH-1:0] saturated_out;
    
    assign current_acc = accumulators[out_cnt];
    
    // Saturation logic
    assign saturated_out = 
        (current_acc > $signed(32'h00007FFF)) ? 16'h7FFF :
        (current_acc < $signed(32'hFFFF8000)) ? 16'h8000 :
        current_acc[DATA_WIDTH-1:0];
    
    always @(*) begin
        if (state == ST_OUTPUT) begin
            data_out = saturated_out;
            valid_out = 1'b1;
        end else begin
            data_out = 0;
            valid_out = 1'b0;
        end
    end
    
    //--------------------------------------------------------------------------
    // Handshake & Status
    //--------------------------------------------------------------------------
    assign ready_in = (state == ST_ACC);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule


//==============================================================================
// Global Average Pooling (Alternative)
//
// Divides sum by FRAME_LEN using shift (if power of 2)
//==============================================================================

module avg_pool #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32,
    parameter CHANNELS   = 16,
    parameter FRAME_LEN  = 4               // Must be power of 2 for shift division
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    output reg  signed [DATA_WIDTH-1:0] data_out,
    output reg                          valid_out,
    input  wire                         ready_out,
    
    output wire                         busy,
    output wire                         done
);

    // Calculate shift amount (log2 of FRAME_LEN)
    localparam SHIFT = $clog2(FRAME_LEN);
    
    // Instantiate sum pool
    wire signed [DATA_WIDTH-1:0] sum_out;
    wire sum_valid;
    wire sum_busy, sum_done;
    
    sum_pool #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .CHANNELS(CHANNELS),
        .FRAME_LEN(FRAME_LEN)
    ) u_sum (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .data_out(sum_out),
        .valid_out(sum_valid),
        .ready_out(ready_out),
        .busy(sum_busy),
        .done(sum_done)
    );
    
    // Divide by FRAME_LEN (arithmetic shift right)
    always @(*) begin
        data_out = sum_out >>> SHIFT;
        valid_out = sum_valid;
    end
    
    assign busy = sum_busy;
    assign done = sum_done;

endmodule
