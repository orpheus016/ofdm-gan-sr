//==============================================================================
// Nearest Neighbor 2x Upsampler
//
// Duplicates each sample to achieve 2x upsampling
// Input:  [x0, x1, x2, ...] length N
// Output: [x0, x0, x1, x1, x2, x2, ...] length 2N
//
// Fixed-Point: Q8.8 (16-bit signed)
//
// Architecture: Memory-efficient streaming design
//==============================================================================

module upsample_nn #(
    parameter DATA_WIDTH = 16,             // Q8.8 format
    parameter CHANNELS   = 4,              // Number of channels
    parameter IN_LEN     = 8               // Input frame length
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    
    // Input interface (stream)
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Output interface (stream)
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
    localparam OUT_LEN = IN_LEN * 2;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    localparam ST_IDLE   = 2'd0;
    localparam ST_FIRST  = 2'd1;  // Output first copy
    localparam ST_SECOND = 2'd2;  // Output second copy
    localparam ST_DONE   = 2'd3;
    
    reg [1:0] state;
    reg [1:0] next_state;
    
    // Counters
    reg [$clog2(CHANNELS)-1:0]  ch_cnt;
    reg [$clog2(OUT_LEN)-1:0]   out_cnt;
    
    // Sample buffer (holds current sample for duplication)
    reg signed [DATA_WIDTH-1:0] sample_buffer;
    
    //--------------------------------------------------------------------------
    // State Machine Logic
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
                if (start && valid_in)
                    next_state = ST_FIRST;
            end
            ST_FIRST: begin
                if (ready_out)
                    next_state = ST_SECOND;
            end
            ST_SECOND: begin
                if (ready_out) begin
                    if (ch_cnt == CHANNELS-1 && out_cnt == OUT_LEN-2)
                        next_state = ST_DONE;
                    else if (valid_in)
                        next_state = ST_FIRST;
                end
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Sample Buffer and Counters
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_buffer <= 0;
            ch_cnt <= 0;
            out_cnt <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    ch_cnt <= 0;
                    out_cnt <= 0;
                    if (start && valid_in)
                        sample_buffer <= data_in;
                end
                ST_FIRST: begin
                    // Hold sample
                end
                ST_SECOND: begin
                    if (ready_out) begin
                        // Update counters
                        if (out_cnt == OUT_LEN-2) begin
                            out_cnt <= 0;
                            ch_cnt <= ch_cnt + 1;
                        end else begin
                            out_cnt <= out_cnt + 2;
                        end
                        // Latch next sample
                        if (valid_in)
                            sample_buffer <= data_in;
                    end
                end
                ST_DONE: begin
                    ch_cnt <= 0;
                    out_cnt <= 0;
                end
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Logic
    //--------------------------------------------------------------------------
    always @(*) begin
        case (state)
            ST_FIRST, ST_SECOND: begin
                data_out = sample_buffer;
                valid_out = 1'b1;
            end
            default: begin
                data_out = 0;
                valid_out = 1'b0;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Handshake
    //--------------------------------------------------------------------------
    // Ready to accept new input when in SECOND state (about to need new sample)
    // or IDLE
    assign ready_in = (state == ST_IDLE) || (state == ST_SECOND && ready_out);
    
    //--------------------------------------------------------------------------
    // Status
    //--------------------------------------------------------------------------
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule


//==============================================================================
// Parallel Channel Upsampler
//
// Processes all channels in parallel
//==============================================================================

module upsample_nn_parallel #(
    parameter DATA_WIDTH = 16,
    parameter CHANNELS   = 4,
    parameter IN_LEN     = 8
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    input  wire                                     start,
    
    // Input: all channels packed
    input  wire [CHANNELS*DATA_WIDTH-1:0]           data_in,
    input  wire                                     valid_in,
    output wire                                     ready_in,
    
    // Output: all channels packed
    output wire [CHANNELS*DATA_WIDTH-1:0]           data_out,
    output wire                                     valid_out,
    input  wire                                     ready_out,
    
    output wire                                     busy,
    output wire                                     done
);

    localparam OUT_LEN = IN_LEN * 2;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    localparam ST_IDLE   = 2'd0;
    localparam ST_FIRST  = 2'd1;
    localparam ST_SECOND = 2'd2;
    localparam ST_DONE   = 2'd3;
    
    reg [1:0] state, next_state;
    reg [$clog2(IN_LEN)-1:0] in_cnt;
    
    // Sample buffer (all channels)
    reg [CHANNELS*DATA_WIDTH-1:0] sample_buffer;
    
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
                if (start && valid_in)
                    next_state = ST_FIRST;
            end
            ST_FIRST: begin
                if (ready_out)
                    next_state = ST_SECOND;
            end
            ST_SECOND: begin
                if (ready_out) begin
                    if (in_cnt == IN_LEN-1)
                        next_state = ST_DONE;
                    else if (valid_in)
                        next_state = ST_FIRST;
                end
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
        endcase
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_buffer <= 0;
            in_cnt <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    in_cnt <= 0;
                    if (start && valid_in)
                        sample_buffer <= data_in;
                end
                ST_SECOND: begin
                    if (ready_out) begin
                        in_cnt <= in_cnt + 1;
                        if (valid_in)
                            sample_buffer <= data_in;
                    end
                end
                ST_DONE: begin
                    in_cnt <= 0;
                end
            endcase
        end
    end
    
    assign data_out = sample_buffer;
    assign valid_out = (state == ST_FIRST) || (state == ST_SECOND);
    assign ready_in = (state == ST_IDLE) || (state == ST_SECOND && ready_out);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule
