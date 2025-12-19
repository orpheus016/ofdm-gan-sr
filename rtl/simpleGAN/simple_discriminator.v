//==============================================================================
// Simple GAN - Discriminator Module
//
// Architecture: 9 (input 3x3) -> 3 (hidden) -> 1 (output)
// Activations: tanh for hidden, sigmoid for output
//
// Based on MATLAB reference:
//   Discriminator: input=9 -> hidden=3 (tanh) -> output=1 (sigmoid)
//
// Fixed-Point Format:
//   - Data: Q8.8 (16-bit signed)
//   - Weights: Q1.7 (8-bit signed)
//
// Note: Weight ROM has 1-cycle latency, so we add wait states
//==============================================================================

`timescale 1ns / 1ps

module simple_discriminator #(
    parameter INPUT_SIZE   = 9,
    parameter HIDDEN_SIZE  = 3,
    parameter OUTPUT_SIZE  = 1,
    parameter DATA_WIDTH   = 16,   // Q8.8
    parameter WEIGHT_WIDTH = 8     // Q1.7
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface (3x3 image flattened to 9 elements)
    input  wire signed [DATA_WIDTH-1:0] data_in [0:INPUT_SIZE-1],
    input  wire                         valid_in,
    
    // Output interface (single discriminator score)
    output reg signed [DATA_WIDTH-1:0]  disc_out,
    output reg                          valid_out,
    output reg                          done,
    
    // Weight ROM interface - Layer 1 (9 -> 3)
    output reg  [4:0]                   w1_addr,  // Up to 27 weights
    input  wire signed [WEIGHT_WIDTH-1:0] w1_data,
    output reg  [1:0]                   b1_addr,
    input  wire signed [DATA_WIDTH-1:0] b1_data,
    
    // Weight ROM interface - Layer 2 (3 -> 1)
    output reg  [1:0]                   w2_addr,  // 3 weights
    input  wire signed [WEIGHT_WIDTH-1:0] w2_data,
    input  wire signed [DATA_WIDTH-1:0] b2_data   // Single bias
);

    //--------------------------------------------------------------------------
    // Local Parameters
    //--------------------------------------------------------------------------
    localparam FRAC_BITS = 7;  // Weight fraction bits (Q1.7)

    //--------------------------------------------------------------------------
    // State Machine - Pipelined MAC (3 Stages)
    //--------------------------------------------------------------------------
    localparam [4:0]
        ST_IDLE         = 5'd0,
        ST_LOAD_INPUT   = 5'd1,
        
        // --- LAYER 1 (9 -> 3) ---
        ST_L1_WAIT      = 5'd2,     // Wait for ROM
        ST_L1_MULT      = 5'd3,     // Stage 1: Multiply
        ST_L1_ACCUM     = 5'd4,     // Stage 2: Accumulate
        ST_L1_UPDATE    = 5'd5,     // Stage 3: Update Index
        ST_L1_BIAS_WAIT = 5'd6,
        ST_L1_BIAS      = 5'd7,
        ST_L1_ACT       = 5'd8,
        ST_L1_ACT_WAIT  = 5'd9,
        
        // --- LAYER 2 (3 -> 1) ---
        ST_L2_WAIT      = 5'd10,    // Wait for ROM
        ST_L2_MULT      = 5'd11,    // Stage 1: Multiply
        ST_L2_ACCUM     = 5'd12,    // Stage 2: Accumulate
        ST_L2_UPDATE    = 5'd13,    // Stage 3: Update Index
        ST_L2_BIAS_WAIT = 5'd14,
        ST_L2_BIAS      = 5'd15,
        ST_L2_ACT       = 5'd16,
        ST_L2_ACT_WAIT  = 5'd17,
        
        ST_DONE         = 5'd18;

    reg [4:0] state, next_state;
    
    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_reg [0:INPUT_SIZE-1];
    reg signed [DATA_WIDTH-1:0] hidden [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] output_reg;
    
    reg signed [31:0] accumulator;
    // New Register for Pipelined MAC
    // Size = Data(16) + Weight(8) = 24 bits
    reg signed [23:0] product_reg;
    
    reg [3:0] out_idx;   // Current output neuron index
    reg [3:0] in_idx;    // Current input element index
    
    //--------------------------------------------------------------------------
    // Tanh Activation Module
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] tanh_in;
    reg                         tanh_valid_in;
    wire signed [DATA_WIDTH-1:0] tanh_out;
    wire                        tanh_valid_out;
    
    activation_tanh #(
        .DATA_WIDTH(DATA_WIDTH),
        .PIPELINED(1)
    ) u_tanh (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(tanh_in),
        .valid_in(tanh_valid_in),
        .data_out(tanh_out),
        .valid_out(tanh_valid_out)
    );
    
    //--------------------------------------------------------------------------
    // Sigmoid Activation Module
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] sigmoid_in;
    reg                         sigmoid_valid_in;
    wire signed [DATA_WIDTH-1:0] sigmoid_out;
    wire                        sigmoid_valid_out;
    
    activation_sigmoid #(
        .DATA_WIDTH(DATA_WIDTH),
        .PIPELINED(1)
    ) u_sigmoid (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(sigmoid_in),
        .valid_in(sigmoid_valid_in),
        .data_out(sigmoid_out),
        .valid_out(sigmoid_valid_out)
    );
    
    //--------------------------------------------------------------------------
    // State Register
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    //--------------------------------------------------------------------------
    // Main Processing Logic
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            done <= 0;
            disc_out <= 0;
            out_idx <= 0;
            in_idx <= 0;
            accumulator <= 0;
            product_reg <= 0; // Reset product reg
            tanh_in <= 0;
            tanh_valid_in <= 0;
            sigmoid_in <= 0;
            sigmoid_valid_in <= 0;
            w1_addr <= 0;
            w2_addr <= 0;
            b1_addr <= 0;
            output_reg <= 0;
            
            for (integer i = 0; i < INPUT_SIZE; i = i + 1)
                input_reg[i] <= 0;
            for (integer i = 0; i < HIDDEN_SIZE; i = i + 1)
                hidden[i] <= 0;
            
        end else begin
            tanh_valid_in <= 0;
            sigmoid_valid_in <= 0;
            valid_out <= 0;
            
            case (state)
                ST_IDLE: begin
                    done <= 0;
                    if (valid_in) begin
                        for (integer i = 0; i < INPUT_SIZE; i = i + 1)
                            input_reg[i] <= data_in[i];
                    end
                end
                
                ST_LOAD_INPUT: begin
                    out_idx <= 0;
                    in_idx <= 0;
                    accumulator <= 0;
                    w1_addr <= 0;
                end
                
                ST_L1_WAIT: begin
                    // Wait for ROM data
                end
                
                // --- LAYER 1 MAC STAGES ---
                ST_L1_MULT: begin
                    // Stage 1: Multiplication
                    product_reg <= $signed(input_reg[in_idx]) * $signed(w1_data);
                end
                
                ST_L1_ACCUM: begin
                    // Stage 2: Accumulation
                    accumulator <= accumulator + product_reg;
                end
                
                ST_L1_UPDATE: begin
                    // Stage 3: Update Control
                    if (in_idx < INPUT_SIZE - 1) begin
                        in_idx <= in_idx + 1;
                        w1_addr <= w1_addr + 1;
                    end else begin
                        b1_addr <= out_idx[1:0];
                    end
                end
                // --------------------------
                
                ST_L1_BIAS_WAIT: begin
                    // Wait for bias ROM
                end
                
                ST_L1_BIAS: begin
                    hidden[out_idx] <= (accumulator >>> FRAC_BITS) + b1_data;
                    
                    if (out_idx < HIDDEN_SIZE - 1) begin
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                        accumulator <= 0;
                        w1_addr <= (out_idx + 1) * INPUT_SIZE;
                    end else begin
                        out_idx <= 0;
                    end
                end
                
                ST_L1_ACT: begin
                    tanh_in <= hidden[out_idx];
                    tanh_valid_in <= 1;
                end
                
                ST_L1_ACT_WAIT: begin
                    if (tanh_valid_out) begin
                        hidden[out_idx] <= tanh_out;
                        
                        if (out_idx < HIDDEN_SIZE - 1) begin
                            out_idx <= out_idx + 1;
                        end else begin
                            out_idx <= 0;
                            in_idx <= 0;
                            accumulator <= 0;
                            w2_addr <= 0;
                        end
                    end
                end
                
                ST_L2_WAIT: begin
                    // Wait for ROM
                end
                
                // --- LAYER 2 MAC STAGES ---
                ST_L2_MULT: begin
                    // Stage 1: Multiplication
                    product_reg <= $signed(hidden[in_idx]) * $signed(w2_data);
                end
                
                ST_L2_ACCUM: begin
                    // Stage 2: Accumulation
                    accumulator <= accumulator + product_reg;
                end
                
                ST_L2_UPDATE: begin
                    // Stage 3: Update Control
                    if (in_idx < HIDDEN_SIZE - 1) begin
                        in_idx <= in_idx + 1;
                        w2_addr <= w2_addr + 1;
                    end
                    // Else: done with inputs, move to bias
                end
                // --------------------------
                
                ST_L2_BIAS_WAIT: begin
                    // Wait if needed (kept for consistency)
                end
                
                ST_L2_BIAS: begin
                    output_reg <= (accumulator >>> FRAC_BITS) + b2_data;
                end
                
                ST_L2_ACT: begin
                    sigmoid_in <= output_reg;
                    sigmoid_valid_in <= 1;
                end
                
                ST_L2_ACT_WAIT: begin
                    if (sigmoid_valid_out) begin
                        output_reg <= sigmoid_out;
                        disc_out <= sigmoid_out;
                    end
                end
                
                ST_DONE: begin
                    valid_out <= 1;
                    done <= 1;
                end
                
                default: ;
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Next State Logic
    //--------------------------------------------------------------------------
    always @(*) begin
        next_state = state;
        
        case (state)
            ST_IDLE: begin
                if (valid_in) next_state = ST_LOAD_INPUT;
            end
            
            ST_LOAD_INPUT: next_state = ST_L1_WAIT;
            
            // --- LAYER 1 FLOW ---
            ST_L1_WAIT:   next_state = ST_L1_MULT;
            ST_L1_MULT:   next_state = ST_L1_ACCUM;
            ST_L1_ACCUM:  next_state = ST_L1_UPDATE;
            ST_L1_UPDATE: begin
                if (in_idx < INPUT_SIZE - 1)
                    next_state = ST_L1_WAIT; // Loop back
                else
                    next_state = ST_L1_BIAS_WAIT; // Done inputs
            end
            
            ST_L1_BIAS_WAIT: next_state = ST_L1_BIAS;
            
            ST_L1_BIAS: begin
                if (out_idx == HIDDEN_SIZE - 1)
                    next_state = ST_L1_ACT;
                else
                    next_state = ST_L1_WAIT;
            end
            
            ST_L1_ACT:      next_state = ST_L1_ACT_WAIT;
            ST_L1_ACT_WAIT: begin
                if (tanh_valid_out) begin
                    if (out_idx == HIDDEN_SIZE - 1)
                        next_state = ST_L2_WAIT;
                    else
                        next_state = ST_L1_ACT;
                end
            end
            
            // --- LAYER 2 FLOW ---
            ST_L2_WAIT:   next_state = ST_L2_MULT;
            ST_L2_MULT:   next_state = ST_L2_ACCUM;
            ST_L2_ACCUM:  next_state = ST_L2_UPDATE;
            ST_L2_UPDATE: begin
                if (in_idx < HIDDEN_SIZE - 1)
                    next_state = ST_L2_WAIT; // Loop back
                else
                    next_state = ST_L2_BIAS_WAIT; // Done inputs
            end
            
            ST_L2_BIAS_WAIT: next_state = ST_L2_BIAS;
            
            ST_L2_BIAS:      next_state = ST_L2_ACT;
            
            ST_L2_ACT:       next_state = ST_L2_ACT_WAIT;
            
            ST_L2_ACT_WAIT: begin
                if (sigmoid_valid_out)
                    next_state = ST_DONE;
            end
            
            ST_DONE: next_state = ST_IDLE;
            
            default: next_state = ST_IDLE;
        endcase
    end

endmodule