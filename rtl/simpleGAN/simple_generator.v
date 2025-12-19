//==============================================================================
// Simple GAN - Generator Module
//
// Architecture: 2 (latent) -> 3 (hidden) -> 9 (output 3x3)
// Activations: tanh for all layers
//
// Based on MATLAB reference:
//   Generator: latent_dim=2 -> hidden=3 (tanh) -> output=9 (tanh)
//
// Fixed-Point Format:
//   - Data: Q8.8 (16-bit signed)
//   - Weights: Q1.7 (8-bit signed)
//
// Note: Weight ROM has 1-cycle latency, so we add wait states
//==============================================================================

`timescale 1ns / 1ps

module simple_generator #(
    parameter LATENT_DIM   = 2,
    parameter HIDDEN_SIZE  = 3,
    parameter OUTPUT_SIZE  = 9,
    parameter DATA_WIDTH   = 16,   // Q8.8
    parameter WEIGHT_WIDTH = 8     // Q1.7
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire signed [DATA_WIDTH-1:0] latent_in [0:LATENT_DIM-1],
    input  wire                         valid_in,
    
    // Output interface
    output reg signed [DATA_WIDTH-1:0]  gen_out [0:OUTPUT_SIZE-1],
    output reg                          valid_out,
    output reg                          done,
    
    // Weight ROM interface - Layer 1 (2 -> 3)
    output reg  [3:0]                   w1_addr,
    input  wire signed [WEIGHT_WIDTH-1:0] w1_data,
    output reg  [1:0]                   b1_addr,
    input  wire signed [DATA_WIDTH-1:0] b1_data,
    
    // Weight ROM interface - Layer 2 (3 -> 9)
    output reg  [4:0]                   w2_addr,
    input  wire signed [WEIGHT_WIDTH-1:0] w2_data,
    output reg  [3:0]                   b2_addr,
    input  wire signed [DATA_WIDTH-1:0] b2_data
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
        
        // --- LAYER 1 ---
        ST_L1_WAIT      = 5'd2,     // Wait for ROM
        ST_L1_MULT      = 5'd3,     // Stage 1: Multiply
        ST_L1_ACCUM     = 5'd4,     // Stage 2: Accumulate
        ST_L1_UPDATE    = 5'd5,     // Stage 3: Update Index/Addr
        ST_L1_BIAS_WAIT = 5'd6,
        ST_L1_BIAS      = 5'd7,
        ST_L1_ACT       = 5'd8,
        ST_L1_ACT_WAIT  = 5'd9,
        
        // --- LAYER 2 ---
        ST_L2_WAIT      = 5'd10,    // Wait for ROM
        ST_L2_MULT      = 5'd11,    // Stage 1: Multiply
        ST_L2_ACCUM     = 5'd12,    // Stage 2: Accumulate
        ST_L2_UPDATE    = 5'd13,    // Stage 3: Update Index/Addr
        ST_L2_BIAS_WAIT = 5'd14,
        ST_L2_BIAS      = 5'd15,
        ST_L2_ACT       = 5'd16,
        ST_L2_ACT_WAIT  = 5'd17,
        
        ST_DONE         = 5'd18;

    reg [4:0] state, next_state;
    
    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_reg [0:LATENT_DIM-1];
    reg signed [DATA_WIDTH-1:0] hidden [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] output_reg [0:OUTPUT_SIZE-1];
    
    reg signed [31:0] accumulator;
    // New Register for Pipelined MAC (Product)
    // Size = Data(16) + Weight(8) = 24 bits
    reg signed [23:0] product_reg; 

    reg [3:0] out_idx;    // Current output neuron index
    reg [3:0] in_idx;     // Current input element index
    
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
            out_idx <= 0;
            in_idx <= 0;
            accumulator <= 0;
            product_reg <= 0; // Reset product reg
            tanh_in <= 0;
            tanh_valid_in <= 0;
            w1_addr <= 0;
            w2_addr <= 0;
            b1_addr <= 0;
            b2_addr <= 0;
            
            for (integer i = 0; i < LATENT_DIM; i = i + 1)
                input_reg[i] <= 0;
            for (integer i = 0; i < HIDDEN_SIZE; i = i + 1)
                hidden[i] <= 0;
            for (integer i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                output_reg[i] <= 0;
                gen_out[i] <= 0;
            end
            
        end else begin
            tanh_valid_in <= 0;
            valid_out <= 0;
            
            case (state)
                ST_IDLE: begin
                    done <= 0;
                    if (valid_in) begin
                        for (integer i = 0; i < LATENT_DIM; i = i + 1)
                            input_reg[i] <= latent_in[i];
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
                    // Stage 3: Update Control / Indices
                    if (in_idx < LATENT_DIM - 1) begin
                        in_idx <= in_idx + 1;
                        w1_addr <= w1_addr + 1; // Prepare fetch for next loop
                    end else begin
                        b1_addr <= out_idx[1:0]; // Done inputs, fetch bias
                    end
                end
                // --------------------------
                
                ST_L1_BIAS_WAIT: begin
                    // Wait for Bias ROM
                end
                
                ST_L1_BIAS: begin
                    hidden[out_idx] <= (accumulator >>> FRAC_BITS) + b1_data;
                    
                    if (out_idx < HIDDEN_SIZE - 1) begin
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                        accumulator <= 0;
                        w1_addr <= (out_idx + 1) * LATENT_DIM;
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
                    // Stage 3: Update Control / Indices
                    if (in_idx < HIDDEN_SIZE - 1) begin
                        in_idx <= in_idx + 1;
                        w2_addr <= w2_addr + 1;
                    end else begin
                        b2_addr <= out_idx;
                    end
                end
                // --------------------------
                
                ST_L2_BIAS_WAIT: begin
                    // Wait for Bias ROM
                end
                
                ST_L2_BIAS: begin
                    output_reg[out_idx] <= (accumulator >>> FRAC_BITS) + b2_data;
                    
                    if (out_idx < OUTPUT_SIZE - 1) begin
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                        accumulator <= 0;
                        w2_addr <= (out_idx + 1) * HIDDEN_SIZE;
                    end else begin
                        out_idx <= 0;
                    end
                end
                
                ST_L2_ACT: begin
                    tanh_in <= output_reg[out_idx];
                    tanh_valid_in <= 1;
                end
                
                ST_L2_ACT_WAIT: begin
                    if (tanh_valid_out) begin
                        output_reg[out_idx] <= tanh_out;
                        gen_out[out_idx] <= tanh_out;
                        
                        if (out_idx < OUTPUT_SIZE - 1) begin
                            out_idx <= out_idx + 1;
                        end
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
                if (in_idx < LATENT_DIM - 1) // Note: Logic is delayed 1 cycle vs original because update is at end
                     next_state = ST_L1_WAIT; // Loop back for next input
                else
                     next_state = ST_L1_BIAS_WAIT; // Done
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
                     next_state = ST_L2_WAIT;
                else
                     next_state = ST_L2_BIAS_WAIT;
            end
            
            ST_L2_BIAS_WAIT: next_state = ST_L2_BIAS;
            
            ST_L2_BIAS: begin
                if (out_idx == OUTPUT_SIZE - 1)
                    next_state = ST_L2_ACT;
                else
                    next_state = ST_L2_WAIT;
            end
            
            ST_L2_ACT:      next_state = ST_L2_ACT_WAIT;
            ST_L2_ACT_WAIT: begin
                if (tanh_valid_out) begin
                    if (out_idx == OUTPUT_SIZE - 1)
                        next_state = ST_DONE;
                    else
                        next_state = ST_L2_ACT;
                end
            end
            
            ST_DONE: next_state = ST_IDLE;
            
            default: next_state = ST_IDLE;
        endcase
    end

endmodule