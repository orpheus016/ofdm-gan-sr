//==============================================================================
// Dense (Fully Connected) Layer Module
//
// Computes: y = W * x + b
// Where W is [OUT_SIZE x IN_SIZE], x is [IN_SIZE x 1], b is [OUT_SIZE x 1]
//
// Fixed-Point Formats:
//   - Input/Output: Q8.8 (16-bit signed)
//   - Weights: Q1.7 (8-bit signed)
//   - Bias: Q8.8 (16-bit signed)
//
// Operation: For each output j:
//   y[j] = sum(W[j][i] * x[i]) + b[j]  for i = 0 to IN_SIZE-1
//
// Simple sequential implementation for small networks
//==============================================================================

`timescale 1ns / 1ps

module dense_layer #(
    parameter IN_SIZE     = 9,            // Input vector size
    parameter OUT_SIZE    = 3,            // Output vector size
    parameter DATA_WIDTH  = 16,           // Q8.8 for activations
    parameter WEIGHT_WIDTH = 8,           // Q1.7 for weights
    parameter WEIGHT_FRAC  = 7,           // Fractional bits in weights
    parameter DATA_FRAC    = 8            // Fractional bits in data
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Control
    input  wire                         start,
    output reg                          done,
    output reg                          busy,
    
    // Input data (sequential load)
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                         data_valid,
    
    // Output data (sequential output)
    output reg signed [DATA_WIDTH-1:0]  data_out,
    output reg                          data_out_valid,
    output reg [$clog2(OUT_SIZE)-1:0]   out_idx,
    
    // Weight/bias interface (directly connected to ROM)
    // Weight address: [out_idx * IN_SIZE + in_idx]
    output reg [$clog2(IN_SIZE*OUT_SIZE)-1:0] weight_addr,
    input  wire signed [WEIGHT_WIDTH-1:0]     weight_data,
    
    // Bias address: [out_idx]
    output reg [$clog2(OUT_SIZE)-1:0]         bias_addr,
    input  wire signed [DATA_WIDTH-1:0]       bias_data
);

    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    
    // Input buffer
    reg signed [DATA_WIDTH-1:0] input_buf [0:IN_SIZE-1];
    reg [$clog2(IN_SIZE):0] in_cnt;
    
    // Accumulator (wider for MAC precision)
    localparam ACC_WIDTH = DATA_WIDTH + WEIGHT_WIDTH + $clog2(IN_SIZE);
    reg signed [ACC_WIDTH-1:0] accumulator;

    // Pipeline Product Register (New)
    // Size = Data(16) + Weight(8) = 24 bits
    reg signed [DATA_WIDTH + WEIGHT_WIDTH - 1 : 0] product_reg;
    
    // State machine
    localparam [3:0]
        ST_IDLE         = 4'd0,
        ST_LOAD_IN      = 4'd1,
        ST_PREP_COMPUTE = 4'd2,     // Prepare for a new neuron
        ST_MAC_WAIT     = 4'd3,     // Wait for ROM
        ST_MAC_MULT     = 4'd4,     // Stage 1: Multiply
        ST_MAC_ACCUM    = 4'd5,     // Stage 2: Accumulate
        ST_MAC_UPDATE   = 4'd6,     // Stage 3: Update Index
        ST_ADD_BIAS     = 4'd7,
        ST_OUTPUT       = 4'd8,
        ST_DONE         = 4'd9;
    
    reg [3:0] state;
    reg [$clog2(IN_SIZE):0] compute_idx;
    reg [$clog2(OUT_SIZE):0] output_idx;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            busy <= 1'b0;
            in_cnt <= 0;
            compute_idx <= 0;
            output_idx <= 0;
            accumulator <= 0;
            product_reg <= 0;
            data_out <= 0;
            data_out_valid <= 1'b0;
            weight_addr <= 0;
            bias_addr <= 0;
            out_idx <= 0;
        end else begin
            // Default outputs
            data_out_valid <= 1'b0;
            done <= 1'b0;
            
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        state <= ST_LOAD_IN;
                        busy <= 1'b1;
                        in_cnt <= 0;
                    end
                end
                
                ST_LOAD_IN: begin
                    if (data_valid) begin
                        input_buf[in_cnt] <= data_in;
                        in_cnt <= in_cnt + 1;
                        
                        if (in_cnt == IN_SIZE - 1) begin
                            state <= ST_PREP_COMPUTE;
                            output_idx <= 0;
                        end
                    end
                end

                ST_PREP_COMPUTE: begin
                    // Initialize for the current output neuron
                    compute_idx <= 0;
                    accumulator <= 0;
                    weight_addr <= output_idx * IN_SIZE; // Request first weight
                    state <= ST_MAC_WAIT;
                end
                
                ST_MAC_WAIT: begin
                    // Wait 1 cycle for ROM latency
                    // Data requested in PREP or UPDATE will be ready next cycle
                    state <= ST_MAC_MULT;
                end

                ST_MAC_MULT: begin
                    // Stage 1: Multiplication
                    // product = input[i] * weight[i]
                    product_reg <= $signed(input_buf[compute_idx]) * $signed(weight_data);
                    state <= ST_MAC_ACCUM;
                end

                ST_MAC_ACCUM: begin
                    // Stage 2: Accumulation
                    // acc += product
                    accumulator <= accumulator + product_reg;
                    state <= ST_MAC_UPDATE;
                end

                ST_MAC_UPDATE: begin
                    // Stage 3: Update Index and Address
                    if (compute_idx < IN_SIZE - 1) begin
                        compute_idx <= compute_idx + 1;
                        weight_addr <= weight_addr + 1; // Prepare next weight address
                        state <= ST_MAC_WAIT;           // Loop back
                    end else begin
                        // Done with all inputs for this neuron
                        bias_addr <= output_idx;        // Request bias
                        state <= ST_ADD_BIAS;
                    end
                end
                
                ST_ADD_BIAS: begin
                    // Wait one cycle implies bias is ready now (since addr set in UPDATE)
                    // Add bias and shift for fixed-point alignment
                    // Result: Q8.8 + (Q8.8 * Q1.7) >> 7 = Q8.8
                    accumulator <= (accumulator >>> WEIGHT_FRAC) + 
                                   {{(ACC_WIDTH-DATA_WIDTH){bias_data[DATA_WIDTH-1]}}, bias_data};
                    state <= ST_OUTPUT;
                end
                
                ST_OUTPUT: begin
                    // Saturate to Q8.8 range
                    if (accumulator > 32767) begin
                        data_out <= 16'h7FFF;
                    end else if (accumulator < -32768) begin
                        data_out <= 16'h8000;
                    end else begin
                        data_out <= accumulator[DATA_WIDTH-1:0];
                    end
                    
                    data_out_valid <= 1'b1;
                    out_idx <= output_idx; // Output index signal
                    
                    output_idx <= output_idx + 1;
                    
                    if (output_idx == OUT_SIZE - 1) begin
                        state <= ST_DONE;
                    end else begin
                        // Next output neuron
                        state <= ST_PREP_COMPUTE;
                    end
                end
                
                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= ST_IDLE;
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule