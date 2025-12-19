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
    
    // State machine
    localparam ST_IDLE     = 3'd0;
    localparam ST_LOAD_IN  = 3'd1;
    localparam ST_COMPUTE  = 3'd2;
    localparam ST_ADD_BIAS = 3'd3;
    localparam ST_OUTPUT   = 3'd4;
    localparam ST_DONE     = 3'd5;
    
    reg [2:0] state;
    reg [$clog2(IN_SIZE):0] compute_idx;
    reg [$clog2(OUT_SIZE):0] output_idx;
    
    // Pipeline registers
    reg signed [DATA_WIDTH-1:0] input_sample;
    reg signed [WEIGHT_WIDTH-1:0] weight_sample;
    reg pipeline_valid;
    
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
            data_out <= 0;
            data_out_valid <= 1'b0;
            weight_addr <= 0;
            bias_addr <= 0;
            out_idx <= 0;
            pipeline_valid <= 1'b0;
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
                            state <= ST_COMPUTE;
                            output_idx <= 0;
                            compute_idx <= 0;
                            accumulator <= 0;
                            // Pre-fetch first weight
                            weight_addr <= 0;
                        end
                    end
                end
                
                ST_COMPUTE: begin
                    // MAC operation: accumulator += input[i] * weight[out][i]
                    if (compute_idx < IN_SIZE) begin
                        // Fetch input and weight
                        input_sample <= input_buf[compute_idx];
                        weight_sample <= weight_data;
                        pipeline_valid <= 1'b1;
                        
                        // Update weight address for next cycle
                        weight_addr <= output_idx * IN_SIZE + compute_idx + 1;
                        compute_idx <= compute_idx + 1;
                    end else begin
                        pipeline_valid <= 1'b0;
                    end
                    
                    // Accumulate (1 cycle delay for pipeline)
                    if (pipeline_valid) begin
                        accumulator <= accumulator + 
                            (input_sample * weight_sample);
                    end
                    
                    // Check if done with this output
                    if (compute_idx == IN_SIZE && !pipeline_valid) begin
                        state <= ST_ADD_BIAS;
                        bias_addr <= output_idx;
                    end
                end
                
                ST_ADD_BIAS: begin
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
                    out_idx <= output_idx;
                    
                    output_idx <= output_idx + 1;
                    
                    if (output_idx == OUT_SIZE - 1) begin
                        state <= ST_DONE;
                    end else begin
                        // Next output neuron
                        state <= ST_COMPUTE;
                        compute_idx <= 0;
                        accumulator <= 0;
                        weight_addr <= (output_idx + 1) * IN_SIZE;
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
