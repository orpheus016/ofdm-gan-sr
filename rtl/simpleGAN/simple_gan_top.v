//==============================================================================
// Simple GAN - Top Level Module
//
// Complete GAN with Generator and Discriminator
// Matches MATLAB reference: 3x3 image generation
//
// Architecture:
//   Generator:     latent(2) -> hidden(3) -> output(9)
//   Discriminator: input(9) -> hidden(3) -> output(1)
//
// Interface:
//   - Provide latent vector (2 values) to generate 3x3 image
//   - Provide image (9 values) to discriminate real/fake
//   - Can run generator only, discriminator only, or full GAN
//
// Fixed-Point Format:
//   - Data: Q8.8 (16-bit signed)
//   - Weights: Q1.7 (8-bit signed)
//==============================================================================

`timescale 1ns / 1ps

module simple_gan_top #(
    parameter LATENT_DIM   = 2,
    parameter HIDDEN_SIZE  = 3,
    parameter IMAGE_SIZE   = 9,    // 3x3 = 9
    parameter DATA_WIDTH   = 16,   // Q8.8
    parameter WEIGHT_WIDTH = 8     // Q1.7
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //----------------------------------------------------------------------
    // Control Interface
    //----------------------------------------------------------------------
    input  wire [1:0]                   mode,
    // 00: Generate only (latent -> image)
    // 01: Discriminate only (image_in -> score)
    // 10: Full GAN (latent -> image -> score)
    
    input  wire                         start,
    output wire                         busy,
    output wire                         done,
    
    //----------------------------------------------------------------------
    // Generator Input (Latent Vector)
    //----------------------------------------------------------------------
    input  wire signed [DATA_WIDTH-1:0] latent_in [0:LATENT_DIM-1],
    
    //----------------------------------------------------------------------
    // Discriminator Input (Direct image input)
    //----------------------------------------------------------------------
    input  wire signed [DATA_WIDTH-1:0] image_in [0:IMAGE_SIZE-1],
    
    //----------------------------------------------------------------------
    // Output Interface
    //----------------------------------------------------------------------
    output wire signed [DATA_WIDTH-1:0] gen_image [0:IMAGE_SIZE-1],
    output wire                         gen_valid,
    output wire signed [DATA_WIDTH-1:0] disc_score,
    output wire                         disc_valid
);

    //==========================================================================
    // Internal Wires
    //==========================================================================
    
    // Generator signals
    wire signed [DATA_WIDTH-1:0] gen_out [0:IMAGE_SIZE-1];
    wire                         gen_done;
    wire [3:0]                   gen_w1_addr;
    wire signed [WEIGHT_WIDTH-1:0] gen_w1_data;
    wire [1:0]                   gen_b1_addr;
    wire signed [DATA_WIDTH-1:0] gen_b1_data;
    wire [4:0]                   gen_w2_addr;
    wire signed [WEIGHT_WIDTH-1:0] gen_w2_data;
    wire [3:0]                   gen_b2_addr;
    wire signed [DATA_WIDTH-1:0] gen_b2_data;
    
    // Discriminator signals
    wire signed [DATA_WIDTH-1:0] disc_out;
    wire                         disc_done;
    wire [4:0]                   disc_w1_addr;
    wire signed [WEIGHT_WIDTH-1:0] disc_w1_data;
    wire [1:0]                   disc_b1_addr;
    wire signed [DATA_WIDTH-1:0] disc_b1_data;
    wire [1:0]                   disc_w2_addr;
    wire signed [WEIGHT_WIDTH-1:0] disc_w2_data;
    wire signed [DATA_WIDTH-1:0] disc_b2_data;
    
    //==========================================================================
    // State Machine
    //==========================================================================
    localparam [2:0]
        ST_IDLE      = 3'd0,
        ST_GEN_START = 3'd1,
        ST_GEN_WAIT  = 3'd2,
        ST_DISC_START = 3'd3,
        ST_DISC_WAIT = 3'd4,
        ST_DONE      = 3'd5;
    
    reg [2:0] state, next_state;
    reg [1:0] mode_reg;
    reg       gen_start_pulse;
    reg       disc_start_pulse;
    
    // Discriminator input mux: either from generator or from external input
    reg signed [DATA_WIDTH-1:0] disc_input [0:IMAGE_SIZE-1];
    
    //==========================================================================
    // State Machine Sequential Logic
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            mode_reg <= 2'b00;
            gen_start_pulse <= 0;
            disc_start_pulse <= 0;
            for (integer i = 0; i < IMAGE_SIZE; i = i + 1)
                disc_input[i] <= 0;
        end else begin
            state <= next_state;
            gen_start_pulse <= 0;
            disc_start_pulse <= 0;
            
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        mode_reg <= mode;
                        // Prepare discriminator input from external source
                        if (mode == 2'b01) begin
                            for (integer i = 0; i < IMAGE_SIZE; i = i + 1)
                                disc_input[i] <= image_in[i];
                        end
                    end
                end
                
                ST_GEN_START: begin
                    gen_start_pulse <= 1;
                end
                
                ST_GEN_WAIT: begin
                    if (gen_done && (mode_reg == 2'b10)) begin
                        // Full GAN mode: copy generator output to discriminator input
                        for (integer i = 0; i < IMAGE_SIZE; i = i + 1)
                            disc_input[i] <= gen_out[i];
                    end
                end
                
                ST_DISC_START: begin
                    disc_start_pulse <= 1;
                end
                
                default: ;
            endcase
        end
    end
    
    //==========================================================================
    // State Machine Next State Logic
    //==========================================================================
    always @(*) begin
        next_state = state;
        
        case (state)
            ST_IDLE: begin
                if (start) begin
                    case (mode)
                        2'b00: next_state = ST_GEN_START;   // Generate only
                        2'b01: next_state = ST_DISC_START;  // Discriminate only
                        2'b10: next_state = ST_GEN_START;   // Full GAN
                        default: next_state = ST_IDLE;
                    endcase
                end
            end
            
            ST_GEN_START: begin
                next_state = ST_GEN_WAIT;
            end
            
            ST_GEN_WAIT: begin
                if (gen_done) begin
                    if (mode_reg == 2'b10)
                        next_state = ST_DISC_START;  // Continue to discriminator
                    else
                        next_state = ST_DONE;        // Generate only complete
                end
            end
            
            ST_DISC_START: begin
                next_state = ST_DISC_WAIT;
            end
            
            ST_DISC_WAIT: begin
                if (disc_done)
                    next_state = ST_DONE;
            end
            
            ST_DONE: begin
                next_state = ST_IDLE;
            end
            
            default: next_state = ST_IDLE;
        endcase
    end
    
    //==========================================================================
    // Output Assignments
    //==========================================================================
    assign busy = (state != ST_IDLE);
    assign done = (state == ST_DONE);
    assign gen_valid = gen_done;
    assign disc_valid = disc_done;
    assign disc_score = disc_out;
    
    genvar gi;
    generate
        for (gi = 0; gi < IMAGE_SIZE; gi = gi + 1) begin : gen_image_assign
            assign gen_image[gi] = gen_out[gi];
        end
    endgenerate
    
    //==========================================================================
    // Generator Instance
    //==========================================================================
    simple_generator #(
        .LATENT_DIM(LATENT_DIM),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(IMAGE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) u_generator (
        .clk(clk),
        .rst_n(rst_n),
        .latent_in(latent_in),
        .valid_in(gen_start_pulse),
        .gen_out(gen_out),
        .valid_out(),  // Not used (we use done signal)
        .done(gen_done),
        .w1_addr(gen_w1_addr),
        .w1_data(gen_w1_data),
        .b1_addr(gen_b1_addr),
        .b1_data(gen_b1_data),
        .w2_addr(gen_w2_addr),
        .w2_data(gen_w2_data),
        .b2_addr(gen_b2_addr),
        .b2_data(gen_b2_data)
    );
    
    //==========================================================================
    // Discriminator Instance
    //==========================================================================
    simple_discriminator #(
        .INPUT_SIZE(IMAGE_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(1),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) u_discriminator (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(disc_input),
        .valid_in(disc_start_pulse),
        .disc_out(disc_out),
        .valid_out(),  // Not used (we use done signal)
        .done(disc_done),
        .w1_addr(disc_w1_addr),
        .w1_data(disc_w1_data),
        .b1_addr(disc_b1_addr),
        .b1_data(disc_b1_data),
        .w2_addr(disc_w2_addr),
        .w2_data(disc_w2_data),
        .b2_data(disc_b2_data)
    );
    
    //==========================================================================
    // Weight ROM Instance
    //==========================================================================
    simple_gan_weights u_weights (
        .clk(clk),
        // Generator Layer 1
        .g_w1_addr(gen_w1_addr),
        .g_w1_data(gen_w1_data),
        .g_b1_addr(gen_b1_addr),
        .g_b1_data(gen_b1_data),
        // Generator Layer 2
        .g_w2_addr(gen_w2_addr),
        .g_w2_data(gen_w2_data),
        .g_b2_addr(gen_b2_addr),
        .g_b2_data(gen_b2_data),
        // Discriminator Layer 1
        .d_w1_addr(disc_w1_addr),
        .d_w1_data(disc_w1_data),
        .d_b1_addr(disc_b1_addr),
        .d_b1_data(disc_b1_data),
        // Discriminator Layer 2
        .d_w2_addr(disc_w2_addr),
        .d_w2_data(disc_w2_data),
        .d_b2_data(disc_b2_data)
    );

endmodule
