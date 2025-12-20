//==============================================================================
// CWGAN-GP Top Level Module
//
// Integrates Generator and Discriminator for OFDM Signal Reconstruction
//
// Modes:
//   - INFERENCE: Generator only (for deployment)
//   - TRAINING:  Generator + Discriminator (for verification)
//
// Interface:
//   - AXI-Stream style data interface
//   - Control/Status registers
//
// Fixed-Point: Q8.8 activations, Q1.7 weights
//==============================================================================

`timescale 1ns / 1ps

module cwgan_gp_top #(
    parameter DATA_WIDTH   = 16,           // Q8.8 activations
    parameter WEIGHT_WIDTH = 8,            // Q1.7 weights
    parameter ACC_WIDTH    = 32,           // Accumulator width
    parameter FRAME_LEN    = 16,           // Frame length (samples)
    parameter IN_CH        = 2             // Input channels (I, Q)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // Control Interface
    //--------------------------------------------------------------------------
    input  wire                         start,
    input  wire                         mode,           // 0: inference, 1: training
    output wire                         busy,
    output wire                         done,
    
    //--------------------------------------------------------------------------
    // Degraded OFDM Input (condition signal)
    //--------------------------------------------------------------------------
    input  wire signed [DATA_WIDTH-1:0] ofdm_degraded_in,
    input  wire                         ofdm_degraded_valid,
    output wire                         ofdm_degraded_ready,
    
    //--------------------------------------------------------------------------
    // Clean OFDM Input (for training - real sample)
    //--------------------------------------------------------------------------
    input  wire signed [DATA_WIDTH-1:0] ofdm_clean_in,
    input  wire                         ofdm_clean_valid,
    output wire                         ofdm_clean_ready,
    
    //--------------------------------------------------------------------------
    // Reconstructed OFDM Output
    //--------------------------------------------------------------------------
    output wire signed [DATA_WIDTH-1:0] ofdm_recon_out,
    output wire                         ofdm_recon_valid,
    input  wire                         ofdm_recon_ready,
    
    //--------------------------------------------------------------------------
    // Discriminator Score Output (training mode)
    //--------------------------------------------------------------------------
    output wire signed [DATA_WIDTH-1:0] disc_score_real,   // D(real, condition)
    output wire                         disc_score_real_valid,
    output wire signed [DATA_WIDTH-1:0] disc_score_fake,   // D(G(cond), condition)
    output wire                         disc_score_fake_valid
);

    //--------------------------------------------------------------------------
    // Internal Signals
    //--------------------------------------------------------------------------
    
    // Generator signals
    wire gen_busy, gen_done;
    wire gen_ready_in;
    wire signed [DATA_WIDTH-1:0] gen_out;
    wire gen_out_valid;
    
    // Condition buffer (store for discriminator)
    reg signed [DATA_WIDTH-1:0] cond_buf [0:IN_CH-1][0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] clean_buf [0:IN_CH-1][0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] fake_buf [0:IN_CH-1][0:FRAME_LEN-1];
    
    // State machine for training flow
    localparam ST_IDLE       = 3'd0;
    localparam ST_GEN        = 3'd1;
    localparam ST_DISC_FAKE  = 3'd2;
    localparam ST_DISC_REAL  = 3'd3;
    localparam ST_DONE       = 3'd4;
    
    reg [2:0] state, next_state;
    
    // Counters for buffer management
    reg [$clog2(IN_CH)-1:0]     buf_ch_cnt;
    reg [$clog2(FRAME_LEN)-1:0] buf_pos_cnt;
    
    // Discriminator interface mux
    reg  signed [DATA_WIDTH-1:0] disc_cand_in;
    reg                          disc_cand_valid;
    reg  signed [DATA_WIDTH-1:0] disc_cond_in;
    reg                          disc_cond_valid;
    wire                         disc_ready;
    wire signed [DATA_WIDTH-1:0] disc_score;
    wire                         disc_score_valid;
    wire                         disc_busy, disc_done;
    
    // Score registers
    reg signed [DATA_WIDTH-1:0] score_fake_reg;
    reg signed [DATA_WIDTH-1:0] score_real_reg;
    reg score_fake_valid_reg;
    reg score_real_valid_reg;
    
    integer i, j;
    
    //--------------------------------------------------------------------------
    // Generator Instance
    //--------------------------------------------------------------------------
    generator_mini #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAME_LEN(FRAME_LEN),
        .IN_CH(IN_CH),
        .OUT_CH(IN_CH)
    ) u_generator (
        .clk(clk),
        .rst_n(rst_n),
        .start(start && (state == ST_IDLE)),
        .data_in(ofdm_degraded_in),
        .valid_in(ofdm_degraded_valid),
        .ready_in(gen_ready_in),
        .cond_in(ofdm_degraded_in),
        .cond_valid(ofdm_degraded_valid),
        .data_out(gen_out),
        .valid_out(gen_out_valid),
        .ready_out(ofdm_recon_ready),
        .busy(gen_busy),
        .done(gen_done)
    );
    
    //--------------------------------------------------------------------------
    // Discriminator Instance (for training mode)
    //--------------------------------------------------------------------------
    discriminator_mini #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAME_LEN(FRAME_LEN),
        .IN_CH(4)  // 2 candidate + 2 condition
    ) u_discriminator (
        .clk(clk),
        .rst_n(rst_n),
        .start((state == ST_DISC_FAKE || state == ST_DISC_REAL) && 
               buf_ch_cnt == 0 && buf_pos_cnt == 0),
        .cand_in(disc_cand_in),
        .cand_valid(disc_cand_valid),
        .cond_in(disc_cond_in),
        .cond_valid(disc_cond_valid),
        .ready_in(disc_ready),
        .score_out(disc_score),
        .score_valid(disc_score_valid),
        .busy(disc_busy),
        .done(disc_done)
    );
    
    //--------------------------------------------------------------------------
    // State Machine (Training Flow)
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
                    next_state = ST_GEN;
            end
            ST_GEN: begin
                if (gen_done) begin
                    if (mode)  // Training mode
                        next_state = ST_DISC_FAKE;
                    else       // Inference mode
                        next_state = ST_DONE;
                end
            end
            ST_DISC_FAKE: begin
                if (disc_done)
                    next_state = ST_DISC_REAL;
            end
            ST_DISC_REAL: begin
                if (disc_done)
                    next_state = ST_DONE;
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Buffer Management
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_ch_cnt <= 0;
            buf_pos_cnt <= 0;
            for (i = 0; i < IN_CH; i = i + 1) begin
                for (j = 0; j < FRAME_LEN; j = j + 1) begin
                    cond_buf[i][j] <= 0;
                    clean_buf[i][j] <= 0;
                    fake_buf[i][j] <= 0;
                end
            end
        end else begin
            // Store condition input
            if (ofdm_degraded_valid && gen_ready_in) begin
                cond_buf[buf_ch_cnt][buf_pos_cnt] <= ofdm_degraded_in;
                
                if (buf_pos_cnt == FRAME_LEN-1) begin
                    buf_pos_cnt <= 0;
                    if (buf_ch_cnt == IN_CH-1)
                        buf_ch_cnt <= 0;
                    else
                        buf_ch_cnt <= buf_ch_cnt + 1;
                end else begin
                    buf_pos_cnt <= buf_pos_cnt + 1;
                end
            end
            
            // Store clean input (training)
            if (mode && ofdm_clean_valid) begin
                clean_buf[buf_ch_cnt][buf_pos_cnt] <= ofdm_clean_in;
            end
            
            // Store generator output (fake)
            if (gen_out_valid) begin
                fake_buf[buf_ch_cnt][buf_pos_cnt] <= gen_out;
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Discriminator Input Mux
    //--------------------------------------------------------------------------
    always @(*) begin
        case (state)
            ST_DISC_FAKE: begin
                disc_cand_in = fake_buf[buf_ch_cnt][buf_pos_cnt];
                disc_cand_valid = 1'b1;
                disc_cond_in = cond_buf[buf_ch_cnt][buf_pos_cnt];
                disc_cond_valid = 1'b1;
            end
            ST_DISC_REAL: begin
                disc_cand_in = clean_buf[buf_ch_cnt][buf_pos_cnt];
                disc_cand_valid = 1'b1;
                disc_cond_in = cond_buf[buf_ch_cnt][buf_pos_cnt];
                disc_cond_valid = 1'b1;
            end
            default: begin
                disc_cand_in = 0;
                disc_cand_valid = 1'b0;
                disc_cond_in = 0;
                disc_cond_valid = 1'b0;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Score Capture
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score_fake_reg <= 0;
            score_real_reg <= 0;
            score_fake_valid_reg <= 1'b0;
            score_real_valid_reg <= 1'b0;
        end else begin
            if (state == ST_DISC_FAKE && disc_score_valid) begin
                score_fake_reg <= disc_score;
                score_fake_valid_reg <= 1'b1;
            end
            if (state == ST_DISC_REAL && disc_score_valid) begin
                score_real_reg <= disc_score;
                score_real_valid_reg <= 1'b1;
            end
            if (state == ST_IDLE) begin
                score_fake_valid_reg <= 1'b0;
                score_real_valid_reg <= 1'b0;
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    assign ofdm_degraded_ready = gen_ready_in;
    assign ofdm_clean_ready = mode && (state == ST_GEN);
    
    assign ofdm_recon_out = gen_out;
    assign ofdm_recon_valid = gen_out_valid;
    
    assign disc_score_fake = score_fake_reg;
    assign disc_score_fake_valid = score_fake_valid_reg;
    assign disc_score_real = score_real_reg;
    assign disc_score_real_valid = score_real_valid_reg;
    
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

endmodule
