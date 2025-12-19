//==============================================================================
// Debug Testbench for Simple Generator
// Traces through MAC operations step by step
//==============================================================================

`timescale 1ns / 1ps

module tb_debug_gen;

    parameter CLK_PERIOD = 10;
    parameter DATA_WIDTH = 16;
    parameter WEIGHT_WIDTH = 8;
    
    reg                          clk;
    reg                          rst_n;
    
    // DUT signals
    reg  signed [DATA_WIDTH-1:0] latent_in [0:1];
    reg                          valid_in;
    wire signed [DATA_WIDTH-1:0] gen_out [0:8];
    wire                         valid_out;
    wire                         done;
    
    // Weight ROM signals
    wire [3:0]  w1_addr;
    wire [1:0]  b1_addr;
    wire [4:0]  w2_addr;
    wire [3:0]  b2_addr;
    reg signed [WEIGHT_WIDTH-1:0] w1_data;
    reg signed [DATA_WIDTH-1:0]  b1_data;
    reg signed [WEIGHT_WIDTH-1:0] w2_data;
    reg signed [DATA_WIDTH-1:0]  b2_data;
    
    // Weight storage (same as ROM)
    reg signed [WEIGHT_WIDTH-1:0] gen_w1 [0:5];
    reg signed [DATA_WIDTH-1:0]  gen_b1 [0:2];
    reg signed [WEIGHT_WIDTH-1:0] gen_w2 [0:26];
    reg signed [DATA_WIDTH-1:0]  gen_b2 [0:8];
    
    // DUT
    simple_generator #(
        .LATENT_DIM(2),
        .HIDDEN_SIZE(3),
        .OUTPUT_SIZE(9),
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .latent_in(latent_in),
        .valid_in(valid_in),
        .gen_out(gen_out),
        .valid_out(valid_out),
        .done(done),
        .w1_addr(w1_addr),
        .w1_data(w1_data),
        .b1_addr(b1_addr),
        .b1_data(b1_data),
        .w2_addr(w2_addr),
        .w2_data(w2_data),
        .b2_addr(b2_addr),
        .b2_data(b2_data)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Synchronous weight ROM
    always @(posedge clk) begin
        w1_data <= gen_w1[w1_addr];
        b1_data <= gen_b1[b1_addr];
        w2_data <= gen_w2[w2_addr];
        b2_data <= gen_b2[b2_addr];
    end
    
    // Initialize weights
    initial begin
        // Generator Layer 1 weights (2->3)
        gen_w1[0] = 8'sd7;    //  0.0538
        gen_w1[1] = 8'sd11;   //  0.0862
        gen_w1[2] = 8'sd23;   //  0.1834
        gen_w1[3] = 8'sd4;    //  0.0319
        gen_w1[4] = -8'sd29;  // -0.2259
        gen_w1[5] = -8'sd17;  // -0.1308
        
        // Biases = 0
        gen_b1[0] = 16'sd0;
        gen_b1[1] = 16'sd0;
        gen_b1[2] = 16'sd0;
        
        // Layer 2 weights (3->9)
        gen_w2[0]  = -8'sd6;   gen_w2[1]  = -8'sd3;   gen_w2[2]  = 8'sd6;
        gen_w2[3]  = 8'sd4;    gen_w2[4]  = -8'sd2;   gen_w2[5]  = 8'sd13;
        gen_w2[6]  = 8'sd46;   gen_w2[7]  = 8'sd19;   gen_w2[8]  = 8'sd9;
        gen_w2[9]  = 8'sd35;   gen_w2[10] = 8'sd18;   gen_w2[11] = -8'sd4;
        gen_w2[12] = -8'sd17;  gen_w2[13] = 8'sd18;   gen_w2[14] = 8'sd4;
        gen_w2[15] = 8'sd39;   gen_w2[16] = 8'sd9;    gen_w2[17] = -8'sd10;
        gen_w2[18] = 8'sd9;    gen_w2[19] = -8'sd15;  gen_w2[20] = 8'sd11;
        gen_w2[21] = -8'sd1;   gen_w2[22] = 8'sd9;    gen_w2[23] = -8'sd15;
        gen_w2[24] = 8'sd9;    gen_w2[25] = 8'sd21;   gen_w2[26] = -8'sd14;
        
        // Biases = 0
        for (integer i = 0; i < 9; i = i + 1)
            gen_b2[i] = 16'sd0;
    end
    
    // Monitor internal signals
    always @(posedge clk) begin
        if (dut.state != 0) begin
            $display("t=%0t state=%0d out_idx=%0d in_idx=%0d acc=%0d hidden=[%0d,%0d,%0d] output_reg=%0d tanh_in=%0d tanh_out=%0d tanh_valid=%0b",
                     $time, dut.state, dut.out_idx, dut.in_idx, dut.accumulator,
                     dut.hidden[0], dut.hidden[1], dut.hidden[2],
                     dut.output_reg[dut.out_idx], dut.tanh_in, dut.u_tanh.data_out, dut.u_tanh.valid_out);
        end
    end
    
    // Helper function
    function real q88_to_real;
        input signed [15:0] val;
        begin
            q88_to_real = $itor(val) / 256.0;
        end
    endfunction
    
    // Test
    initial begin
        $display("=== Debug Generator Test ===");
        rst_n = 0;
        valid_in = 0;
        latent_in[0] = 16'h0000;
        latent_in[1] = 16'h0000;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        
        // Set latent vector: [1.5442, 0.0859] in Q8.8
        latent_in[0] = 16'h018B;  // 1.5442 * 256 = 395
        latent_in[1] = 16'h0016;  // 0.0859 * 256 = 22
        
        $display("Latent[0] = 0x%04X = %f", latent_in[0], q88_to_real(latent_in[0]));
        $display("Latent[1] = 0x%04X = %f", latent_in[1], q88_to_real(latent_in[1]));
        
        // Start
        @(posedge clk);
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for done
        wait(done);
        @(posedge clk);
        
        $display("\n=== Generator Output ===");
        for (integer i = 0; i < 9; i = i + 1) begin
            $display("gen_out[%0d] = 0x%04X = %f", i, gen_out[i], q88_to_real(gen_out[i]));
        end
        
        $display("\n=== Expected (MATLAB) ===");
        $display("[-0.0265, -0.0361, 0.0486, 0.0745, 0.0171, 0.0731, -0.0577, 0.0589, 0.0885]");
        
        repeat(10) @(posedge clk);
        $finish;
    end
    
endmodule
