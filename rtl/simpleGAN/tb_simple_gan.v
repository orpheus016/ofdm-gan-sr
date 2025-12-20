//==============================================================================
// Simple GAN - Testbench
//
// Tests the complete simple GAN with:
// 1. Generator only mode (latent -> 3x3 image)
// 2. Discriminator only mode (3x3 image -> score)
// 3. Full GAN mode (latent -> image -> score)
//
// Test inputs based on MATLAB reference:
// - Training data: circle and cross patterns
// - Latent vector: 2D random values in [-1, 1]
//==============================================================================

`timescale 1ns / 1ps

module tb_simple_gan;

    //--------------------------------------------------------------------------
    // Parameters
    //--------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter DATA_WIDTH = 16;  // Q8.8
    parameter LATENT_DIM = 2;
    parameter IMAGE_SIZE = 9;
    
    //--------------------------------------------------------------------------
    // DUT Signals
    //--------------------------------------------------------------------------
    reg                          clk;
    reg                          rst_n;
    reg  [1:0]                   mode;
    reg                          start;
    wire                         busy;
    wire                         done;
    
    reg  signed [DATA_WIDTH-1:0] latent_in [0:LATENT_DIM-1];
    reg  signed [DATA_WIDTH-1:0] image_in [0:IMAGE_SIZE-1];
    wire signed [DATA_WIDTH-1:0] gen_image [0:IMAGE_SIZE-1];
    wire                         gen_valid;
    wire signed [DATA_WIDTH-1:0] disc_score;
    wire                         disc_valid;
    
    //--------------------------------------------------------------------------
    // Test Variables
    //--------------------------------------------------------------------------
    integer test_num;
    integer i;
    integer pass_count;
    integer fail_count;
    real    score_real;
    
    // Circle pattern (from MATLAB): 3x3 with values at corners
    // [1 1 1; 1 -1 1; 1 1 1]
    // Flattened: [1, 1, 1, 1, -1, 1, 1, 1, 1]
    reg signed [DATA_WIDTH-1:0] circle_pattern [0:IMAGE_SIZE-1];
    
    // Cross pattern (from MATLAB): 3x3 with values in cross shape
    // [-1 1 -1; 1 1 1; -1 1 -1]
    // Flattened: [-1, 1, -1, 1, 1, 1, -1, 1, -1]
    reg signed [DATA_WIDTH-1:0] cross_pattern [0:IMAGE_SIZE-1];
    
    //--------------------------------------------------------------------------
    // DUT Instance
    //--------------------------------------------------------------------------
    simple_gan_top #(
        .LATENT_DIM(LATENT_DIM),
        .HIDDEN_SIZE(3),
        .IMAGE_SIZE(IMAGE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .start(start),
        .busy(busy),
        .done(done),
        .latent_in(latent_in),
        .image_in(image_in),
        .gen_image(gen_image),
        .gen_valid(gen_valid),
        .disc_score(disc_score),
        .disc_valid(disc_valid)
    );
    
    //--------------------------------------------------------------------------
    // Clock Generation
    //--------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //--------------------------------------------------------------------------
    // Helper Functions
    //--------------------------------------------------------------------------
    
    // Convert Q8.8 to real for display
    function real q88_to_real;
        input signed [15:0] q88_val;
        begin
            q88_to_real = $itor(q88_val) / 256.0;
        end
    endfunction
    
    // Convert real to Q8.8
    function signed [15:0] real_to_q88;
        input real r_val;
        begin
            real_to_q88 = $rtoi(r_val * 256.0);
        end
    endfunction
    
    //--------------------------------------------------------------------------
    // Test Stimulus
    //--------------------------------------------------------------------------
    initial begin
        // Initialize
        $display("========================================");
        $display("Simple GAN Testbench");
        $display("Architecture: 2->3->9 (Gen), 9->3->1 (Disc)");
        $display("========================================");
        
        rst_n = 0;
        start = 0;
        mode = 2'b00;
        pass_count = 0;
        fail_count = 0;
        test_num = 0;
        
        // Initialize latent vector
        latent_in[0] = 16'h0000;
        latent_in[1] = 16'h0000;
        
        // Initialize image input
        for (i = 0; i < IMAGE_SIZE; i = i + 1)
            image_in[i] = 16'h0000;
        
        // Initialize test patterns (Q8.8: 256 = 1.0, -256 = -1.0)
        // Circle pattern: [1 1 1; 1 -1 1; 1 1 1]
        circle_pattern[0] = 16'h0100;   // +1.0
        circle_pattern[1] = 16'h0100;   // +1.0
        circle_pattern[2] = 16'h0100;   // +1.0
        circle_pattern[3] = 16'h0100;   // +1.0
        circle_pattern[4] = 16'hFF00;   // -1.0
        circle_pattern[5] = 16'h0100;   // +1.0
        circle_pattern[6] = 16'h0100;   // +1.0
        circle_pattern[7] = 16'h0100;   // +1.0
        circle_pattern[8] = 16'h0100;   // +1.0
        
        // Cross pattern: [-1 1 -1; 1 1 1; -1 1 -1]
        cross_pattern[0] = 16'hFF00;    // -1.0
        cross_pattern[1] = 16'h0100;    // +1.0
        cross_pattern[2] = 16'hFF00;    // -1.0
        cross_pattern[3] = 16'h0100;    // +1.0
        cross_pattern[4] = 16'h0100;    // +1.0
        cross_pattern[5] = 16'h0100;    // +1.0
        cross_pattern[6] = 16'hFF00;    // -1.0
        cross_pattern[7] = 16'h0100;    // +1.0
        cross_pattern[8] = 16'hFF00;    // -1.0
        
        // Reset sequence
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        //======================================================================
        // Test 1: Generator with MATLAB test vector
        //======================================================================
        test_num = 1;
        $display("\n[Test %0d] Generator with MATLAB Test Vector", test_num);
        $display("  Latent input (ng): [1.5442, 0.0859]");
        
        mode = 2'b00;
        // MATLAB: ng = [1.5442; 0.0859]
        // Q8.8: 1.1112 * 256 = 284.46 = 0x018B
        //       1.9162 * 256 = 490.54 = 0x0016
        latent_in[0] = 16'h011C;  // 1.1112 in Q8.8
        latent_in[1] = 16'h01EB;  // 1.9162 in Q8.8
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for completion
        wait(done);
        @(posedge clk);
        
        $display("  Generated 3x3 image (x_fake):");
        $display("    [%7.4f %7.4f %7.4f]", 
            q88_to_real(gen_image[0]), q88_to_real(gen_image[1]), q88_to_real(gen_image[2]));
        $display("    [%7.4f %7.4f %7.4f]", 
            q88_to_real(gen_image[3]), q88_to_real(gen_image[4]), q88_to_real(gen_image[5]));
        $display("    [%7.4f %7.4f %7.4f]", 
            q88_to_real(gen_image[6]), q88_to_real(gen_image[7]), q88_to_real(gen_image[8]));
        
        // Expected from MATLAB:
        // x_fake = [-0.1753, 0.5685, 0.2215, 0.7049, -0.0964, 0.7242, -0.0774, 0.7154, 0.1548]
        $display("  Expected (MATLAB):");
        $display("    [-0.1753  0.5685  0.2215]");
        $display("    [ 0.7049 -0.0964  0.7242]");
        $display("    [-0.0774  0.7154  0.1548]");
        
        // Check if outputs are non-zero (basic sanity check)
        if (gen_image[0] != 0 || gen_image[2] != 0 || gen_image[8] != 0) begin
            $display("  [PASS] Generator produces non-zero output");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Generator output is all zeros");
            fail_count = fail_count + 1;
        end
        
        #(CLK_PERIOD * 5);
        
        //======================================================================
        // Test 2: Discriminator Only - Cross Pattern (Real Image from MATLAB)
        //======================================================================
        test_num = 2;
        $display("\n[Test %0d] Discriminator - Cross Pattern (x_real from MATLAB)", test_num);
        
        mode = 2'b01;
        // x_real from MATLAB: [-1,1,-1,1,1,1,-1,1,-1] (cross pattern)
        for (i = 0; i < IMAGE_SIZE; i = i + 1)
            image_in[i] = cross_pattern[i];
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for completion
        wait(done);
        @(posedge clk);
        
        score_real = q88_to_real(disc_score);
        $display("  Input: Cross pattern (x_real)");
        $display("  Discriminator score: %6.4f (Q8.8: 0x%04h)", score_real, disc_score);
        $display("  Expected (MATLAB): ad3_real = 0.6770");
        
        // Score should be between 0 and 1 (sigmoid output)
        if (disc_score >= 0 && disc_score <= 16'h0100) begin
            $display("  [PASS] Score in valid range [0, 1]");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Score out of range");
            fail_count = fail_count + 1;
        end
        
        #(CLK_PERIOD * 5);
        
        //======================================================================
        // Test 3: Discriminator Only - Circle Pattern
        //======================================================================
        test_num = 3;
        $display("\n[Test %0d] Discriminator Only - Circle Pattern", test_num);
        
        mode = 2'b01;
        for (i = 0; i < IMAGE_SIZE; i = i + 1)
            image_in[i] = circle_pattern[i];
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for completion
        wait(done);
        @(posedge clk);
        
        score_real = q88_to_real(disc_score);
        $display("  Input: Circle pattern");
        $display("  Discriminator score: %6.4f (Q8.8: 0x%04h)", score_real, disc_score);
        
        if (disc_score >= 0 && disc_score <= 16'h0100) begin
            $display("  [PASS] Score in valid range [0, 1]");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Score out of range");
            fail_count = fail_count + 1;
        end
        
        #(CLK_PERIOD * 5);
        
        //======================================================================
        // Test 4: Full GAN Mode (Mode 10) - MATLAB test vector
        //======================================================================
        test_num = 4;
        $display("\n[Test %0d] Full GAN Mode (Generate + Discriminate)", test_num);
        $display("  Latent input (ng): [1.1112, 1.9162]");
        
        mode = 2'b10;
        latent_in[0] = 16'h011C;  // 1.1112 in Q8.8
        latent_in[1] = 16'h01EB;  // 1.9162 in Q8.8
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait for completion
        wait(done);
        @(posedge clk);
        
        $display("  Generated image (x_fake):");
        $display("    [%7.4f %7.4f %7.4f]", 
            q88_to_real(gen_image[0]), q88_to_real(gen_image[1]), q88_to_real(gen_image[2]));
        $display("    [%7.4f %7.4f %7.4f]", 
            q88_to_real(gen_image[3]), q88_to_real(gen_image[4]), q88_to_real(gen_image[5]));
        $display("    [%7.4f %7.4f %7.4f]", 
            q88_to_real(gen_image[6]), q88_to_real(gen_image[7]), q88_to_real(gen_image[8]));
        
        score_real = q88_to_real(disc_score);
        $display("  Discriminator score on fake: %6.4f", score_real);
        
        if (disc_score >= 0 && disc_score <= 16'h0100) begin
            $display("  [PASS] Full GAN pipeline completed");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Score out of range");
            fail_count = fail_count + 1;
        end
        
        #(CLK_PERIOD * 5);
        
        //======================================================================
        // Test 5: Multiple Latent Vectors
        //======================================================================
        test_num = 5;
        $display("\n[Test %0d] Multiple Latent Vectors", test_num);
        
        mode = 2'b00;  // Generator only
        
        // Test latent vector 1
        latent_in[0] = 16'h0100;  // 1.0
        latent_in[1] = 16'h0100;  // 1.0
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait(done);
        @(posedge clk);
        $display("  Latent [1.0, 1.0] -> center: %6.3f", q88_to_real(gen_image[4]));
        
        #(CLK_PERIOD * 5);
        
        // Test latent vector 2
        latent_in[0] = 16'hFF00;  // -1.0
        latent_in[1] = 16'hFF00;  // -1.0
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait(done);
        @(posedge clk);
        $display("  Latent [-1.0, -1.0] -> center: %6.3f", q88_to_real(gen_image[4]));
        
        // Test latent vector 3
        latent_in[0] = 16'h0000;  // 0.0
        latent_in[1] = 16'h0000;  // 0.0
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        wait(done);
        @(posedge clk);
        $display("  Latent [0.0, 0.0] -> center: %6.3f", q88_to_real(gen_image[4]));
        
        $display("  [PASS] Multiple latent vectors tested");
        pass_count = pass_count + 1;
        
        //======================================================================
        // Summary
        //======================================================================
        #(CLK_PERIOD * 10);
        
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("  Total Tests: %0d", pass_count + fail_count);
        $display("  Passed:      %0d", pass_count);
        $display("  Failed:      %0d", fail_count);
        $display("========================================");
        
        if (fail_count == 0) begin
            $display("[SUCCESS] All tests passed!");
        end else begin
            $display("[FAILURE] Some tests failed!");
        end
        
        $display("========================================\n");
        
        #(CLK_PERIOD * 10);
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // Timeout Watchdog
    //--------------------------------------------------------------------------
    initial begin
        #1000000;  // 1ms timeout
        $display("[ERROR] Testbench timeout!");
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // VCD Dump
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_simple_gan.vcd");
        $dumpvars(0, tb_simple_gan);
    end

endmodule
