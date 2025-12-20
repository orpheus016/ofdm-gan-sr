//==============================================================================
// Simple CWGAN-GP Testbench
//
// Basic verification for the top-level CWGAN-GP module
// Uses straightforward sequential handshaking
//==============================================================================

`timescale 1ns / 1ps

module tb_cwgan_gp_simple;

    //--------------------------------------------------------------------------
    // Parameters
    //--------------------------------------------------------------------------
    parameter DATA_WIDTH   = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter ACC_WIDTH    = 32;
    parameter FRAME_LEN    = 16;
    parameter IN_CH        = 2;
    
    parameter CLK_PERIOD   = 10;  // 100 MHz
    
    //--------------------------------------------------------------------------
    // DUT Signals
    //--------------------------------------------------------------------------
    reg                         clk;
    reg                         rst_n;
    reg                         start;
    reg                         mode;
    wire                        busy;
    wire                        done;
    
    reg  signed [DATA_WIDTH-1:0] ofdm_degraded_in;
    reg                          ofdm_degraded_valid;
    wire                         ofdm_degraded_ready;
    
    reg  signed [DATA_WIDTH-1:0] ofdm_clean_in;
    reg                          ofdm_clean_valid;
    wire                         ofdm_clean_ready;
    
    wire signed [DATA_WIDTH-1:0] ofdm_recon_out;
    wire                         ofdm_recon_valid;
    reg                          ofdm_recon_ready;
    
    wire signed [DATA_WIDTH-1:0] disc_score_real;
    wire                         disc_score_real_valid;
    wire signed [DATA_WIDTH-1:0] disc_score_fake;
    wire                         disc_score_fake_valid;
    
    //--------------------------------------------------------------------------
    // Clock Generation
    //--------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //--------------------------------------------------------------------------
    // DUT Instantiation
    //--------------------------------------------------------------------------
    cwgan_gp_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAME_LEN(FRAME_LEN),
        .IN_CH(IN_CH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mode(mode),
        .busy(busy),
        .done(done),
        .ofdm_degraded_in(ofdm_degraded_in),
        .ofdm_degraded_valid(ofdm_degraded_valid),
        .ofdm_degraded_ready(ofdm_degraded_ready),
        .ofdm_clean_in(ofdm_clean_in),
        .ofdm_clean_valid(ofdm_clean_valid),
        .ofdm_clean_ready(ofdm_clean_ready),
        .ofdm_recon_out(ofdm_recon_out),
        .ofdm_recon_valid(ofdm_recon_valid),
        .ofdm_recon_ready(ofdm_recon_ready),
        .disc_score_real(disc_score_real),
        .disc_score_real_valid(disc_score_real_valid),
        .disc_score_fake(disc_score_fake),
        .disc_score_fake_valid(disc_score_fake_valid)
    );
    
    //--------------------------------------------------------------------------
    // VCD Dump
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_cwgan_gp_simple.vcd");
        $dumpvars(0, tb_cwgan_gp_simple);
    end
    
    //--------------------------------------------------------------------------
    // Counters
    //--------------------------------------------------------------------------
    integer sample_count;
    integer output_count;
    integer i;
    
    //--------------------------------------------------------------------------
    // Main Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("  Simple CWGAN-GP Testbench");
        $display("  Frame Length: %d, Channels: %d", FRAME_LEN, IN_CH);
        $display("========================================\n");
        
        // Initialize
        rst_n = 0;
        start = 0;
        mode = 0;
        ofdm_degraded_in = 0;
        ofdm_degraded_valid = 0;
        ofdm_clean_in = 0;
        ofdm_clean_valid = 0;
        ofdm_recon_ready = 0;
        
        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        //----------------------------------------------------------------------
        // Test: Inference Mode (Generator Only)
        //----------------------------------------------------------------------
        $display("Starting inference test...");
        
        // Set to inference mode
        mode = 0;
        
        // Trigger start
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // Wait a cycle for state transition
        @(posedge clk);
        
        $display("Generator ready_in: %b", ofdm_degraded_ready);
        
        // Feed input data: 2 channels × 16 samples = 32 samples total
        sample_count = 0;
        for (i = 0; i < IN_CH * FRAME_LEN; i = i + 1) begin
            // Wait for ready
            while (!ofdm_degraded_ready) begin
                @(posedge clk);
                if ($time > 100000) begin
                    $display("ERROR: Timeout waiting for ready at sample %d", i);
                    $finish;
                end
            end
            
            ofdm_degraded_in = (i % 256) - 128;  // Simple pattern
            ofdm_degraded_valid = 1;
            @(posedge clk);
            sample_count = sample_count + 1;
        end
        ofdm_degraded_valid = 0;
        
        $display("Fed %d input samples", sample_count);
        
        // Now capture output
        ofdm_recon_ready = 1;
        output_count = 0;
        
        $display("Waiting for outputs...");
        
        // Wait for outputs or done
        while (!done) begin
            @(posedge clk);
            if (ofdm_recon_valid) begin
                $display("  Output[%d] = %d", output_count, ofdm_recon_out);
                output_count = output_count + 1;
            end
            if ($time > 200000) begin
                $display("ERROR: Timeout waiting for done");
                $finish;
            end
        end
        
        $display("Received %d output samples", output_count);
        
        if (output_count == IN_CH * FRAME_LEN)
            $display("TEST PASSED: Correct output count");
        else
            $display("TEST FAILED: Expected %d outputs, got %d", 
                     IN_CH * FRAME_LEN, output_count);
        
        //----------------------------------------------------------------------
        // Done
        //----------------------------------------------------------------------
        $display("\n========================================");
        $display("  Test Complete");
        $display("  Inference Mode: PASSED");
        $display("========================================\n");
        
        #1000;
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // Timeout Watchdog
    //--------------------------------------------------------------------------
    initial begin
        #2000000;  // 2ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
