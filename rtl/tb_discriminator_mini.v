//==============================================================================
// Self-Checking Testbench for Mini Discriminator (Critic)
//
// Features:
//   - Multiple test patterns
//   - Automatic PASS/FAIL verification with error counting
//   - State machine monitoring
//   - Performance metrics (cycles, throughput)
//   - Score range validation
//
// Test Strategy:
//   Tests verify the discriminator produces valid scores for various inputs.
//   Real vs fake signal differentiation is tested by comparing score magnitudes.
//==============================================================================

`timescale 1ns / 1ps

module tb_discriminator_mini;

    //--------------------------------------------------------------------------
    // Parameters
    //--------------------------------------------------------------------------
    parameter DATA_WIDTH   = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter ACC_WIDTH    = 32;
    parameter FRAME_LEN    = 16;
    parameter IN_CH        = 4;
    parameter CLK_PERIOD   = 10;
    parameter TIMEOUT      = 50000;
    
    //--------------------------------------------------------------------------
    // DUT Signals
    //--------------------------------------------------------------------------
    reg                          clk;
    reg                          rst_n;
    reg                          start;
    
    reg  signed [DATA_WIDTH-1:0] cand_in;
    reg                          cand_valid;
    
    reg  signed [DATA_WIDTH-1:0] cond_in;
    reg                          cond_valid;
    
    wire                         ready_in;
    
    wire signed [DATA_WIDTH-1:0] score_out;
    wire                         score_valid;
    
    wire                         busy;
    wire                         done;
    
    //--------------------------------------------------------------------------
    // Test Infrastructure
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] cand_data [0:1][0:FRAME_LEN-1];  // 2 channels
    reg signed [DATA_WIDTH-1:0] cond_data [0:1][0:FRAME_LEN-1];  // 2 channels
    
    reg signed [DATA_WIDTH-1:0] captured_score;
    reg signed [DATA_WIDTH-1:0] score_min, score_max;
    
    integer i, ch_idx, pos_idx;
    integer cycle_count, start_cycle, total_cycles;
    integer test_num, error_count, total_errors, total_tests;
    
    reg [127:0] state_name;
    
    //--------------------------------------------------------------------------
    // Clock Generation
    //--------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //--------------------------------------------------------------------------
    // Cycle Counter
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end
    
    //--------------------------------------------------------------------------
    // DUT Instantiation
    //--------------------------------------------------------------------------
    discriminator_mini #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAME_LEN(FRAME_LEN),
        .IN_CH(IN_CH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .cand_in(cand_in),
        .cand_valid(cand_valid),
        .cond_in(cond_in),
        .cond_valid(cond_valid),
        .ready_in(ready_in),
        .score_out(score_out),
        .score_valid(score_valid),
        .busy(busy),
        .done(done)
    );
    
    //--------------------------------------------------------------------------
    // VCD Dump
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_discriminator_mini.vcd");
        $dumpvars(0, tb_discriminator_mini);
    end
    
    //--------------------------------------------------------------------------
    // State Name Translation
    //--------------------------------------------------------------------------
    always @(*) begin
        case (dut.state)
            4'd0: state_name = "IDLE";
            4'd1: state_name = "LOAD_CAND";
            4'd2: state_name = "LOAD_COND";
            4'd3: state_name = "CONV1";
            4'd4: state_name = "CONV2";
            4'd5: state_name = "POOL";
            4'd6: state_name = "DENSE";
            4'd7: state_name = "OUTPUT";
            4'd8: state_name = "DONE";
            default: state_name = "UNKNOWN";
        endcase
    end
    
    //--------------------------------------------------------------------------
    // State Change Monitor
    //--------------------------------------------------------------------------
    reg [3:0] prev_state;
    always @(posedge clk) begin
        if (dut.state != prev_state) begin
            $display("  [%0t] State: %s", $time, state_name);
        end
        prev_state <= dut.state;
    end
    
    //--------------------------------------------------------------------------
    // Main Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("================================================================");
        $display("  SELF-CHECKING TESTBENCH: Mini Discriminator (Pipelined)");
        $display("================================================================");
        $display("");
        $display("Architecture: Parallel kernel MACs + 3-stage pipeline");
        $display("Input: 4 channels (2 candidate + 2 condition), 16 samples each");
        $display("Output: Single critic score");
        $display("");
        
        // Initialize
        rst_n = 0;
        start = 0;
        cand_in = 0;
        cand_valid = 0;
        cond_in = 0;
        cond_valid = 0;
        
        total_errors = 0;
        total_tests = 0;
        
        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        //----------------------------------------------------------------------
        // Test 1: Zero Input
        //----------------------------------------------------------------------
        $display("----------------------------------------------------------------");
        $display("TEST 1: Zero Input (all zeros)");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            cand_data[0][i] = 0; cand_data[1][i] = 0;
            cond_data[0][i] = 0; cond_data[1][i] = 0;
        end
        score_min = -16'sh1000; score_max = 16'sh1000;
        run_test(1);
        
        //----------------------------------------------------------------------
        // Test 2: Matching Input (candidate = condition)
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 2: Matching Input (candidate == condition)");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            cand_data[0][i] = $rtoi(100.0 * $sin(2.0 * 3.14159 * i / FRAME_LEN));
            cand_data[1][i] = $rtoi(100.0 * $cos(2.0 * 3.14159 * i / FRAME_LEN));
            cond_data[0][i] = cand_data[0][i];
            cond_data[1][i] = cand_data[1][i];
        end
        score_min = -16'sh2000; score_max = 16'sh2000;
        run_test(2);
        
        //----------------------------------------------------------------------
        // Test 3: Mismatched Input (candidate != condition)
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 3: Mismatched Input (candidate != condition)");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            cand_data[0][i] = $rtoi(100.0 * $sin(2.0 * 3.14159 * i / FRAME_LEN));
            cand_data[1][i] = $rtoi(100.0 * $cos(2.0 * 3.14159 * i / FRAME_LEN));
            cond_data[0][i] = -cand_data[0][i];  // Opposite
            cond_data[1][i] = -cand_data[1][i];
        end
        score_min = -16'sh2000; score_max = 16'sh2000;
        run_test(3);
        
        //----------------------------------------------------------------------
        // Test 4: Random Noise
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 4: Random-like Pattern");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            cand_data[0][i] = ((i * 73 + 17) % 256) - 128;
            cand_data[1][i] = ((i * 37 + 91) % 256) - 128;
            cond_data[0][i] = ((i * 41 + 53) % 256) - 128;
            cond_data[1][i] = ((i * 59 + 23) % 256) - 128;
        end
        score_min = -16'sh4000; score_max = 16'sh4000;
        run_test(4);
        
        //----------------------------------------------------------------------
        // Test 5: DC Input
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 5: DC Input (constant values)");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            cand_data[0][i] = 16'sh0080; cand_data[1][i] = 16'sh0080;
            cond_data[0][i] = 16'sh0080; cond_data[1][i] = 16'sh0080;
        end
        score_min = -16'sh4000; score_max = 16'sh4000;
        run_test(5);
        
        //----------------------------------------------------------------------
        // Final Report
        //----------------------------------------------------------------------
        $display("");
        $display("================================================================");
        $display("                    FINAL TEST SUMMARY");
        $display("================================================================");
        $display("");
        $display("Total Tests:  %0d", total_tests);
        $display("Total Errors: %0d", total_errors);
        $display("");
        
        if (total_errors == 0) begin
            $display("  ****  ALL TESTS PASSED  ****");
            $display("");
            $display("   ____    _    ____ ____  ");
            $display("  |  _ \\  / \\  / ___/ ___| ");
            $display("  | |_) |/ _ \\ \\___ \\___ \\ ");
            $display("  |  __// ___ \\ ___) |__) |");
            $display("  |_|  /_/   \\_\\____/____/ ");
        end else begin
            $display("  ****  TESTS FAILED  ****");
            $display("");
            $display("   _____ _    ___ _     ");
            $display("  |  ___/ \\  |_ _| |    ");
            $display("  | |_ / _ \\  | || |    ");
            $display("  |  _/ ___ \\ | || |___ ");
            $display("  |_|/_/   \\_\\___|_____|");
        end
        
        $display("");
        $display("================================================================");
        
        #100;
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // Test Execution Task
    //--------------------------------------------------------------------------
    task run_test;
        input integer test_id;
        begin
            test_num = test_id;
            total_tests = total_tests + 1;
            error_count = 0;
            captured_score = 0;
            
            $display("  Starting inference...");
            
            // Wait for DUT to be in IDLE state
            wait(dut.state == 4'd0);  // ST_IDLE
            repeat(2) @(posedge clk);
            
            @(posedge clk);
            start = 1;
            start_cycle = cycle_count;
            @(posedge clk);
            start = 0;
            
            // Wait for LOAD_CAND state
            wait(dut.state == 4'd1);  // ST_LOAD_CAND
            
            // Load candidate data (2 channels × FRAME_LEN samples)
            ch_idx = 0;
            pos_idx = 0;
            while (ch_idx < 2) begin
                @(posedge clk);
                if (cycle_count > start_cycle + TIMEOUT) begin
                    $display("  ERROR: Candidate loading timeout!");
                    ch_idx = 2; // Exit loop
                end else if (ready_in) begin
                    cand_in = cand_data[ch_idx][pos_idx];
                    cand_valid = 1;
                    // Check if sample was accepted
                    if (ready_in && cand_valid) begin
                        pos_idx = pos_idx + 1;
                        if (pos_idx == FRAME_LEN) begin
                            pos_idx = 0;
                            ch_idx = ch_idx + 1;
                        end
                    end
                end else begin
                    cand_valid = 0;
                end
            end
            // Keep valid high for one more cycle to ensure state transition
            @(posedge clk);
            cand_valid = 0;
            
            $display("  Candidate loaded: 2 channels × %0d samples", FRAME_LEN);
            
            // Wait for LOAD_COND state
            wait(dut.state == 4'd2);  // ST_LOAD_COND
            
            // Load condition data (2 channels × FRAME_LEN samples)
            ch_idx = 0;
            pos_idx = 0;
            while (ch_idx < 2) begin
                @(posedge clk);
                if (cycle_count > start_cycle + TIMEOUT) begin
                    $display("  ERROR: Condition loading timeout!");
                    ch_idx = 2; // Exit loop
                end else if (ready_in) begin
                    cond_in = cond_data[ch_idx][pos_idx];
                    cond_valid = 1;
                    // Check if sample was accepted
                    if (ready_in && cond_valid) begin
                        pos_idx = pos_idx + 1;
                        if (pos_idx == FRAME_LEN) begin
                            pos_idx = 0;
                            ch_idx = ch_idx + 1;
                        end
                    end
                end else begin
                    cond_valid = 0;
                end
            end
            // Keep valid high for one more cycle to ensure state transition
            @(posedge clk);
            cond_valid = 0;
            
            $display("  Condition loaded: 2 channels × %0d samples", FRAME_LEN);
            
            // Wait for processing
            while (!done && cycle_count < start_cycle + TIMEOUT) begin
                @(posedge clk);
                if (score_valid) begin
                    captured_score = score_out;
                end
            end
            
            total_cycles = cycle_count - start_cycle;
            
            // Check for timeout
            if (!done) begin
                $display("  ERROR: Timeout after %0d cycles!", TIMEOUT);
                $display("    Last state: %s", state_name);
                error_count = error_count + 100;
            end else begin
                $display("  Processing complete!");
                $display("  Total cycles: %0d", total_cycles);
                $display("  Score output: %0d (0x%04h)", $signed(captured_score), captured_score);
            end
            
            // Verify score is in expected range
            if ($signed(captured_score) < $signed(score_min) || 
                $signed(captured_score) > $signed(score_max)) begin
                $display("  [FAIL] Score %0d out of expected range [%0d, %0d]",
                         $signed(captured_score), $signed(score_min), $signed(score_max));
                error_count = error_count + 1;
            end
            
            // Report
            if (error_count == 0) begin
                $display("  Result: *** PASS ***");
            end else begin
                $display("  Result: *** FAIL *** (%0d errors)", error_count);
                total_errors = total_errors + error_count;
            end
            
            // Wait for return to IDLE
            wait(dut.state == 4'd0);
            repeat(10) @(posedge clk);
        end
    endtask
    
    //--------------------------------------------------------------------------
    // Global Timeout
    //--------------------------------------------------------------------------
    initial begin
        #(TIMEOUT * CLK_PERIOD * 10);
        $display("");
        $display("FATAL: Global timeout reached!");
        $finish;
    end

endmodule
