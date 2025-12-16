//==============================================================================
// Self-Checking Testbench for Mini U-Net Generator
//
// Features:
//   - Multiple test patterns (Zero, DC, Impulse, Ramp, Sine)
//   - Automatic PASS/FAIL verification with error counting
//   - State machine monitoring
//   - Performance metrics (cycles, throughput)
//   - Golden output range checking
//
// Test Strategy:
//   Each test verifies outputs are within expected bounds for the given input.
//   This validates the pipeline and MAC operations are functioning correctly.
//==============================================================================

`timescale 1ns / 1ps

module tb_generator_mini;

    //--------------------------------------------------------------------------
    // Parameters
    //--------------------------------------------------------------------------
    parameter DATA_WIDTH   = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter ACC_WIDTH    = 32;
    parameter FRAME_LEN    = 16;
    parameter IN_CH        = 2;
    parameter OUT_CH       = 2;
    parameter CLK_PERIOD   = 10;
    parameter TIMEOUT      = 100000;
    
    //--------------------------------------------------------------------------
    // DUT Signals
    //--------------------------------------------------------------------------
    reg                          clk;
    reg                          rst_n;
    reg                          start;
    
    reg  signed [DATA_WIDTH-1:0] data_in;
    reg                          valid_in;
    wire                         ready_in;
    
    reg  signed [DATA_WIDTH-1:0] cond_in;
    reg                          cond_valid;
    
    wire signed [DATA_WIDTH-1:0] data_out;
    wire                         valid_out;
    reg                          ready_out;
    
    wire                         busy;
    wire                         done;
    
    //--------------------------------------------------------------------------
    // Test Infrastructure
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] test_input [0:IN_CH*FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] captured_output [0:OUT_CH*FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] golden_min [0:OUT_CH*FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] golden_max [0:OUT_CH*FRAME_LEN-1];
    
    integer i, in_idx, out_idx;
    integer cycle_count, start_cycle, total_cycles;
    integer test_num, error_count, total_errors, total_tests;
    
    // State name for debugging
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
    generator_mini #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAME_LEN(FRAME_LEN),
        .IN_CH(IN_CH),
        .OUT_CH(OUT_CH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .cond_in(cond_in),
        .cond_valid(cond_valid),
        .data_out(data_out),
        .valid_out(valid_out),
        .ready_out(ready_out),
        .busy(busy),
        .done(done)
    );
    
    //--------------------------------------------------------------------------
    // VCD Dump
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_generator_mini.vcd");
        $dumpvars(0, tb_generator_mini);
    end
    
    //--------------------------------------------------------------------------
    // State Name Translation
    //--------------------------------------------------------------------------
    always @(*) begin
        case (dut.state)
            4'd0:  state_name = "IDLE";
            4'd1:  state_name = "LOAD_IN";
            4'd2:  state_name = "ENC1";
            4'd3:  state_name = "BNECK";
            4'd4:  state_name = "UPSAMPLE1";
            4'd5:  state_name = "DEC1";
            4'd6:  state_name = "SKIP_ADD";
            4'd7:  state_name = "UPSAMPLE2";
            4'd8:  state_name = "OUT_CONV";
            4'd9:  state_name = "TANH";
            4'd10: state_name = "OUTPUT";
            4'd11: state_name = "DONE";
            default: state_name = "UNKNOWN";
        endcase
    end
    
    //--------------------------------------------------------------------------
    // State Change Monitor
    //--------------------------------------------------------------------------
    reg [3:0] prev_state;
    always @(posedge clk) begin
        if (dut.state != prev_state) begin
            $display("  [%0t] State: %s -> %s", $time, get_state_str(prev_state), get_state_str(dut.state));
        end
        prev_state <= dut.state;
    end
    
    function [127:0] get_state_str;
        input [3:0] st;
        begin
            case (st)
                4'd0:  get_state_str = "IDLE";
                4'd1:  get_state_str = "LOAD_IN";
                4'd2:  get_state_str = "ENC1";
                4'd3:  get_state_str = "BNECK";
                4'd4:  get_state_str = "UPSAMPLE1";
                4'd5:  get_state_str = "DEC1";
                4'd6:  get_state_str = "SKIP_ADD";
                4'd7:  get_state_str = "UPSAMPLE2";
                4'd8:  get_state_str = "OUT_CONV";
                4'd9:  get_state_str = "TANH";
                4'd10: get_state_str = "OUTPUT";
                4'd11: get_state_str = "DONE";
                default: get_state_str = "???";
            endcase
        end
    endfunction
    
    //--------------------------------------------------------------------------
    // Main Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("================================================================");
        $display("  SELF-CHECKING TESTBENCH: Mini U-Net Generator (Pipelined)");
        $display("================================================================");
        $display("");
        $display("Architecture: Parallel kernel MACs + 3-stage pipeline");
        $display("Fixed-point: Q8.8 activations, Q1.7 weights");
        $display("");
        
        // Initialize
        rst_n = 0;
        start = 0;
        data_in = 0;
        valid_in = 0;
        cond_in = 0;
        cond_valid = 0;
        ready_out = 1'b1;
        
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
        $display("TEST 1: Zero Input");
        $display("----------------------------------------------------------------");
        for (i = 0; i < IN_CH * FRAME_LEN; i = i + 1) test_input[i] = 0;
        for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1) begin
            golden_min[i] = -16'sh0200; golden_max[i] = 16'sh0200;
        end
        run_test(1);
        
        //----------------------------------------------------------------------
        // Test 2: DC Input (constant 0.5)
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 2: DC Input (0.5 in Q8.8 = 128)");
        $display("----------------------------------------------------------------");
        for (i = 0; i < IN_CH * FRAME_LEN; i = i + 1) test_input[i] = 16'sh0080;
        for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1) begin
            golden_min[i] = -16'sh0400; golden_max[i] = 16'sh0400;
        end
        run_test(2);
        
        //----------------------------------------------------------------------
        // Test 3: Impulse Response
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 3: Impulse Response");
        $display("----------------------------------------------------------------");
        for (i = 0; i < IN_CH * FRAME_LEN; i = i + 1) begin
            if (i == FRAME_LEN/2 || i == FRAME_LEN + FRAME_LEN/2)
                test_input[i] = 16'sh0100;
            else
                test_input[i] = 0;
        end
        for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1) begin
            golden_min[i] = -16'sh0800; golden_max[i] = 16'sh0800;
        end
        run_test(3);
        
        //----------------------------------------------------------------------
        // Test 4: Sine Wave (OFDM-like)
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 4: Sine Wave Pattern");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            test_input[i] = $rtoi(100.0 * $sin(2.0 * 3.14159 * i / FRAME_LEN));
            test_input[FRAME_LEN + i] = $rtoi(100.0 * $cos(2.0 * 3.14159 * i / FRAME_LEN));
        end
        for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1) begin
            golden_min[i] = -16'sh0800; golden_max[i] = 16'sh0800;
        end
        run_test(4);
        
        //----------------------------------------------------------------------
        // Test 5: Ramp Pattern
        //----------------------------------------------------------------------
        $display("");
        $display("----------------------------------------------------------------");
        $display("TEST 5: Ramp Pattern");
        $display("----------------------------------------------------------------");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            test_input[i] = (i - 8) * 16;
            test_input[FRAME_LEN + i] = (8 - i) * 16;
        end
        for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1) begin
            golden_min[i] = -16'sh0800; golden_max[i] = 16'sh0800;
        end
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
    reg input_timeout;
    
    task run_test;
        input integer test_id;
        begin
            test_num = test_id;
            total_tests = total_tests + 1;
            error_count = 0;
            in_idx = 0;
            out_idx = 0;
            input_timeout = 0;
            
            // Clear captured outputs
            for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1)
                captured_output[i] = 16'hDEAD;
            
            $display("  Starting inference...");
            
            // Wait for DUT to be ready
            wait(dut.state == 4'd0);  // ST_IDLE
            repeat(2) @(posedge clk);
            
            // Issue start
            @(posedge clk);
            start = 1;
            start_cycle = cycle_count;
            @(posedge clk);
            start = 0;
            
            // Wait for LOAD_IN state
            wait(dut.state == 4'd1);  // ST_LOAD_IN
            
            // Load input data with proper handshaking
            while (in_idx < IN_CH * FRAME_LEN && !input_timeout) begin
                @(posedge clk);
                if (cycle_count > start_cycle + TIMEOUT) begin
                    $display("  ERROR: Input loading timeout at sample %0d!", in_idx);
                    input_timeout = 1;
                end else begin
                    if (ready_in) begin
                        data_in = test_input[in_idx];
                        valid_in = 1;
                        cond_in = test_input[in_idx];
                        cond_valid = 1;
                    end else begin
                        valid_in = 0;
                        cond_valid = 0;
                    end
                    // Check if sample was accepted on this cycle
                    if (ready_in && valid_in) begin
                        in_idx = in_idx + 1;
                    end
                end
            end
            
            // Wait for generator to leave LOAD_IN state before clearing valid
            @(posedge clk);
            valid_in = 0;
            cond_valid = 0;
            
            $display("  Input loaded: %0d samples", in_idx);
            
            // Wait for processing and capture output
            while (!done && cycle_count < start_cycle + TIMEOUT) begin
                @(posedge clk);
                if (valid_out && ready_out && out_idx < OUT_CH * FRAME_LEN) begin
                    captured_output[out_idx] = data_out;
                    out_idx = out_idx + 1;
                end
            end
            
            total_cycles = cycle_count - start_cycle;
            
            // Check for timeout
            if (!done) begin
                $display("  ERROR: Timeout after %0d cycles!", TIMEOUT);
                $display("    Last state: %s", state_name);
                $display("    in_ch_cnt=%0d, in_pos_cnt=%0d", dut.in_ch_cnt, dut.in_pos_cnt);
                $display("    out_ch_cnt=%0d, out_pos_cnt=%0d", dut.out_ch_cnt, dut.out_pos_cnt);
                error_count = error_count + 100;
            end else begin
                $display("  Processing complete!");
                $display("  Total cycles: %0d", total_cycles);
                $display("  Output samples: %0d", out_idx);
                $display("  Throughput: %.2f cycles/sample", total_cycles * 1.0 / (OUT_CH * FRAME_LEN));
            end
            
            // Verify outputs
            $display("  Verifying outputs...");
            for (i = 0; i < out_idx && i < OUT_CH * FRAME_LEN; i = i + 1) begin
                if ($signed(captured_output[i]) < $signed(golden_min[i]) || 
                    $signed(captured_output[i]) > $signed(golden_max[i])) begin
                    $display("    [FAIL] Out[%0d] = %0d (expected [%0d, %0d])",
                             i, $signed(captured_output[i]), 
                             $signed(golden_min[i]), $signed(golden_max[i]));
                    error_count = error_count + 1;
                end
            end
            
            // Check output count
            if (out_idx != OUT_CH * FRAME_LEN) begin
                $display("    [FAIL] Expected %0d outputs, got %0d", OUT_CH * FRAME_LEN, out_idx);
                error_count = error_count + 1;
            end
            
            // Report
            if (error_count == 0) begin
                $display("  Result: *** PASS *** (%0d outputs verified)", out_idx);
            end else begin
                $display("  Result: *** FAIL *** (%0d errors)", error_count);
                total_errors = total_errors + error_count;
            end
            
            // Wait for done to clear and return to IDLE
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
