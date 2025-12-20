//==============================================================================
// CWGAN-GP Full Verification Testbench
//
// Demonstrates OFDM Signal Reconstruction Improvement
//
// This testbench:
//   1. Generates realistic OFDM signals (QAM constellation + channel effects)
//   2. Applies degradation (AWGN noise, fading, interference)
//   3. Runs the Generator to reconstruct the signal
//   4. Computes quantitative metrics:
//      - MSE (Mean Squared Error)
//      - SNR improvement
//      - EVM (Error Vector Magnitude)
//   5. Displays before/after comparison
//
// Expected Result: Reconstructed signal should have LOWER MSE than degraded
//==============================================================================

`timescale 1ns / 1ps

module tb_cwgan_gp_full;

    //--------------------------------------------------------------------------
    // Parameters
    //--------------------------------------------------------------------------
    parameter DATA_WIDTH   = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter ACC_WIDTH    = 32;
    parameter FRAME_LEN    = 16;
    parameter IN_CH        = 2;
    
    parameter CLK_PERIOD   = 10;  // 100 MHz
    
    // Test configuration
    parameter NUM_TESTS    = 5;
    
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
    // Test Data Storage
    //--------------------------------------------------------------------------
    // Original clean OFDM signal
    reg signed [DATA_WIDTH-1:0] clean_i [0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] clean_q [0:FRAME_LEN-1];
    
    // Degraded signal (with noise/fading)
    reg signed [DATA_WIDTH-1:0] degraded_i [0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] degraded_q [0:FRAME_LEN-1];
    
    // Reconstructed signal from Generator
    reg signed [DATA_WIDTH-1:0] recon_i [0:FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] recon_q [0:FRAME_LEN-1];
    
    //--------------------------------------------------------------------------
    // Metrics
    //--------------------------------------------------------------------------
    real mse_degraded, mse_reconstructed;
    real snr_degraded, snr_reconstructed;
    real evm_degraded, evm_reconstructed;
    real improvement_ratio;
    real signal_power;
    
    // Accumulated metrics across tests
    real total_mse_degraded, total_mse_recon;
    real total_improvement;
    integer tests_improved;
    
    //--------------------------------------------------------------------------
    // Control Variables
    //--------------------------------------------------------------------------
    integer i, j, test_num;
    integer output_count;
    integer seed;
    
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
        $dumpfile("tb_cwgan_gp_full.vcd");
        $dumpvars(0, tb_cwgan_gp_full);
    end
    
    //--------------------------------------------------------------------------
    // Signal Generation Functions
    //--------------------------------------------------------------------------
    
    // Generate 16-QAM OFDM signal
    task generate_qam16_ofdm;
        input integer noise_level;  // 0-100 (percentage)
        input integer fade_depth;   // 0-100 (percentage of amplitude)
        input integer fade_pos;     // Position of fade (0-15)
        integer idx;
        integer qam_level;
        integer noise_i, noise_q;
        real fade_factor;
        begin
            for (idx = 0; idx < FRAME_LEN; idx = idx + 1) begin
                // Generate 16-QAM symbol: {-3, -1, +1, +3} * scale
                // Q8.8 format, scale by 32 for reasonable amplitude
                qam_level = (($random(seed) % 4) * 2 - 3) * 32;
                clean_i[idx] = qam_level;
                
                qam_level = (($random(seed) % 4) * 2 - 3) * 32;
                clean_q[idx] = qam_level;
                
                // Apply fading (simulate multipath/deep fade)
                fade_factor = 1.0;
                if (idx >= fade_pos && idx < fade_pos + 4) begin
                    fade_factor = (100.0 - fade_depth) / 100.0;
                end
                
                // Add AWGN noise
                noise_i = (($random(seed) % 64) - 32) * noise_level / 100;
                noise_q = (($random(seed) % 64) - 32) * noise_level / 100;
                
                // Degraded = faded_clean + noise
                degraded_i[idx] = $rtoi(clean_i[idx] * fade_factor) + noise_i;
                degraded_q[idx] = $rtoi(clean_q[idx] * fade_factor) + noise_q;
            end
        end
    endtask
    
    // Generate OFDM with burst interference
    task generate_ofdm_with_burst;
        input integer burst_start;
        input integer burst_len;
        input integer burst_power;
        integer idx;
        integer qam_level;
        begin
            for (idx = 0; idx < FRAME_LEN; idx = idx + 1) begin
                // Clean QAM signal
                qam_level = (($random(seed) % 4) * 2 - 3) * 32;
                clean_i[idx] = qam_level;
                qam_level = (($random(seed) % 4) * 2 - 3) * 32;
                clean_q[idx] = qam_level;
                
                // Copy to degraded
                degraded_i[idx] = clean_i[idx];
                degraded_q[idx] = clean_q[idx];
                
                // Add burst interference
                if (idx >= burst_start && idx < burst_start + burst_len) begin
                    degraded_i[idx] = degraded_i[idx] + burst_power;
                    degraded_q[idx] = degraded_q[idx] - burst_power;
                end
                
                // Add light background noise
                degraded_i[idx] = degraded_i[idx] + ($random(seed) % 16) - 8;
                degraded_q[idx] = degraded_q[idx] + ($random(seed) % 16) - 8;
            end
        end
    endtask
    
    // Generate frequency-selective fading
    task generate_freq_selective_fade;
        input integer fade_strength;  // 0-100
        integer idx;
        integer qam_level;
        real fade_coef;
        begin
            for (idx = 0; idx < FRAME_LEN; idx = idx + 1) begin
                // Clean QAM signal
                qam_level = (($random(seed) % 4) * 2 - 3) * 32;
                clean_i[idx] = qam_level;
                qam_level = (($random(seed) % 4) * 2 - 3) * 32;
                clean_q[idx] = qam_level;
                
                // Frequency-selective fade: varies across subcarriers
                // Simulates multipath with sinusoidal pattern
                fade_coef = 1.0 - (fade_strength/100.0) * 0.5 * (1.0 + $sin(3.14159 * idx / 4.0));
                
                degraded_i[idx] = $rtoi(clean_i[idx] * fade_coef) + ($random(seed) % 20) - 10;
                degraded_q[idx] = $rtoi(clean_q[idx] * fade_coef) + ($random(seed) % 20) - 10;
            end
        end
    endtask
    
    //--------------------------------------------------------------------------
    // Metric Calculation Functions
    //--------------------------------------------------------------------------
    
    // Calculate MSE between two signals
    task calculate_mse;
        output real mse;
        input integer use_recon;  // 0: compare degraded vs clean, 1: compare recon vs clean
        integer idx;
        real diff_i, diff_q;
        real sum_sq;
        begin
            sum_sq = 0.0;
            for (idx = 0; idx < FRAME_LEN; idx = idx + 1) begin
                if (use_recon) begin
                    diff_i = recon_i[idx] - clean_i[idx];
                    diff_q = recon_q[idx] - clean_q[idx];
                end else begin
                    diff_i = degraded_i[idx] - clean_i[idx];
                    diff_q = degraded_q[idx] - clean_q[idx];
                end
                sum_sq = sum_sq + diff_i * diff_i + diff_q * diff_q;
            end
            mse = sum_sq / (2.0 * FRAME_LEN);
        end
    endtask
    
    // Calculate signal power
    task calculate_signal_power;
        output real power;
        integer idx;
        real sum_sq;
        begin
            sum_sq = 0.0;
            for (idx = 0; idx < FRAME_LEN; idx = idx + 1) begin
                sum_sq = sum_sq + clean_i[idx] * clean_i[idx] + clean_q[idx] * clean_q[idx];
            end
            power = sum_sq / (2.0 * FRAME_LEN);
        end
    endtask
    
    // Calculate SNR in dB
    task calculate_snr;
        output real snr;
        input real sig_power;
        input real noise_mse;
        begin
            if (noise_mse > 0.0001)
                snr = 10.0 * $log10(sig_power / noise_mse);
            else
                snr = 99.9;  // Very high SNR
        end
    endtask
    
    // Calculate EVM (Error Vector Magnitude) in percentage
    task calculate_evm;
        output real evm;
        input real mse;
        input real sig_power;
        begin
            if (sig_power > 0.0001)
                evm = 100.0 * $sqrt(mse / sig_power);
            else
                evm = 0.0;
        end
    endtask
    
    //--------------------------------------------------------------------------
    // DUT Interface Tasks
    //--------------------------------------------------------------------------
    
    task reset_dut;
        begin
            rst_n = 0;
            start = 0;
            mode = 0;
            ofdm_degraded_in = 0;
            ofdm_degraded_valid = 0;
            ofdm_clean_in = 0;
            ofdm_clean_valid = 0;
            ofdm_recon_ready = 0;
            repeat(10) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);
        end
    endtask
    
    task run_inference;
        integer ch, pos;
        begin
            // Start inference mode
            mode = 0;
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            // Feed degraded signal
            for (ch = 0; ch < IN_CH; ch = ch + 1) begin
                for (pos = 0; pos < FRAME_LEN; pos = pos + 1) begin
                    while (!ofdm_degraded_ready) @(posedge clk);
                    ofdm_degraded_valid = 1;
                    if (ch == 0)
                        ofdm_degraded_in = degraded_i[pos];
                    else
                        ofdm_degraded_in = degraded_q[pos];
                    @(posedge clk);
                end
            end
            ofdm_degraded_valid = 0;
            
            // Capture reconstructed output
            ofdm_recon_ready = 1;
            output_count = 0;
            
            while (!done) begin
                @(posedge clk);
                if (ofdm_recon_valid) begin
                    if (output_count < FRAME_LEN)
                        recon_i[output_count] = ofdm_recon_out;
                    else if (output_count < 2*FRAME_LEN)
                        recon_q[output_count - FRAME_LEN] = ofdm_recon_out;
                    output_count = output_count + 1;
                end
            end
            ofdm_recon_ready = 0;
        end
    endtask
    
    //--------------------------------------------------------------------------
    // Display Functions
    //--------------------------------------------------------------------------
    
    task print_signal_table;
        input [255:0] test_name;
        integer idx;
        begin
            $display("\n  Sample Comparison for %s:", test_name);
            $display("  -------------------------------------------------------------------------");
            $display("  Idx |  Clean_I  Clean_Q | Degraded_I Degraded_Q | Recon_I  Recon_Q");
            $display("  -------------------------------------------------------------------------");
            for (idx = 0; idx < FRAME_LEN; idx = idx + 1) begin
                $display("  %3d | %7d  %7d | %10d  %10d | %7d  %7d",
                    idx, clean_i[idx], clean_q[idx],
                    degraded_i[idx], degraded_q[idx],
                    recon_i[idx], recon_q[idx]);
            end
            $display("  -------------------------------------------------------------------------");
        end
    endtask
    
    task print_metrics;
        begin
            $display("\n  === SIGNAL QUALITY METRICS ===");
            $display("  +---------------------------+-------------+-------------+");
            $display("  | Metric                    |   Degraded  | Reconstructed |");
            $display("  +---------------------------+-------------+-------------+");
            $display("  | MSE (lower is better)     | %11.2f | %11.2f |", mse_degraded, mse_reconstructed);
            $display("  | SNR [dB] (higher better)  | %11.2f | %11.2f |", snr_degraded, snr_reconstructed);
            $display("  | EVM %% (lower is better)   | %11.2f | %11.2f |", evm_degraded, evm_reconstructed);
            $display("  +---------------------------+-------------+-------------+");
            
            if (mse_reconstructed < mse_degraded) begin
                improvement_ratio = (mse_degraded - mse_reconstructed) / mse_degraded * 100.0;
                $display("  | IMPROVEMENT               |    %.1f%% MSE reduction    |", improvement_ratio);
                $display("  +---------------------------+---------------------------+");
                $display("  >>> GENERATOR IMPROVED THE SIGNAL! <<<");
            end else begin
                $display("  | RESULT                    | No improvement detected   |");
                $display("  +---------------------------+---------------------------+");
            end
        end
    endtask
    
    //--------------------------------------------------------------------------
    // Main Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        seed = 12345;
        total_mse_degraded = 0.0;
        total_mse_recon = 0.0;
        total_improvement = 0.0;
        tests_improved = 0;
        
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║     CWGAN-GP OFDM Signal Reconstruction Verification            ║");
        $display("║                                                                  ║");
        $display("║  Testing if the GAN Generator improves degraded OFDM signals    ║");
        $display("║  by comparing MSE, SNR, and EVM before and after processing     ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        $display("Configuration:");
        $display("  - Frame Length: %d samples", FRAME_LEN);
        $display("  - Channels: %d (I and Q)", IN_CH);
        $display("  - Data Format: Q8.8 fixed-point");
        $display("  - Weight Format: Q1.7 fixed-point");
        $display("");
        
        //----------------------------------------------------------------------
        // Test 1: Moderate AWGN Noise
        //----------------------------------------------------------------------
        test_num = 1;
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST %0d: Moderate AWGN Noise (30%% noise level)", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        generate_qam16_ofdm(30, 0, 0);  // 30% noise, no fading
        
        run_inference();
        
        // Calculate metrics
        calculate_signal_power(signal_power);
        calculate_mse(mse_degraded, 0);
        calculate_mse(mse_reconstructed, 1);
        calculate_snr(snr_degraded, signal_power, mse_degraded);
        calculate_snr(snr_reconstructed, signal_power, mse_reconstructed);
        calculate_evm(evm_degraded, mse_degraded, signal_power);
        calculate_evm(evm_reconstructed, mse_reconstructed, signal_power);
        
        print_signal_table("AWGN Test");
        print_metrics();
        
        total_mse_degraded = total_mse_degraded + mse_degraded;
        total_mse_recon = total_mse_recon + mse_reconstructed;
        if (mse_reconstructed < mse_degraded) tests_improved = tests_improved + 1;
        
        //----------------------------------------------------------------------
        // Test 2: Deep Fade + Noise
        //----------------------------------------------------------------------
        test_num = 2;
        $display("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST %0d: Deep Fade (60%% at position 4-7) + Light Noise", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        generate_qam16_ofdm(15, 60, 4);  // 15% noise, 60% fade at pos 4
        
        run_inference();
        
        calculate_signal_power(signal_power);
        calculate_mse(mse_degraded, 0);
        calculate_mse(mse_reconstructed, 1);
        calculate_snr(snr_degraded, signal_power, mse_degraded);
        calculate_snr(snr_reconstructed, signal_power, mse_reconstructed);
        calculate_evm(evm_degraded, mse_degraded, signal_power);
        calculate_evm(evm_reconstructed, mse_reconstructed, signal_power);
        
        print_signal_table("Deep Fade Test");
        print_metrics();
        
        total_mse_degraded = total_mse_degraded + mse_degraded;
        total_mse_recon = total_mse_recon + mse_reconstructed;
        if (mse_reconstructed < mse_degraded) tests_improved = tests_improved + 1;
        
        //----------------------------------------------------------------------
        // Test 3: Burst Interference
        //----------------------------------------------------------------------
        test_num = 3;
        $display("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST %0d: Burst Interference (positions 8-11, power=64)", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        generate_ofdm_with_burst(8, 4, 64);  // Burst at pos 8-11
        
        run_inference();
        
        calculate_signal_power(signal_power);
        calculate_mse(mse_degraded, 0);
        calculate_mse(mse_reconstructed, 1);
        calculate_snr(snr_degraded, signal_power, mse_degraded);
        calculate_snr(snr_reconstructed, signal_power, mse_reconstructed);
        calculate_evm(evm_degraded, mse_degraded, signal_power);
        calculate_evm(evm_reconstructed, mse_reconstructed, signal_power);
        
        print_signal_table("Burst Interference Test");
        print_metrics();
        
        total_mse_degraded = total_mse_degraded + mse_degraded;
        total_mse_recon = total_mse_recon + mse_reconstructed;
        if (mse_reconstructed < mse_degraded) tests_improved = tests_improved + 1;
        
        //----------------------------------------------------------------------
        // Test 4: Frequency-Selective Fading
        //----------------------------------------------------------------------
        test_num = 4;
        $display("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST %0d: Frequency-Selective Fading (50%% strength)", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        generate_freq_selective_fade(50);
        
        run_inference();
        
        calculate_signal_power(signal_power);
        calculate_mse(mse_degraded, 0);
        calculate_mse(mse_reconstructed, 1);
        calculate_snr(snr_degraded, signal_power, mse_degraded);
        calculate_snr(snr_reconstructed, signal_power, mse_reconstructed);
        calculate_evm(evm_degraded, mse_degraded, signal_power);
        calculate_evm(evm_reconstructed, mse_reconstructed, signal_power);
        
        print_signal_table("Freq-Selective Fade Test");
        print_metrics();
        
        total_mse_degraded = total_mse_degraded + mse_degraded;
        total_mse_recon = total_mse_recon + mse_reconstructed;
        if (mse_reconstructed < mse_degraded) tests_improved = tests_improved + 1;
        
        //----------------------------------------------------------------------
        // Test 5: Heavy Noise + Severe Fade (Worst Case)
        //----------------------------------------------------------------------
        test_num = 5;
        $display("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST %0d: Worst Case - Heavy Noise (50%%) + Severe Fade (80%%)", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        generate_qam16_ofdm(50, 80, 6);  // 50% noise, 80% fade
        
        run_inference();
        
        calculate_signal_power(signal_power);
        calculate_mse(mse_degraded, 0);
        calculate_mse(mse_reconstructed, 1);
        calculate_snr(snr_degraded, signal_power, mse_degraded);
        calculate_snr(snr_reconstructed, signal_power, mse_reconstructed);
        calculate_evm(evm_degraded, mse_degraded, signal_power);
        calculate_evm(evm_reconstructed, mse_reconstructed, signal_power);
        
        print_signal_table("Worst Case Test");
        print_metrics();
        
        total_mse_degraded = total_mse_degraded + mse_degraded;
        total_mse_recon = total_mse_recon + mse_reconstructed;
        if (mse_reconstructed < mse_degraded) tests_improved = tests_improved + 1;
        
        //----------------------------------------------------------------------
        // Final Summary
        //----------------------------------------------------------------------
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                    FINAL TEST SUMMARY                            ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║                                                                  ║");
        $display("║  Total Tests:              %2d                                    ║", NUM_TESTS);
        $display("║  Tests with Improvement:   %2d                                    ║", tests_improved);
        $display("║                                                                  ║");
        $display("║  Average MSE (Degraded):     %10.2f                          ║", total_mse_degraded / NUM_TESTS);
        $display("║  Average MSE (Reconstructed): %10.2f                          ║", total_mse_recon / NUM_TESTS);
        $display("║                                                                  ║");
        
        if (total_mse_recon < total_mse_degraded) begin
            improvement_ratio = (total_mse_degraded - total_mse_recon) / total_mse_degraded * 100.0;
            $display("║  ★ OVERALL IMPROVEMENT: %.1f%% average MSE reduction            ║", improvement_ratio);
            $display("║                                                                  ║");
            $display("║  ✓ THE GAN GENERATOR SUCCESSFULLY RECONSTRUCTS OFDM SIGNALS!    ║");
        end else begin
            $display("║  ✗ Generator did not show overall improvement                   ║");
            $display("║    (Note: Untrained weights - improvement expected after        ║");
            $display("║     training with real OFDM data)                               ║");
        end
        
        $display("║                                                                  ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Note about untrained weights
        $display("NOTE: This testbench uses placeholder/random weights from weight_rom.v");
        $display("      After training with real OFDM data and exporting weights via:");
        $display("        python utils/export_rtl_weights.py --checkpoint <trained.pt>");
        $display("      The reconstruction quality will significantly improve.");
        $display("");
        
        #1000;
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // Timeout Watchdog
    //--------------------------------------------------------------------------
    initial begin
        #50000000;  // 50ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
