//==============================================================================
// Generator Mini Testbench
//
// Focused testbench for the Mini U-Net Generator
// Tests the generator in isolation with known stimulus
//
// Verifies:
//   - Data loading
//   - Convolution operations
//   - Skip connections
//   - Activation functions
//   - Output generation
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
    
    parameter CLK_PERIOD   = 5;
    
    //--------------------------------------------------------------------------
    // DUT Signals
    //--------------------------------------------------------------------------
    reg                         clk;
    reg                         rst_n;
    reg                         start;
    
    reg  signed [DATA_WIDTH-1:0] data_in;
    reg                          valid_in;
    wire                         ready_in;
    
    reg  signed [DATA_WIDTH-1:0] cond_in;
    reg                          cond_valid;
    
    wire signed [DATA_WIDTH-1:0] data_out;
    wire                         data_out_valid;
    reg                          data_out_ready;
    
    wire                         busy;
    wire                         done;
    
    //--------------------------------------------------------------------------
    // Test Data
    //--------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_data [0:IN_CH*FRAME_LEN-1];
    reg signed [DATA_WIDTH-1:0] output_data [0:OUT_CH*FRAME_LEN-1];
    
    integer i;
    integer cycles_count;
    integer output_count;
    
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
        .valid_out(data_out_valid),
        .ready_out(data_out_ready),
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
    // Initialize Test Data
    //--------------------------------------------------------------------------
    initial begin
        // Generate sine-like test pattern
        // I channel
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            input_data[i] = $rtoi(100.0 * $sin(2.0 * 3.14159 * i / FRAME_LEN));
        end
        // Q channel
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            input_data[FRAME_LEN + i] = $rtoi(100.0 * $cos(2.0 * 3.14159 * i / FRAME_LEN));
        end
        
        // Initialize outputs
        for (i = 0; i < OUT_CH * FRAME_LEN; i = i + 1) begin
            output_data[i] = 0;
        end
    end
    
    //--------------------------------------------------------------------------
    // Cycle Counter
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycles_count <= 0;
        else if (busy)
            cycles_count <= cycles_count + 1;
    end
    
    //--------------------------------------------------------------------------
    // Output Capture
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (data_out_valid && data_out_ready) begin
            output_data[output_count] <= data_out;
            output_count <= output_count + 1;
            $display("[%0t] Output[%0d] = %d (0x%04x)", $time, output_count, data_out, data_out);
        end
    end
    
    //--------------------------------------------------------------------------
    // Main Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("  Generator Mini Testbench");
        $display("========================================\n");
        
        // Initialize
        rst_n <= 1'b0;
        start <= 1'b0;
        data_in <= 0;
        valid_in <= 1'b0;
        cond_in <= 0;
        cond_valid <= 1'b0;
        data_out_ready <= 1'b0;
        output_count <= 0;
        
        // Reset
        repeat(10) @(posedge clk);
        rst_n <= 1'b1;
        repeat(5) @(posedge clk);
        
        $display("Starting generator...");
        
        // Start generator
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        
        // Feed input data
        $display("Feeding input data...");
        for (i = 0; i < IN_CH * FRAME_LEN; i = i + 1) begin
            @(posedge clk);
            wait(ready_in);
            data_in <= input_data[i];
            valid_in <= 1'b1;
            cond_in <= input_data[i];
            cond_valid <= 1'b1;
            @(posedge clk);
        end
        valid_in <= 1'b0;
        cond_valid <= 1'b0;
        
        $display("Input feeding complete. Waiting for processing...");
        
        // Enable output capture
        data_out_ready <= 1'b1;
        
        // Wait for completion
        wait(done);
        
        $display("\n========================================");
        $display("  Results");
        $display("========================================");
        $display("Processing cycles: %0d", cycles_count);
        $display("Output samples captured: %0d", output_count);
        
        // Print input/output comparison
        $display("\n--- I Channel ---");
        $display("Idx | Input  | Output");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            $display("%3d | %6d | %6d", i, input_data[i], output_data[i]);
        end
        
        $display("\n--- Q Channel ---");
        $display("Idx | Input  | Output");
        for (i = 0; i < FRAME_LEN; i = i + 1) begin
            $display("%3d | %6d | %6d", i, input_data[FRAME_LEN+i], output_data[FRAME_LEN+i]);
        end
        
        // Performance summary
        $display("\n========================================");
        $display("  Performance Summary");
        $display("========================================");
        $display("Frame Length: %0d samples", FRAME_LEN);
        $display("Channels: %0d", IN_CH);
        $display("Total Cycles: %0d", cycles_count);
        $display("Cycles per Sample: %.2f", cycles_count / (1.0 * FRAME_LEN));
        $display("Throughput @ 200MHz: %.2f Msps", 200.0 * FRAME_LEN / cycles_count);
        
        #100;
        $display("\nSimulation complete.");
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // Timeout
    //--------------------------------------------------------------------------
    initial begin
        #500000;
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule
