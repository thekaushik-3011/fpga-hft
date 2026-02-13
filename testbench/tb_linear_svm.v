`timescale 1ns / 1ps

module tb_linear_svm;

    // Parameters
    parameter DATA_WIDTH = 16;
    parameter NUM_FEATURES = 16;
    
    // Signals
    reg clk;
    reg rst_n;
    reg input_valid;
    reg signed [DATA_WIDTH*NUM_FEATURES-1:0] features_flat;
    reg signed [DATA_WIDTH*NUM_FEATURES-1:0] weights_flat;
    reg signed [DATA_WIDTH-1:0] bias;
    
    wire output_valid;
    wire signed [DATA_WIDTH-1:0] decision_value;
    wire prediction;
    
    // Instantiate DUT
    linear_svm #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_FEATURES(NUM_FEATURES)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(input_valid),
        .features_flat(features_flat),
        .weights_flat(weights_flat),
        .bias(bias),
        .output_valid(output_valid),
        .decision_value(decision_value),
        .prediction(prediction)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequences
    integer i;
    initial begin
        // Setup VCD dumping
        $dumpfile("svm.vcd");
        $dumpvars(0, tb_linear_svm);
        
        // Initialize
        rst_n = 0;
        input_valid = 0;
        features_flat = 0;
        weights_flat = 0;
        bias = 0;
        
        // Reset
        #100;
        rst_n = 1;
        #20;
        
        // Set weights (Example: all 1s in Q8.8 -> 0x0100)
        // Use non-blocking or delay to avoid race
        @(posedge clk);
        #1;
        for (i=0; i<NUM_FEATURES; i=i+1) begin
            weights_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0100; // 1.0
        end
        bias = 16'h0000; // 0.0
        
        // Test Case 1: All inputs 1.0
        // Dot product = 16 * (1.0 * 1.0) = 16.0
        // 16.0 in Q8.8 is 16 * 256 = 4096 = 0x1000
        @(posedge clk);
        #1;
        for (i=0; i<NUM_FEATURES; i=i+1) begin
            features_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0100; // 1.0
        end
        input_valid = 1;
        
        @(posedge clk);
        #1;
        input_valid = 0;
        
        // Wait for output
        repeat(10) @(posedge clk);
        
        // Test Case 2: Mixed inputs
        @(posedge clk);
        #1;
        for (i=0; i<NUM_FEATURES; i=i+1) begin
            features_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = (i % 2 == 0) ? 16'h0080 : -16'h0080; // 0.5 or -0.5 (Q8.8 0.5 = 128 = 0x80)
        end
        input_valid = 1;
        
        @(posedge clk);
        #1;
        input_valid = 0;
        
        repeat(10) @(posedge clk);
        
        $finish;
    end
    
    // Monitor
    always @(posedge clk) begin
        if (output_valid) begin
            $display("Time=%t | Pred=%b | Decision=%d", $time, prediction, decision_value);
        end
    end

endmodule
