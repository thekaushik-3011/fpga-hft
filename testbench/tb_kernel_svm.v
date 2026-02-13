`timescale 1ns / 1ps

module tb_kernel_svm;

    // Parameters
    parameter DATA_WIDTH = 16;
    parameter NUM_FEATURES = 16;
    parameter NUM_SUPPORT_VECTORS = 16;
    
    // Signals
    reg clk;
    reg rst_n;
    reg input_valid;
    reg signed [DATA_WIDTH*NUM_FEATURES-1:0] features_flat;
    reg signed [DATA_WIDTH*NUM_FEATURES*NUM_SUPPORT_VECTORS-1:0] sv_flat;
    reg signed [DATA_WIDTH*NUM_SUPPORT_VECTORS-1:0] dual_coef_flat;
    reg signed [DATA_WIDTH-1:0] bias;
    
    wire output_valid;
    wire signed [DATA_WIDTH-1:0] decision_value;
    wire prediction;
    
    // Instantiate DUT
    kernel_svm #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_FEATURES(NUM_FEATURES),
        .NUM_SUPPORT_VECTORS(NUM_SUPPORT_VECTORS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(input_valid),
        .features_flat(features_flat),
        .sv_flat(sv_flat),
        .dual_coef_flat(dual_coef_flat),
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
    integer i, j;
    initial begin
        // Setup VCD dumping
        $dumpfile("svm_kernel.vcd");
        $dumpvars(0, tb_kernel_svm);
        
        // Initialize
        rst_n = 0;
        input_valid = 0;
        features_flat = 0;
        sv_flat = 0;
        dual_coef_flat = 0;
        bias = 0;
        
        // Reset
        #100;
        rst_n = 1;
        #20;
        
        // Setup Logic
        // SVs: Let's make SV[0] identical to input, others far away
        // SV[0] = All 1.0 (0x0100)
        // SV[1..15] = All 0.0
        // Dual Coefs: Coef[0] = 1.0, others 0
        // Expected:
        // DistSq(x, SV[0]) = 0 -> exp(0) = 1.0 (256)
        // Weighted = 1.0 * 1.0 = 1.0
        // Result = 1.0 + bias
        
        // Initialize SVs
        // Flattened array: index = (j*NUM_FEATURES + i)
        for (j=0; j<NUM_SUPPORT_VECTORS; j=j+1) begin
             // Dual coefs
             if (j == 0) dual_coef_flat[(j+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0100; // 1.0
             else dual_coef_flat[(j+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0000;
             
             for (i=0; i<NUM_FEATURES; i=i+1) begin
                 if (j == 0) begin
                     // SV[0] = 1.0
                     sv_flat[((j*NUM_FEATURES)+i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0100;
                 end else begin
                     // Others = 0
                     sv_flat[((j*NUM_FEATURES)+i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0000;
                 end
             end
        end
        bias = 0;
        
        // Test Case 1: Input matches SV[0]
        // X = All 1.0
        @(posedge clk);
        #1;
        for (i=0; i<NUM_FEATURES; i=i+1) begin
             features_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0100;
        end
        input_valid = 1;
        
        @(posedge clk);
        #1;
        input_valid = 0;
        
        repeat(20) @(posedge clk);
        
        // Test Case 2: Input far from SV[0]
        // X = All 0.0
        // DistSq(x, SV[0]) = 16 * (1.0 - 0.0)^2 = 16.0
        // Exp(-gamma * 16.0) -> Should be small
        // SV[1] is 0.0, so DistSq(x, SV[1]) = 0.
        // But DualCoef[1] is 0. So contribution is 0.
        // Result should be near 0 (from SV[0] contribution being small)
        @(posedge clk);
        #1;
        for (i=0; i<NUM_FEATURES; i=i+1) begin
             features_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = 16'h0000;
        end
        input_valid = 1;
        
        @(posedge clk);
        #1;
        input_valid = 0;
         
        repeat(20) @(posedge clk);
        
        $finish;
    end
    
    // Monitor
    always @(posedge clk) begin
        if (output_valid) begin
            $display("Time=%t | Pred=%b | Decision=%d", $time, prediction, decision_value);
        end
    end

endmodule
