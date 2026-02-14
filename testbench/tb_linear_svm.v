
`timescale 1ns / 1ps

module tb_linear_svm;

    // Parameters
    parameter DATA_WIDTH = 16;
    parameter FRAC_BITS = 8;
    parameter NUM_FEATURES = 20;

    // Inputs
    reg clk;
    reg rst_n;
    reg input_valid;
    reg signed [DATA_WIDTH*NUM_FEATURES-1:0] features_flat;
    reg signed [DATA_WIDTH*NUM_FEATURES-1:0] weights_flat;
    reg signed [DATA_WIDTH-1:0] bias;

    // Outputs
    wire output_valid;
    wire signed [DATA_WIDTH-1:0] decision_value;
    wire prediction;

    // Include Test Vectors
    `include "svm_test_vectors.vh"

    // Instantiate the Unit Under Test (UUT)
    linear_svm #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .NUM_FEATURES(NUM_FEATURES)
    ) uut (
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

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Test Memory
    // Width: 20 features * 16 bits + 16 bits (decision) + 16 bits (pred) = 352 bits
    // But $readmemh works best with fixed width words or we can use multiple arrays.
    // Let's use a flat memory and unpack.
    // Each line in .mem is: F0 F1 ... F19 Dec Pred
    // Total 22 words of 16-bits.
    
    reg signed [15:0] test_mem [0:NUM_TEST_CASES*22-1];
    
    integer i, j;
    integer pass_count = 0;
    integer fail_count = 0;
    
    reg signed [15:0] expected_decision;
    reg expected_prediction;

    initial begin
        // Initialize Inputs
        rst_n = 0;
        input_valid = 0;
        features_flat = 0;
        weights_flat = 0;
        bias = 0;

        // Load Memory
        $readmemh("testbench/test_data.mem", test_mem);
        
        // Wait for global reset
        #100;
        
        // Load Weights once
        bias = TEST_BIAS;
        for (i = 0; i < NUM_FEATURES; i = i + 1) begin
            weights_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = TEST_WEIGHTS[i];
        end

        rst_n = 1;
        #20;
        
        $display("Starting Verification of %d Test Cases...", NUM_TEST_CASES);
        
        for (j = 0; j < NUM_TEST_CASES; j = j + 1) begin
            // 1. Unpack Features for Case j
            // Base index in flat memory = j * 22
            for (i = 0; i < NUM_FEATURES; i = i + 1) begin
                 features_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = test_mem[j*22 + i];
            end
            
            // 2. Unpack Expected Results
            expected_decision = test_mem[j*22 + 20];
            expected_prediction = test_mem[j*22 + 21];
            
            // 3. Apply Stimulus
            @(posedge clk);
            input_valid = 1;
            @(posedge clk);
            input_valid = 0;
            
            // 4. Wait for output
            wait(output_valid);
            @(posedge clk); // Capture
            
            // 5. Check
            if (decision_value !== expected_decision || prediction !== expected_prediction) begin
                $display("ERROR Case %d: RTL(Dec=%d, Pred=%d) vs Exp(Dec=%d, Pred=%d)", 
                         j, decision_value, prediction, expected_decision, expected_prediction);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
            
            // Wait for not valid to ensure clean restart? 
            // Valid pulse is 1 cycle? My RTL asserts valid for 1 cycle.
            // Wait a cycle to separate tests visually if needed
            #10;
        end
        
        $display("----------------------------------------");
        $display("Tests Completed: %d", NUM_TEST_CASES);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else $display("SOME TESTS FAILED");
        $display("----------------------------------------");
        
        $finish;
    end
      
endmodule
