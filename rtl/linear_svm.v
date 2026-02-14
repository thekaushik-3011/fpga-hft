`timescale 1ns / 1ps

module linear_svm #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS = 8,
    parameter NUM_FEATURES = 20
)(
    input wire clk,
    input wire rst_n,
    input wire input_valid,
    input wire signed [DATA_WIDTH*NUM_FEATURES-1:0] features_flat,
    input wire signed [DATA_WIDTH*NUM_FEATURES-1:0] weights_flat,
    input wire signed [DATA_WIDTH-1:0] bias,
    
    output reg output_valid,
    output reg signed [DATA_WIDTH-1:0] decision_value,
    output reg prediction
);

    // Unpack arrays
    wire signed [DATA_WIDTH-1:0] features [0:NUM_FEATURES-1];
    wire signed [DATA_WIDTH-1:0] weights [0:NUM_FEATURES-1];
    
    genvar i;
    generate
        for (i = 0; i < NUM_FEATURES; i = i + 1) begin
            assign features[i] = features_flat[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
            assign weights[i] = weights_flat[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
        end
    endgenerate

    // Pipeline Stage 1: Multiplication
    // Result of Q8.8 * Q8.8 is Q16.16 (32 bits)
    reg signed [2*DATA_WIDTH-1:0] products [0:NUM_FEATURES-1];
    reg stage1_valid;
    
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 0;
            for (j = 0; j < NUM_FEATURES; j = j + 1) begin
                products[j] <= 0;
            end
        end else begin
            stage1_valid <= input_valid;
            if (input_valid) begin
                for (j = 0; j < NUM_FEATURES; j = j + 1) begin
                    products[j] <= features[j] * weights[j];
                end
            end
        end
    end

    // Pipeline Stage 2: Accumulation Level 1 (20 -> 10)
    reg signed [2*DATA_WIDTH+3:0] level1_sum [0:9]; 
    reg stage2_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_valid <= 0;
            for (j = 0; j < 10; j = j + 1) level1_sum[j] <= 0;
        end else begin
            stage2_valid <= stage1_valid;
            if (stage1_valid) begin
                for (j = 0; j < 10; j = j + 1) begin
                    level1_sum[j] <= products[2*j] + products[2*j+1];
                end
            end
        end
    end
    
    // Pipeline Stage 3: Accumulation Level 2 (10 -> 5)
    reg signed [2*DATA_WIDTH+4:0] level2_sum [0:4];
    reg stage3_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage3_valid <= 0;
            for (j = 0; j < 5; j = j + 1) level2_sum[j] <= 0;
        end else begin
            stage3_valid <= stage2_valid;
            if (stage2_valid) begin
                for (j = 0; j < 5; j = j + 1) begin
                    level2_sum[j] <= level1_sum[2*j] + level1_sum[2*j+1];
                end
            end
        end
    end
    
    // Pipeline Stage 4: Accumulation Level 3 (5 -> 3)
    // 5 Inputs: Sum[0]+Sum[1], Sum[2]+Sum[3], Sum[4] (Pass)
    reg signed [2*DATA_WIDTH+5:0] level3_sum [0:2];
    reg stage4_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage4_valid <= 0;
            for (j = 0; j < 3; j = j + 1) level3_sum[j] <= 0;
        end else begin
            stage4_valid <= stage3_valid;
            if (stage3_valid) begin
                level3_sum[0] <= level2_sum[0] + level2_sum[1];
                level3_sum[1] <= level2_sum[2] + level2_sum[3];
                level3_sum[2] <= level2_sum[4]; // Pass through
            end
        end
    end
    
    // Pipeline Stage 5: Accumulation Level 4 (3 -> 2)
    // 3 Inputs: Sum[0]+Sum[1], Sum[2] (Pass)
    reg signed [2*DATA_WIDTH+6:0] level4_sum [0:1];
    reg stage5_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage5_valid <= 0;
            for (j = 0; j < 2; j = j + 1) level4_sum[j] <= 0;
        end else begin
            stage5_valid <= stage4_valid;
            if (stage4_valid) begin
                level4_sum[0] <= level3_sum[0] + level3_sum[1];
                level4_sum[1] <= level3_sum[2]; // Pass through
            end
        end
    end

    // Pipeline Stage 6: Final Accumulation (2 -> 1)
    reg signed [2*DATA_WIDTH+7:0] final_sum;
    reg stage6_valid;
    reg signed [DATA_WIDTH-1:0] bias_delayed [0:5]; // Delay bias to match pipeline depth (6 stages)
    
    // Bias delay line
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k=0; k<6; k=k+1) bias_delayed[k] <= 0;
        end else begin
            bias_delayed[0] <= bias;
            for (k=1; k<6; k=k+1) bias_delayed[k] <= bias_delayed[k-1];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage6_valid <= 0;
            final_sum <= 0;
        end else begin
            stage6_valid <= stage5_valid;
            if (stage5_valid) begin
                final_sum <= level4_sum[0] + level4_sum[1];
            end
        end
    end
    
    // Final Stage: Add Bias and Quantize/Saturate
    reg signed [2*DATA_WIDTH+8:0] sum_with_bias; 
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valid <= 0;
            decision_value <= 0;
            prediction <= 0;
        end else begin
            output_valid <= stage6_valid;
            
            if (stage6_valid) begin
                // sum_with_bias = final_sum + (bias << 8)
                sum_with_bias = final_sum + (bias_delayed[5] <<< FRAC_BITS);
                
                // Scale back to Q8.8 (shift right by 8) and Saturate
                // Extract upper bits above the kept 16-bit result
                // Range: [40 : 23]
                // Use a signed comparison to check for all 0s or all 1s (-1)
                
                if ($signed(sum_with_bias[2*DATA_WIDTH+8 : FRAC_BITS + DATA_WIDTH - 1]) != 0 && 
                    $signed(sum_with_bias[2*DATA_WIDTH+8 : FRAC_BITS + DATA_WIDTH - 1]) != -1) begin
                    // Overflow
                    if (sum_with_bias[2*DATA_WIDTH+8]) // Negative
                        decision_value <= -32768;
                    else
                        decision_value <= 32767;
                end else begin
                     decision_value <= sum_with_bias[FRAC_BITS + DATA_WIDTH - 1 : FRAC_BITS];
                end
                
                
                // Prediction: >= 0 is class 1, else 0
                // Use the sign bit of the full sum (Bit 40). 0=Pos, 1=Neg.
                prediction <= (sum_with_bias[2*DATA_WIDTH+8] == 0) ? 1'b1 : 1'b0;
            end
        end
    end

endmodule
