`timescale 1ns / 1ps

module kernel_svm #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS = 8,
    parameter NUM_FEATURES = 16,
    parameter NUM_SUPPORT_VECTORS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire input_valid,
    input wire signed [DATA_WIDTH*NUM_FEATURES-1:0] features_flat,
    
    // Support Vectors and Dual Coefs would typically be stored in BRAM
    // For this module, we'll assume they are inputs or internal memory
    // To simplify, let's assume they are provided as large flat inputs for now
    // In a real design, we'd stream them or read from BRAM
    input wire signed [DATA_WIDTH*NUM_FEATURES*NUM_SUPPORT_VECTORS-1:0] sv_flat,
    input wire signed [DATA_WIDTH*NUM_SUPPORT_VECTORS-1:0] dual_coef_flat,
    
    input wire signed [DATA_WIDTH-1:0] bias,
    
    output reg output_valid,
    output reg signed [DATA_WIDTH-1:0] decision_value,
    output reg prediction
);

    // Unpack inputs
    wire signed [DATA_WIDTH-1:0] features [0:NUM_FEATURES-1];
    wire signed [DATA_WIDTH-1:0] support_vectors [0:NUM_SUPPORT_VECTORS-1][0:NUM_FEATURES-1];
    wire signed [DATA_WIDTH-1:0] dual_coefs [0:NUM_SUPPORT_VECTORS-1];

    genvar i, j;
    generate
        for (i = 0; i < NUM_FEATURES; i = i + 1) begin
             assign features[i] = features_flat[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
        end
        
        for (j = 0; j < NUM_SUPPORT_VECTORS; j = j + 1) begin
             assign dual_coefs[j] = dual_coef_flat[(j+1)*DATA_WIDTH-1 : j*DATA_WIDTH];
             for (i = 0; i < NUM_FEATURES; i = i + 1) begin
                 assign support_vectors[j][i] = sv_flat[((j*NUM_FEATURES)+i+1)*DATA_WIDTH-1 : ((j*NUM_FEATURES)+i)*DATA_WIDTH];
             end
        end
    endgenerate

    // Stage 1: Compute Squared Euclidean Distance ||x - sv||^2
    // Parallelize across all SVs (Area-heavy but fast)
    // For each SV, compute sum((x_k - sv_jk)^2)
    
    reg signed [2*DATA_WIDTH+4:0] dist_sq [0:NUM_SUPPORT_VECTORS-1]; // Result is positive
    reg stage1_valid;
    
    integer k, m;
    reg signed [DATA_WIDTH-1:0] diff;
    reg signed [2*DATA_WIDTH-1:0] diff_sq;
    reg signed [2*DATA_WIDTH+4:0] sum_sq;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 0;
            for (m=0; m<NUM_SUPPORT_VECTORS; m=m+1) dist_sq[m] <= 0;
        end else begin
            stage1_valid <= input_valid;
            if (input_valid) begin
                for (m=0; m<NUM_SUPPORT_VECTORS; m=m+1) begin
                    sum_sq = 0;
                    for (k=0; k<NUM_FEATURES; k=k+1) begin
                        diff = features[k] - support_vectors[m][k];
                        diff_sq = diff * diff; // Q8.8 * Q8.8 = Q16.16
                        // Accumulate Q16.16
                        // Note: Depending on range, might need to right shift first
                        // Let's assume standard accumulation
                        sum_sq = sum_sq + (diff_sq >>> FRAC_BITS); // Back to Q8.8 range roughly
                    end
                    dist_sq[m] <= sum_sq; 
                end
            end
        end
    end

    // Stage 2: LUT Lookup (RBF Kernel)
    // kernel_val = exp(-gamma * dist_sq)
    // Map dist_sq to LUT address
    
    reg [7:0] lut_addr [0:NUM_SUPPORT_VECTORS-1];
    wire signed [DATA_WIDTH-1:0] kernel_vals [0:NUM_SUPPORT_VECTORS-1];
    reg stage2_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
           stage2_valid <= 0;
        end else begin
           stage2_valid <= stage1_valid;
           // Address mapping logic
        end
    end
    
    generate
        for (j=0; j<NUM_SUPPORT_VECTORS; j=j+1) begin : gen_luts
            // Saturate distance to 8-bit address
            always @(*) begin
                 if (dist_sq[j] > 255) lut_addr[j] = 255;
                 else if (dist_sq[j] < 0) lut_addr[j] = 0;
                 else lut_addr[j] = dist_sq[j][7:0];
            end
            
            exp_lut #(
                .DATA_WIDTH(DATA_WIDTH),
                .ADDR_WIDTH(8)
            ) lut_inst (
                .clk(clk),
                .addr(lut_addr[j]),
                .data(kernel_vals[j])
            );
        end
    endgenerate

    // Stage 3: Multiply by Dual Coefficients
    // weighted = alpha_j * K(x, sv_j)
    reg signed [2*DATA_WIDTH-1:0] weighted_vals [0:NUM_SUPPORT_VECTORS-1];
    reg stage3_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage3_valid <= 0;
            for (m=0; m<NUM_SUPPORT_VECTORS; m=m+1) weighted_vals[m] <= 0;
        end else begin
            stage3_valid <= stage2_valid;
            if (stage2_valid) begin
                for (m=0; m<NUM_SUPPORT_VECTORS; m=m+1) begin
                    weighted_vals[m] <= dual_coefs[m] * kernel_vals[m];
                end
            end
        end
    end

    // Stage 4: Accumulate
    reg signed [2*DATA_WIDTH+4:0] final_sum;
    reg stage4_valid;
    
    // Adder tree or serial accumulation
    // Let's do a simple behavioral sum for readability (synthesis handles tree)
    reg signed [2*DATA_WIDTH+4:0] accum;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage4_valid <= 0;
            final_sum <= 0;
        end else begin
            stage4_valid <= stage3_valid;
            if (stage3_valid) begin
                accum = 0;
                for (m=0; m<NUM_SUPPORT_VECTORS; m=m+1) begin
                    accum = accum + weighted_vals[m];
                end
                final_sum <= accum;
            end
        end
    end

    // Stage 5: Add Bias and Output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valid <= 0;
            decision_value <= 0;
            prediction <= 0;
        end else begin
            output_valid <= stage4_valid;
            if (stage4_valid) begin
                // final_sum is sum of (Q8.8 * Q8.8) = Q16.16
                // bias is Q8.8
                // result = (final_sum >> 8) + bias
                
                decision_value <= (final_sum >>> FRAC_BITS) + bias;
                prediction <= (((final_sum >>> FRAC_BITS) + bias) >= 0) ? 1'b1 : 1'b0;
            end
        end
    end

endmodule
