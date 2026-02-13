`timescale 1ns / 1ps

module exp_lut #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 8
)(
    input wire clk,
    input wire [ADDR_WIDTH-1:0] addr,
    output reg signed [DATA_WIDTH-1:0] data
);

    // 256-entry LUT for exp(-gamma * ||x-sv||^2)
    // Input address maps to distance squared
    // Output data maps to kernel value (0 to 1.0 in Q8.8, i.e., 0 to 256)
    
    reg signed [DATA_WIDTH-1:0] rom [0:2**ADDR_WIDTH-1];

    integer i;
    initial begin
        // Initialize with default values (will be replaced by synthesis or mem load)
        for (i = 0; i < 2**ADDR_WIDTH; i = i + 1) begin
             // Simple placeholder: 256 * exp(-i/32)
             // This needs to match the Python model's generation logic
             // Python uses gamma and dist_sq
             // Let's assume a generic decay for now to allow synthesis
             if (i < 64) rom[i] = 256 - i*3;
             else if (i < 128) rom[i] = 64 - (i-64);
             else rom[i] = 0;
        end
    end

    always @(posedge clk) begin
        data <= rom[addr];
    end

endmodule
