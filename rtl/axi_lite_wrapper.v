
`timescale 1ns / 1ps

module axi_lite_wrapper #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 8,
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 8
)(
    // AXI4-Lite Interface
    input wire  s_axi_aclk,
    input wire  s_axi_aresetn,
    
    // Write Address Channel
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] s_axi_awaddr,
    input wire [2 : 0] s_axi_awprot,
    input wire  s_axi_awvalid,
    output wire  s_axi_awready,
    
    // Write Data Channel
    input wire [C_S_AXI_DATA_WIDTH-1 : 0] s_axi_wdata,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] s_axi_wstrb,
    input wire  s_axi_wvalid,
    output wire  s_axi_wready,
    
    // Write Response Channel
    output wire [1 : 0] s_axi_bresp,
    output wire  s_axi_bvalid,
    input wire  s_axi_bready,
    
    // Read Address Channel
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] s_axi_araddr,
    input wire [2 : 0] s_axi_arprot,
    input wire  s_axi_arvalid,
    output wire  s_axi_arready,
    
    // Read Data Channel
    output wire [C_S_AXI_DATA_WIDTH-1 : 0] s_axi_rdata,
    output wire [1 : 0] s_axi_rresp,
    output wire  s_axi_rvalid,
    input wire  s_axi_rready
);

    // AXI4-Lite Signals
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_awaddr;
    reg  axi_awready;
    reg  axi_wready;
    reg [1 : 0] axi_bresp;
    reg  axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_araddr;
    reg  axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1 : 0] axi_rdata;
    reg [1 : 0] axi_rresp;
    reg  axi_rvalid;

    // AXI Assignments
    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = axi_bresp;
    assign s_axi_bvalid  = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = axi_rresp;
    assign s_axi_rvalid  = axi_rvalid;

    // Register Map
    // 0x00: Control (Bit 0: Start, Bit 1: Soft Reset)
    // 0x04: Status  (Bit 0: Done, Bit 1: Busy, Bit 2: Prediction)
    // 0x08: Decision Value (Result)
    // 0x0C: Latency Counter
    // 0x10: Bias
    // 0x20 - 0x6C: Weights (20 regs) -> offsets 0x20 + i*4
    // 0x80 - 0xCC: Features (20 regs) -> offsets 0x80 + i*4
    
    reg [31:0] reg_control;
    reg [31:0] reg_status;
    reg [31:0] reg_result;
    reg [31:0] reg_latency;
    reg [31:0] reg_bias;
    reg [31:0] reg_weights [0:19];
    reg [31:0] reg_features [0:19];
    
    // Internal Signals
    wire start_pulse;
    wire soft_reset;
    wire svm_done;
    wire [15:0] svm_decision; // Q8.8
    wire svm_prediction;
    
    // Flattened Arrays for SVM Core
    reg [16*20-1:0] features_flat;
    reg [16*20-1:0] weights_flat;
    
    genvar k;
    generate
        for (k = 0; k < 20; k = k + 1) begin : gen_flat
            always @(*) begin
                features_flat[(k+1)*16-1 : k*16] = reg_features[k][15:0];
                weights_flat[(k+1)*16-1 : k*16]  = reg_weights[k][15:0];
            end
        end
    endgenerate
    
    // Instantiate Linear SVM
    
    linear_svm #(
        .DATA_WIDTH(16),
        .FRAC_BITS(8),
        .NUM_FEATURES(20)
    ) core (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn & ~reg_control[1]), // Use reg_control directly
        .input_valid(start_pulse), // Pulse acts as valid
        .features_flat(features_flat),
        .weights_flat(weights_flat),
        .bias(reg_bias[15:0]),
        .output_valid(svm_done),
        .decision_value(svm_decision),
        .prediction(svm_prediction)
    );
    
    // Status Logic
    // Edge detection for start pulse
    reg reg_control_0_d;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) reg_control_0_d <= 0;
        else reg_control_0_d <= reg_control[0];
    end
    
    // Pulse only on 0->1 transition
    assign start_pulse = reg_control[0] & ~reg_control_0_d;
    assign soft_reset  = reg_control[1];

    reg [31:0] cycle_counter; // Moved here from Latency Timer block
    always @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            reg_status <= 0;
            cycle_counter <= 0;
            reg_result <= 0;
            // reg_control logic moved to AXI write block exclusively
        end else begin
            // Capture Result
            if (svm_done) begin
                reg_status[0] <= 1; // Done
                reg_status[1] <= 0; // Not Busy
                reg_status[2] <= svm_prediction;
                reg_result <= {{16{svm_decision[15]}}, svm_decision}; // Sign extend
                reg_latency <= cycle_counter + 1; // Capture final latency
            end else if (start_pulse) begin
                reg_status[0] <= 0; // Not Done
                reg_status[1] <= 1; // Busy
                cycle_counter <= 0; // Reset Counter
            end else if (reg_status[1]) begin
                cycle_counter <= cycle_counter + 1; // Increment if Busy
            end
        end
    end

    // AXI Write Logic
    integer i;
    always @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_bvalid  <= 1'b0;
            axi_bresp   <= 2'b0;
            // Registers reset
            reg_bias <= 0;
            for (i=0; i<20; i=i+1) begin
                 reg_weights[i] <= 0;
                 reg_features[i] <= 0;
            end
        end else begin
            // Handshake Logic (Standard AXI Lite)
            if (~axi_awready && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready <= 1'b1;
                axi_wready  <= 1'b1;
                
                // Write Operation
                case (s_axi_awaddr[7:2]) // Word aligned address
                    6'h00: reg_control <= s_axi_wdata; // 0x00
                    // 0x04 Status is Read Only
                    // 0x08 Result is Read Only
                    // 0x0C Latency is Read Only
                    6'h04: reg_bias <= s_axi_wdata; // 0x10 -> 16/4 = 4. 
                    
                    // Weights: 0x20 = 32. 32/4 = 8.
                    // Range 8 to 8+19 (27)
                    
                    // Features: 0x80 = 128. 128/4 = 32.
                    // Range 32 to 32+19 (51)
                    
                    default: begin
                        if (s_axi_awaddr[7:2] >= 8 && s_axi_awaddr[7:2] < 28)
                             reg_weights[s_axi_awaddr[7:2] - 8] <= s_axi_wdata;
                        else if (s_axi_awaddr[7:2] >= 32 && s_axi_awaddr[7:2] < 52)
                             reg_features[s_axi_awaddr[7:2] - 32] <= s_axi_wdata;
                    end
                endcase
                
            end else begin
                axi_awready <= 1'b0;
                axi_wready  <= 1'b0;
            end
            
            if (axi_awready && s_axi_awvalid && ~axi_bvalid && axi_wready && s_axi_wvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b0;
            end else if (s_axi_bready && axi_bvalid) begin
                axi_bvalid <= 1'b0; 
            end
        end
    end

    // AXI Read Logic
    always @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            axi_rresp   <= 2'b0;
            axi_rdata   <= 0;
        end else begin
            if (~axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
                axi_araddr  <= s_axi_araddr;
            end else begin
                axi_arready <= 1'b0;
            end
            
            if (axi_arready && s_axi_arvalid && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b0;
                
                // Read Mux
                case (axi_araddr[7:2])
                    0: axi_rdata <= reg_control;
                    1: axi_rdata <= reg_status;
                    2: axi_rdata <= reg_result;
                    3: axi_rdata <= reg_latency;
                    4: axi_rdata <= reg_bias;
                    default: begin
                         if (axi_araddr[7:2] >= 8 && axi_araddr[7:2] < 28)
                             axi_rdata <= reg_weights[axi_araddr[7:2] - 8];
                         else if (axi_araddr[7:2] >= 32 && axi_araddr[7:2] < 52)
                             axi_rdata <= reg_features[axi_araddr[7:2] - 32];
                         else
                             axi_rdata <= 0;
                    end
                endcase
            end else if (axi_rvalid && s_axi_rready) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
