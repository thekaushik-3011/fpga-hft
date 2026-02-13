SIM_DIR = simulation
RTL_DIR = rtl
TB_DIR = testbench
BUILD_DIR = build

IV = iverilog
VVP = vvp

# Sources
RTL_SRCS_LINEAR = $(RTL_DIR)/linear_svm.v
RTL_SRCS_KERNEL = $(RTL_DIR)/kernel_svm.v $(RTL_DIR)/exp_lut.v
TB_SRCS_LINEAR = $(TB_DIR)/tb_linear_svm.v
TB_SRCS_KERNEL = $(TB_DIR)/tb_kernel_svm.v

# Targets
.PHONY: all sim_linear sim_kernel clean

all: sim_linear sim_kernel

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/svm_linear_sim: $(RTL_SRCS_LINEAR) $(TB_SRCS_LINEAR) | $(BUILD_DIR)
	$(IV) -o $@ -I $(RTL_DIR) $^

$(BUILD_DIR)/svm_kernel_sim: $(RTL_SRCS_KERNEL) $(TB_SRCS_KERNEL) | $(BUILD_DIR)
	$(IV) -o $@ -I $(RTL_DIR) $^

sim_linear: $(BUILD_DIR)/svm_linear_sim
	$(VVP) $<

sim_kernel: $(BUILD_DIR)/svm_kernel_sim
	$(VVP) $<

clean:
	rm -rf $(BUILD_DIR) *.vcd
