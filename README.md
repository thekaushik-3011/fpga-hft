# HFT Linear SVM FPGA Accelerator (20 Features)
High-Frequency Trading (HFT) requires ultra-low latency decision making. This project implements a **Linear Support Vector Machine (SVM)** on a **PYNQ-Z2 FPGA**, achieving **~60ns latency** (6 clock cycles @ 100MHz) for inference, compared to microseconds in software.

This design supports **20 Input Features** and uses **Q8.8 Fixed-Point Arithmetic** for high-speed, DSP-optimized processing.

---

## 🚀 Key Features
*   **Architecture:** 6-Stage Pipelined Adder Tree.
*   **Throughput:** 1 Prediction per Clock Cycle (after pipeline fill).
*   **Latency:** 6 Clock Cycles (~60ns @ 100MHz).
*   **Precision:** Q8.8 Fixed-Point (8 integer bits, 8 fractional bits).
*   **Interface:** AXI4-Lite Slave (Compatible with PYNQ/Python).
*   **Accuracy:** **~72.8%** (Matches Software Simulation).

---

## 📂 Project Structure
```
svm_fpga_accelerator/
├── rtl/                    # Verilog Hardware Design
│   ├── linear_svm.v        # 20-Feature Pipelined Dot Product Engine
│   └── axi_lite_wrapper.v  # AXI4-Lite Interface (Registers: 0x00-0xCC)
├── testbench/              # Verification
│   ├── tb_linear_svm.v     # Verilog Testbench (1000 Test Cases)
│   ├── svm_test_vectors.vh # Generated Test Vectors
│   └── test_data.mem       # Test Data Memory File
├── simulation/             # Python Model & Data
│   ├── dataset_generation.py   # Generates 20-feature HFT market data
│   └── fixed_point_sim.py      # Bit-accurate Q8.8 Simulator
├── pynq/                   # Deployment Scripts
│   ├── build_bitstream.tcl     # Vivado Automation Script
│   ├── pynq_driver.py          # Python Driver (Running on PYNQ Board)
│   └── deployment/             # Ready-to-deploy files
├── results/                # Generated Artifacts
│   ├── linear_params.json      # Model Weights/Bias (Integer Scaled)
│   └── test_data.json          # Test vectors for validation
└── README.md               # Project Documentation
```

---

## 🛠️ Reproduction Steps

### 1. Simulation & Data Generation
Use Python to generate synthetic market data, train the Linear SVM, and export parameters.
```bash
# Generate data and train model
python3 generate_test_vectors.py
```
**Output:** `results/linear_params.json`, `testbench/svm_test_vectors.vh`, `testbench/test_data.mem`

### 2. RTL Verification
Run the Verilog testbench to verify the logic against 1000 test cases.
```bash
iverilog -o testbench/tb_linear_svm.vvp -I testbench rtl/linear_svm.v testbench/tb_linear_svm.v
vvp testbench/tb_linear_svm.vvp
```
**Expected Output:** `ALL TESTS PASSED`

### 3. Bitstream Generation (Vivado)
Generate the FPGA configuration file (`.bit`).
```bash
vivado -mode batch -source pynq/build_bitstream.tcl
```
**Output:** `pynq/output/svm.bit`, `pynq/output/svm.hwh`

---

## ⚡ Deployment on PYNQ-Z2

### 1. Transfer Files to Board
Copy the bitstream, hardware handoff, driver, and model parameter files to the PYNQ board using `scp`.
*(Replace `192.168.2.99` with your board's IP address)*

```bash
scp pynq/output/svm.bit \
    pynq/output/svm.hwh \
    pynq/pynq_driver.py \
    results/linear_params.json \
    results/test_data.json \
    xilinx@192.168.2.99:/home/xilinx/svm_accelerator/
```

### 2. Run Inference on Board
SSH into the board and execute the Python driver.
```bash
ssh xilinx@192.168.2.99
cd svm_accelerator
sudo python3 pynq_driver.py
```

**Expected Console Output:**
```
Loading Bitstream...
Bitstream Loaded.
Loading Model Parameters to FPGA Registers...
Model Loaded.
Starting Hardware Inference on 1000 samples...
Hardware Inference Complete.
Accuracy: 72.80%
Latency per sample: 6 cycles
```

---

## 🐛 Resolved Issues & Technical Notes

### 1. Signed Saturation Logic
*   **Issue:** The initial reduction-based saturation check `!(&...) && (|...)` failed for signed negative numbers, causing output to lock at `-32768`.
*   **Fix:** Replaced with explicit signed upper-bit comparison:
    ```verilog
    if ($signed(sum[...]) != 0 && $signed(sum[...]) != -1) // Overflow
    ```

### 2. Feature Writing (Driver)
*   **Issue:** The Python driver sent negative integers (e.g., `-128`) directly to MMIO, which were interpreted incorrectly by the 32-bit interface.
*   **Fix:** Enforced 32-bit masking in `pynq_driver.py`:
    ```python
    self.mmio.write(offset, int(value) & 0xFFFFFFFF)
    ```

### 3. Feature Count
*   **Update:** The design was upgraded from 16 to **20 features** to match the latest trained model. Address maps in `axi_lite_wrapper.v` and `pynq_driver.py` were updated accordingly.
