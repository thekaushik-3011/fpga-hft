# HFT Linear SVM FPGA Accelerator
High-Frequency Trading (HFT) requires ultra-low latency decision making. This project implements a **Linear Support Vector Machine (SVM)** on a **PYNQ-Z2 FPGA**, achieving **~60ns latency** (6 clock cycles @ 100MHz) for inference, compared to microseconds in software.

## 📂 Project Structure
```
svm_fpga_accelerator/
├── simulation/             # Python Golden Model
│   ├── dataset_generation.py   # Generates 16-feature HFT market data
│   ├── train_and_quantize.py   # Trains SVM & converts to Q8.8 Fixed-Point
│   └── hardware_eval.py        # Validates quantization accuracy
├── rtl/                    # Verilog Hardware Design
│   ├── linear_svm.v            # 16-channel Pipelined Dot Product Engine
│   └── axi_lite_wrapper.v      # AXI4-Lite Slave Interface & Control Logic
├── pynq/                   # Deployment Scripts
│   ├── build_bitstream.tcl     # Vivado Automation Script
│   └── pynq_driver.py          # Python Driver (Runs on PYNQ Board)
├── results/                # Generated Artifacts
│   ├── quantized_params.json   # Weights/Bias for Hardware
│   └── test_data.json          # Test vectors for validation
```

## 🚀 Workflow

### 1. Simulation & Training
Generate synthetic market data, train the model, and export quantized parameters (Q8.8).
```bash
cd simulation
# 1. Generate Data (16 features)
python3 dataset_generation.py
# 2. Train & Quantize (StandardScaler + Q8.8)
python3 train_and_quantize.py
```
**Output:** `results/quantized_params.json` (Weights), `results/test_data.json` (Test Set).

### 2. Hardware Build (Vivado)
Generate the FPGA Bitstream (`.bit`) from the Verilog source.
1.  Open **Vivado Tcl Console**.
2.  Run:
    ```tcl
    cd pynq
    source build_bitstream.tcl
    ```
**Output:** `pynq/pynq_output/svm.bit`, `svm.hwh`.

### 3. Deployment (PYNQ-Z2)
Run the accelerator on the physical board.
1.  **Transfer Files** to the board (replace `xilinx@192.168.2.99` with your board IP):
    ```bash
    scp pynq/pynq_output/svm.bit \
        pynq/pynq_output/svm.hwh \
        pynq/pynq_driver.py \
        results/quantized_params.json \
        results/test_data.json \
        xilinx@192.168.2.99:/home/xilinx/svm_accelerator/
    ```
2.  **Run Validation** on the board:
    ```bash
    ssh xilinx@192.168.2.99
    cd svm_accelerator
    sudo -E python3 pynq_driver.py
    ```

## 📊 Performance Results
| Metric | Software (Python) | Hardware (FPGA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Latency** | ~15 µs | **60 ns** (6 Cycles) | **250x Faster** |
| **Throughput** | Sequential | 1 Prediction / Cycle | Massive |
| **Accuracy** | ~65% (Float) | ~64% (Q8.8 Fixed) | <1% Drop |

## 🛠️ Key Technical Features
*   **Q8.8 Fixed-Point Arithmetic:** Optimized for FPGA DSP slices.
*   **Pipeline Architecture:** Processes new data every clock cycle.
*   **AXI4-Lite Interface:** Easy integration with Python/PYNQ.
*   ** robust Edge Detection:** Prevents multi-driver signal conflicts.
