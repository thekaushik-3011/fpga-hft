# PYNQ-Z2 Deployment Guide: Linear SVM Accelerator

This directory contains the files needed to deploy the Linear SVM to the Xilinx PYNQ-Z2 board.

## File Overview

| File | Location | Purpose |
| :--- | :--- | :--- |
| **`axi_lite_wrapper.v`** | `../rtl/` | **The Hardware Wrapper.** Wraps the Linear SVM core with an AXI4-Lite interface so the processor can talk to it. Includes a cycle counter for precise latency measurement. |
| **`build_bitstream.tcl`** | `./` | **The Builder Script.** run this in Vivado (on your PC) to automatically create the project, synthesize the design, and generate the bitstream (`.bit`) and hardware handoff (`.hwh`). |
| **`pynq_driver.py`** | `./` | **The Board Driver.** Run this on the PYNQ-Z2 board (via Jupyter or SSH). It loads the hardware overlay, sends test data, and verifies the results. |

## Board Setup & Connection

Before running anything, ensure your PYNQ-Z2 board is set up correctly:

1.  **Boot Mode Jumper:** Set the jumper (near the HDMI port) to **SD** (if booting from SD card).
2.  **Power:** Connect the Micro-USB cable to **PROG/UART** (or use external 12V power). Turn the switch **ON**.
3.  **Ethernet:** Connect an Ethernet cable from the board to your PC (or router).
4.  **Network Config (PC Side):**
    *   The board default IP is usually `192.168.2.99`.
    *   Set your PC's Ethernet adapter to a **Static IP**: `192.168.2.1` (Subnet Mask: `255.255.255.0`).
    *   Test connection: `ping 192.168.2.99`

---

## Step 1: Generate the Hardware (On Host PC)

1.  Open **Vivado** (2018.3 or newer recommended).
2.  Open the Tcl Console (Window -> Tcl Console).
3.  Navigate to this directory:
    ```tcl
    cd /path/to/project/pynq
    ```
4.  Run the build script:
    ```tcl
    source build_bitstream.tcl
    ```
    *This will take 10-20 minutes. It creates a project `svm_pynq`, runs synthesis/implementation, and outputs `pynq_output/svm.bit` and `pynq_output/svm.hwh`.*

---

## Step 2: Transfer Files to PYNQ-Z2

Connect to your board (via Ethernet/WiFi) and copy the following files to a folder (e.g., `/home/xilinx/svm_accelerator`):

1.  `pynq_output/svm.bit` (Generated in Step 1)
2.  `pynq_output/svm.hwh` (Generated in Step 1)
3.  `pynq_driver.py`
4.  `../results/quantized_params.json` (Model weights)
5.  `../results/svm_models.pkl` (Test dataset)

**Example using SCP:**
```bash
scp -r pynq_output/svm.bit pynq_output/svm.hwh pynq_driver.py ../results/quantized_params.json ../results/svm_models.pkl xilinx@192.168.2.99:/home/xilinx/svm_accelerator/
```

---

## Step 3: Run Validation (On PYNQ-Z2 Board)

1.  SSH into the board or open a Terminal in Jupyter Lab.
    ```bash
    ssh xilinx@192.168.2.99
    # password: xilinx
    ```
2.  Navigate to the directory:
    ```bash
    cd /home/xilinx/svm_accelerator
    ```
3.  Run the driver with root privileges (required for loading overlays):
    ```bash
    sudo python3 pynq_driver.py
    ```

### Expected Output
```text
--- FPGA Hardware Validation ---
Loading Model Parameters to FPGA Registers...
Model Loaded.
Running Validation on 2400 samples...
Sample 0: True=1.0, Pred=1, Val=0.7852, Cycles=6
Sample 1: True=0.0, Pred=0, Val=-1.234, Cycles=6
...
--- Results ---
Accuracy: 65.29%
Avg HW Latency: 6.0 Cycles (Approx 30.0 ns @ 200MHz)
```
