
from pynq import Overlay
from pynq import MMIO
import numpy as np
import time
import pickle
import json

class SVMAccelerator:
    def __init__(self, bitstream_path, params_path):
        # Load Overlay
        self.overlay = Overlay(bitstream_path)
        # Find IP in the overlay (name 'axi_lite_wrapper_0' from Tcl)
        self.ip_box = self.overlay.axi_lite_wrapper_0
        self.mmio = self.ip_box.mmio
        
        # Load Parameters
        # self.load_parameters(params_path) # Moved to main to avoid overwrite by test
        
    def write_reg(self, offset, value):
        self.mmio.write(offset, int(value))
        
    def read_reg(self, offset):
        return self.mmio.read(offset)

    def load_parameters(self, params_path):
        print("Loading Model Parameters to FPGA Registers...")
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        linear_params = params['linear']
        
        # Write Bias (0x10)
        # Note: Registers take 32-bit int, but hardware uses lower 16 bits signed
        # Python int handles sign, but we might need to mask for display or raw writing
        # MMIO writes unsigned 32-bit usually.
        # If bias is -200 (Q8.8), Python needs to send the 2's complement 32-bit equivalent
        
        bias = linear_params['bias']
        self.write_reg(0x10, bias & 0xFFFFFFFF)
        
        # Write Weights (0x20 + i*4)
        weights = linear_params['weights']
        for i, w in enumerate(weights):
            self.write_reg(0x20 + i*4, int(w) & 0xFFFFFFFF)
            
        print("Model Loaded.")

    def predict(self, features):
        """
        Run inference on a single sample
        reg_map:
        0x00: Control (Bit 0: Start)
        0x04: Status (Bit 0: Done)
        0x08: Result
        0x60: Features Base
        """
        
        # 1. Write Features
        for i, f in enumerate(features):
            self.write_reg(0x60 + i*4, int(f) & 0xFFFFFFFF)
            
        # 1.5 Soft Reset (Bit 1) to clear internal state
        self.write_reg(0x00, 2) 
        self.write_reg(0x00, 0)
        
        # 2. Start (Pulse Bit 0)
        self.write_reg(0x00, 1)
        self.write_reg(0x00, 0)
        
        # 3. Poll for Done (Bit 0 of Status 0x04)
        timeout = 1000
        while timeout > 0:
            status = self.read_reg(0x04)
            if status & 0x01:
                break
            timeout -= 1
            if timeout % 100 == 0:
                 print(f"DEBUG: Waiting for Done... Status=0x{status:08x}")
                 
        if timeout == 0:
            print(f"ERROR: SVM Core Timed out! Status=0x{status:08x}")
            return 0, 0, 0
                
        # 4. Read Result
        # Result is sign-extended 32-bit in RTL, so reading it as unsigned 32-bit from MMIO
        raw_res = self.read_reg(0x08)
        # Convert to signed 32-bit
        if raw_res & 0x80000000:
            result = -((~raw_res + 1) & 0xFFFFFFFF)
        else:
            result = raw_res
            
        prediction = 1 if result >= 0 else 0
        
        # 5. Read Latency (0x0C)
        latency_cycles = self.read_reg(0x0C)
        
        return prediction, result, latency_cycles

def main():
    print("--- FPGA Hardware Validation ---")
    
    # Paths
    bitstream = "svm.bit" 
    params_file = "quantized_params.json"
    
    # Load Real Test Data
    # We need the quantized inputs from the test set
    # Using 'train_and_quantize.py' output would be best, but we'll reload and re-quantize on the fly
    # or save test vectors in json.
    
    # Load Real Test Data (JSON format to avoid sklearn dependency)
    with open('test_data.json', 'r') as f:
        data = json.load(f)
        
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Quantize Test Data (Q8.8)
    def float_to_fixed(val):
        return int(round(val * 256))
        
    accelerator = SVMAccelerator(bitstream, params_file)
    
    print(f"Running Validation on {len(X_test)} samples...")
    
    # DEBUG: Connection Test
    print("DEBUG: Checking AXI Connection...")
    accelerator.write_reg(0x10, 0xAAAA) 
    val = accelerator.read_reg(0x10)
    print(f"DEBUG: Bias Reg (0x10) -> Wrote 0xAAAA, Read 0x{val:04X}")
    
    if val != 0xAAAA:
        print("CRITICAL ERROR: AXI Bus Read/Write Failed. FPGA is not responding.")
        return 0, 0, 0

    # Load Parameters (AFTER Test, to restore Bias!)
    accelerator.load_parameters(params_file)

    correct = 0
    total_cycles = 0
    
    # Benchmark Loop
    start_time = time.time()
    
    for i in range(len(X_test)):
        # Quantize sample
        x_q = [float_to_fixed(f) for f in X_test[i]]
        
        # Run FPGA
        pred, val, cycles = accelerator.predict(x_q)
        
        if pred == y_test[i]:
            correct += 1
            
        total_cycles += cycles
        
        if i < 5:
            print(f"Sample {i}: True={y_test[i]}, Pred={pred}, Val={val/256.0:.4f}, Cycles={cycles}")
            
    end_time = time.time()
    
    acc = correct / len(X_test)
    avg_cycles = total_cycles / len(X_test)
    sw_throughput = len(X_test) / (end_time - start_time)
    
    print("\n--- Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Avg HW Latency: {avg_cycles} Cycles (Approx {avg_cycles * 10} ns @ 100MHz)")
    print(f"SW Loop Throughput: {sw_throughput:.2f} samples/sec (limited by Python MMIO)")

if __name__ == "__main__":
    main()
