# simulation/train_and_quantize.py
"""
SVM Training and Fixed-Point Quantization
Trains Linear SVM
Evaluates Accuracy for:
1. Floating Point (Baseline)
2. 16-bit Fixed Point (Q8.8) - Hardware Target
3. 8-bit Fixed Point (Q3.5) - Low Precision Experiment
"""

import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import json
import pickle

class FixedPointConverter:
    """Fixed-Point Converter"""
    
    def __init__(self, total_bits, frac_bits):
        self.total_bits = total_bits
        self.frac_bits = frac_bits
        self.int_bits = total_bits - frac_bits
        self.scale = 2 ** frac_bits
        
        # Signed range: -2^(N-1) to 2^(N-1) - 1
        self.min_val = -(2 ** (total_bits - 1))
        self.max_val = (2 ** (total_bits - 1)) - 1
        
        print(f"Initialized Q{self.int_bits}.{self.frac_bits} ({total_bits}-bit): Scale={self.scale}, Range=[{self.min_val/self.scale:.4f}, {self.max_val/self.scale:.4f}]")
        
    def float_to_fixed(self, value):
        """Convert float to integer representation"""
        scaled = value * self.scale
        rounded = np.round(scaled)
        clipped = np.clip(rounded, self.min_val, self.max_val)
        return clipped.astype(np.int32)
    
    def fixed_to_float(self, fixed_val):
        """Convert integer representation back to float"""
        return fixed_val / self.scale

    def quantize_array(self, arr):
        return self.float_to_fixed(arr)

class SVMSimulator:
    """Simulates Fixed-Point Inference"""
    
    @staticmethod
    def simulate_linear(X_float, weights_float, bias_float, converter):
        """
        Simulate dot product using integer arithmetic.
        X, W, B are converted to fixed-point integers.
        Result = (Sum(X_i * W_i)) + (B << frac_bits)
        Note: X*W produces 2*frac_bits. B needs to be aligned.
        Actually, usually we align everything to the same point or handle the double precision.
        
        Standard FPGA Mult: Q8.8 * Q8.8 = Q16.16
        Accumulator: Q16.16
        Bias: Q8.8 -> Shift to Q16.16? Or add before?
        
        Let's assume a simpler model often used:
        y = (Sum(x_q * w_q) >> frac_bits) + b_q
        This keeps the accumulator result compatible with the bias format.
        """
        
        print(f"Simulating {converter.total_bits}-bit Inference...")
        
        # 1. Quantize Inputs and Weights
        X_q = converter.quantize_array(X_float) # Shape (N, 40)
        W_q = converter.quantize_array(weights_float) # Shape (40,)
        b_q = converter.quantize_array(bias_float) # Scalar
        
        # 2. Integer Dot Product
        # Accumulator can grow large, Python handles large ints auto
        dot_product = np.dot(X_q, W_q) # (N,) array of integers
        
        # 3. Scaling Adjustment
        # X was scaled by 2^F, W by 2^F. Dot product is scaled by 2^(2F).
        # We need to bring it back to 2^F to add with Bias (which is 2^F).
        # Standard approach: Shift right by F.
        dot_product_scaled = np.floor(dot_product / (2 ** converter.frac_bits)).astype(np.int32)
        
        # 4. Add Bias
        decision_function = dot_product_scaled + b_q
        
        # 5. Prediction (Sign Bit)
        # Class 1 if >= 0, else 0
        y_pred = (decision_function >= 0).astype(int)
        
        return y_pred

def main():
    # 1. Load Data
    data_dir = "../svm_fpga_accelerator/results" # Adjusted path based on execution location
    if not os.path.exists(data_dir):
        # Fallback if running from within simulation folder
        data_dir = "../results"
        
    print(f"Loading data from {data_dir}...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Map labels {-1, 1} or {1, 2, 3}? 
    # FI-2010 labels are 1 (Up), 2 (Stationary), 3 (Down).
    # We need binary for simple SVM.
    # Let's map: 1 (Up) -> 1, 2/3 (Stationary/Down) -> 0?
    # Or strict Up vs Down (drop Stationary)?
    # Let's check unique values
    unique_labels = np.unique(y_test)
    print(f"Original Labels: {unique_labels}")
    
    # Filter/Map Labels
    # Logs show labels are Z-scored floats, approximately centered at 0? 
    # Or maybe they are just regression targets.
    # We'll threshold at 0 to create binary classes (Up vs Down/Stationary).
    # To be more precise for HFT, we might want to drop small values (Stationary).
    
    threshold = 0.0
    print(f"Thresholding labels at {threshold}...")
    
    # Create Binary Target: 1 if > 0, else 0
    y_train_bin = np.where(y_train > threshold, 1, 0)
    y_test_bin = np.where(y_test > threshold, 1, 0)
    
    # For simulation stability, if "Stationary" is dominant, we might want to drop it.
    # But let's verify class balance first.
    unique, counts = np.unique(y_train_bin, return_counts=True)
    print(f"Train Class Balance: {dict(zip(unique, counts))}")
    
    if len(unique) < 2:
        print("WARNING: Single class detected after thresholding. Adjusting strategy.")
        # Try Median split if 0 doesn't work?
        median = np.median(y_train)
        print(f"Retrying with Median Split ({median})...")
        y_train_bin = np.where(y_train > median, 1, 0)
        y_test_bin = np.where(y_test > median, 1, 0)
        unique, counts = np.unique(y_train_bin, return_counts=True)
        print(f"New Train Class Balance: {dict(zip(unique, counts))}")

    X_train_bin = X_train
    X_test_bin = X_test

    # 2. Train Float Model (Linear SVM)
    print("\nTraining Linear SVM (Float32)...")
    clf = LinearSVC(C=1.0, dual=False, max_iter=10000)
    clf.fit(X_train_bin, y_train_bin)
    
    # Evaluate Float
    y_pred_float = clf.predict(X_test_bin)
    acc_float = accuracy_score(y_test_bin, y_pred_float)
    print(f"Floating Point Accuracy: {acc_float*100:.2f}%")
    
    # Extract Weights/Bias
    W_float = clf.coef_[0]
    b_float = clf.intercept_[0]
    
    # 3. define Converters
    # 16-bit: Q8.8 (Standard)
    # Range [-128, 127], Precision 1/256 (~0.0039)
    # Data is Z-score scaled (~[-3, 3]), so Q8.8 is plenty of headroom, decent precision.
    conv_16 = FixedPointConverter(16, 8)
    
    # 8-bit: Need to fit [-3, 3]. 
    # 3 bits integer (incl sign) -> range [-4, 3.xxx]. 
    # int_bits=3 => frac_bits=5.
    # Q3.5: Sign + 2 integer + 5 fractional.
    # Range: -4.00 to 3.96. Precision: 1/32 (0.03125).
    # Let's try Q3.5 for 8-bit.
    conv_8 = FixedPointConverter(8, 5)
    
    # 4. Simulate & Evaluate
    
    # 16-bit
    y_pred_16 = SVMSimulator.simulate_linear(X_test_bin, W_float, b_float, conv_16)
    acc_16 = accuracy_score(y_test_bin, y_pred_16)
    
    # 8-bit
    y_pred_8 = SVMSimulator.simulate_linear(X_test_bin, W_float, b_float, conv_8)
    acc_8 = accuracy_score(y_test_bin, y_pred_8)
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Floating Point Accuracy:   {acc_float*100:.2f}%")
    print(f"16-bit (Q8.8) Accuracy:    {acc_16*100:.2f}%  ( Delta: {acc_16-acc_float:.2%} )")
    print(f"8-bit  (Q3.5) Accuracy:    {acc_8*100:.2f}%  ( Delta: {acc_8-acc_float:.2%} )")
    
    # 5. Save Outputs
    print("\nSaving 16-bit Quantized Parameters for Hardware...")
    
    W_q16 = conv_16.quantize_array(W_float)
    b_q16 = conv_16.float_to_fixed(b_float)
    
    params = {
        'linear': {
            'weights': W_q16.tolist(),
            'bias': int(b_q16),
            'n_features': len(W_q16),
            'format': "Q8.8"
        }
    }
    
    # Save JSON
    json_path = os.path.join(data_dir, 'quantized_params.json')
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)
        
    print(f"Saved to {json_path}")

    # Save test data for PYNQ driver (Binary Labels)
    # Scale X_test to Q8.8 integers
    X_test_int = conv_16.quantize_array(X_test_bin)
    
    # Create test data dictionary
    test_data = {
        'X_test': X_test_int.tolist(),
        'y_test': y_test_bin.tolist() 
    }
    
    test_data_path = os.path.join(data_dir, 'test_data.json')
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f)
        
    print(f"Saved test data (binary labels) to {test_data_path}")
    
    # Save models pickle
    with open(os.path.join(data_dir, 'svm_models.pkl'), 'wb') as f:
        pickle.dump({'model_float': clf, 'metrics': {'acc_float': acc_float, 'acc_16': acc_16, 'acc_8': acc_8}}, f)

if __name__ == "__main__":
    main()