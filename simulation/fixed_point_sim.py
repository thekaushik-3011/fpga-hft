# simulation/fixed_point_sim.py
"""
Bit-Accurate Fixed-Point Inference Simulation
Simulates hardware behavior in Python
"""

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class FixedPointSVM:
    """Bit-accurate fixed-point SVM simulator"""
    
    def __init__(self, int_bits=8, frac_bits=8):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.total_bits = int_bits + frac_bits
        
    def saturate(self, value):
        """Saturate to 16-bit signed range"""
        max_val = 2**15 - 1
        min_val = -2**15
        return np.clip(value, min_val, max_val)
    
    def fixed_mult(self, a, b):
        """Q8.8 x Q8.8 -> Q8.8 multiplication"""
        # Multiply: Q8.8 * Q8.8 = Q16.16
        product = np.int32(a) * np.int32(b)
        # Shift right by frac_bits to get Q8.8
        result = product >> self.frac_bits
        # Saturate
        return self.saturate(result)
    
    def fixed_add(self, a, b):
        """Q8.8 + Q8.8 -> Q8.8 addition"""
        result = a + b
        return self.saturate(result)

class LinearSVMSimulator(FixedPointSVM):
    """Bit-accurate Linear SVM simulator"""
    
    def __init__(self, weights_fp, bias_fp):
        super().__init__()
        self.weights = np.array(weights_fp, dtype=np.int16)
        self.bias = np.int16(bias_fp)
        self.n_features = len(weights_fp)
        self.operation_count = 0
        
    def predict_sample(self, x_fp):
        """Predict single sample"""
        self.operation_count = 0
        x = np.array(x_fp, dtype=np.int16)
        
        # Dot product: w^T x
        acc = np.int32(0)
        for i in range(self.n_features):
            prod = self.fixed_mult(self.weights[i], x[i])
            acc = self.fixed_add(acc, prod)
            self.operation_count += 2  # 1 mult, 1 add
        
        # Add bias
        result = self.fixed_add(acc, self.bias)
        self.operation_count += 1
        
        # Sign bit determines class
        prediction = 1 if result >= 0 else 0
        
        return prediction, result
    
    def predict(self, X_fp):
        """Predict batch"""
        predictions = []
        decision_values = []
        total_ops = 0
        
        for sample in X_fp:
            pred, val = self.predict_sample(sample)
            predictions.append(pred)
            decision_values.append(val)
            total_ops += self.operation_count
        
        avg_ops = total_ops / len(X_fp)
        
        return np.array(predictions), np.array(decision_values), avg_ops

class KernelSVMSimulator(FixedPointSVM):
    """Bit-accurate RBF Kernel SVM simulator"""
    
    def __init__(self, sv_fp, dual_coef_fp, bias_fp, gamma_fp):
        super().__init__()
        self.support_vectors = np.array(sv_fp, dtype=np.int16)
        self.dual_coef = np.array(dual_coef_fp, dtype=np.int16)
        self.bias = np.int16(bias_fp)
        self.gamma = np.int16(gamma_fp)
        self.n_support = len(dual_coef_fp)
        self.n_features = self.support_vectors.shape[1]
        self.operation_count = 0
        
        # Build LUT for exp(-x) approximation
        self.build_exp_lut()
    
    def build_exp_lut(self, lut_size=256):
        """Build LUT for exp(-gamma * ||x||^2)"""
        # LUT input range: 0 to max_distance_sq
        # For normalized features, typical range is 0 to 16
        max_dist = 16.0
        
        self.lut = np.zeros(lut_size, dtype=np.int16)
        
        for i in range(lut_size):
            # Map index to distance squared
            dist_sq = (i / lut_size) * max_dist
            # Compute exp(-gamma * dist_sq)
            gamma_float = self.gamma / self.scale
            exp_val = np.exp(-gamma_float * dist_sq)
            # Quantize to Q8.8
            self.lut[i] = np.int16(np.clip(exp_val * self.scale, -32768, 32767))
    
    def compute_distance_sq(self, x, sv):
        """Compute ||x - sv||^2 in fixed-point"""
        dist_sq = 0
        ops = 0
        
        for i in range(self.n_features):
            diff = self.saturate(x[i] - sv[i])
            # Square the difference
            sq = self.fixed_mult(diff, diff)
            dist_sq = self.fixed_add(dist_sq, sq)
            ops += 3  # sub, mult, add
        
        return dist_sq, ops
    
    def exp_approximation(self, dist_sq):
        """LUT-based exp approximation"""
        # Map dist_sq to LUT index
        # Assume dist_sq is in Q8.8
        dist_float = dist_sq / self.scale
        max_dist = 16.0
        
        # Normalize to [0, 255]
        index = int(np.clip((dist_float / max_dist) * 255, 0, 255))
        
        return self.lut[index], 1  # 1 operation (LUT lookup)
    
    def predict_sample(self, x_fp):
        """Predict single sample"""
        self.operation_count = 0
        x = np.array(x_fp, dtype=np.int16)
        
        # Sum over support vectors
        acc = np.int32(0)
        
        for j in range(self.n_support):
            sv = self.support_vectors[j]
            
            # Compute kernel: exp(-gamma * ||x - sv||^2)
            dist_sq, ops1 = self.compute_distance_sq(x, sv)
            self.operation_count += ops1
            
            kernel_val, ops2 = self.exp_approximation(dist_sq)
            self.operation_count += ops2
            
            # Multiply by dual coefficient
            weighted = self.fixed_mult(self.dual_coef[j], kernel_val)
            self.operation_count += 1
            
            # Accumulate
            acc = self.fixed_add(acc, weighted)
            self.operation_count += 1
        
        # Add bias
        result = self.fixed_add(acc, self.bias)
        self.operation_count += 1
        
        # Sign bit determines class
        prediction = 1 if result >= 0 else 0
        
        return prediction, result
    
    def predict(self, X_fp):
        """Predict batch"""
        predictions = []
        decision_values = []
        total_ops = 0
        
        for sample in X_fp:
            pred, val = self.predict_sample(sample)
            predictions.append(pred)
            decision_values.append(val)
            total_ops += self.operation_count
        
        avg_ops = total_ops / len(X_fp)
        
        return np.array(predictions), np.array(decision_values), avg_ops

def quantize_input(X, fp_converter):
    """Quantize input features to Q8.8"""
    return fp_converter.quantize_array(X)

def main():
    # Load models
    with open('../results/svm_models.pkl', 'rb') as f:
        results = pickle.load(f)
    
    X_test = results['test_data']['X_test']
    y_test = results['test_data']['y_test']
    
    # Quantize test data
    from train_and_quantize import FixedPointConverter
    fp_converter = FixedPointConverter()
    X_test_fp = quantize_input(X_test, fp_converter)
    
    print("\n" + "="*60)
    print("FIXED-POINT SIMULATION")
    print("="*60)
    
    # Linear SVM simulation
    print("\nLinear SVM Fixed-Point Simulation:")
    linear_quant = results['linear']['quant_params']
    linear_sim = LinearSVMSimulator(
        linear_quant['weights_fp'],
        linear_quant['bias_fp']
    )
    
    y_pred_linear, decision_linear, ops_linear = linear_sim.predict(X_test_fp)
    acc_linear = accuracy_score(y_test, y_pred_linear)
    cm_linear = confusion_matrix(y_test, y_pred_linear)
    
    print(f"  Accuracy: {acc_linear:.4f}")
    print(f"  Avg operations per sample: {ops_linear:.1f}")
    print(f"  Confusion Matrix:\n{cm_linear}")
    
    # Compare with float model
    y_pred_float = results['linear']['model'].predict(X_test)
    acc_float = accuracy_score(y_test, y_pred_float)
    print(f"  Float accuracy: {acc_float:.4f}")
    print(f"  Accuracy degradation: {(acc_float - acc_linear)*100:.2f}%")
    
    # Kernel SVM simulation
    print("\nKernel SVM Fixed-Point Simulation:")
    kernel_quant = results['kernel']['quant_params']
    kernel_sim = KernelSVMSimulator(
        kernel_quant['sv_fp'],
        kernel_quant['dual_coef_fp'],
        kernel_quant['bias_fp'],
        kernel_quant['gamma_fp']
    )
    
    y_pred_kernel, decision_kernel, ops_kernel = kernel_sim.predict(X_test_fp)
    acc_kernel = accuracy_score(y_test, y_pred_kernel)
    cm_kernel = confusion_matrix(y_test, y_pred_kernel)
    
    print(f"  Accuracy: {acc_kernel:.4f}")
    print(f"  Avg operations per sample: {ops_kernel:.1f}")
    print(f"  Confusion Matrix:\n{cm_kernel}")
    
    # Compare with reduced SV model
    print(f"  Support vectors used: {kernel_quant['n_support']}")
    
    # Save simulation results
    sim_results = {
        'linear': {
            'accuracy_fp': acc_linear,
            'accuracy_float': acc_float,
            'operations': ops_linear,
            'confusion_matrix': cm_linear.tolist(),
            'predictions': y_pred_linear.tolist()
        },
        'kernel': {
            'accuracy_fp': acc_kernel,
            'operations': ops_kernel,
            'confusion_matrix': cm_kernel.tolist(),
            'predictions': y_pred_kernel.tolist()
        }
    }
    
    with open('../results/simulation_results.pkl', 'wb') as f:
        pickle.dump(sim_results, f)
    
    print("\n" + "="*60)
    print("Simulation complete. Results saved.")
    print("="*60)

if __name__ == "__main__":
    main()