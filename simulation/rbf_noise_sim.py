
import numpy as np

def analyze_rbf_noise():
    print("--- RBF Quantization Noise Simulation ---")
    
    # Constants
    gamma = 0.0625
    n_features = 16
    
    # Q8.8 noise model
    # Uniform noise in [-0.5/256, 0.5/256] -> approx std dev = 1/(256 * sqrt(12))
    q_step = 1/256.0
    noise_std = q_step / np.sqrt(12)
    
    # Theoretical Error Propagation
    # DistSq = sum((x - sv)^2)
    # Error in (x-sv) is approx sqrt(2) * noise_std (since both quantized)
    
    # Let's simulate
    trials = 10000
    accum_errors = []
    
    for _ in range(trials):
        # Random feature vector (normalized ~ N(0,1))
        x = np.random.randn(n_features)
        sv = np.random.randn(n_features)
        
        # True Quantities
        diff = x - sv
        dist_sq_true = np.sum(diff**2)
        exp_true = np.exp(-gamma * dist_sq_true)
        
        # Quantized Quantities (Q8.8)
        x_q = np.round(x * 256) / 256
        sv_q = np.round(sv * 256) / 256
        
        diff_q = x_q - sv_q
        # Squared Diff (Q8.8 * Q8.8 = Q16.16 -> shifted back to Q8.8)
        # In python simulation we just did float math on quantized vals
        # But fixed_point_sim used: (a*b) >> 8
        
        dist_sq_q = 0
        for k in range(n_features):
            # Model the fixed point multiply truncation
            d_val = diff_q[k]
            # Convert to int representation for bit-accurate mult
            d_int = int(round(d_val * 256))
            sq_int = (d_int * d_int) >> 8 # Q8.8 result
            dist_sq_q += sq_int / 256.0
            
        exp_q = np.exp(-gamma * dist_sq_q)
        
        # Error
        accum_errors.append(exp_q - exp_true)
        
    accum_errors = np.array(accum_errors)
    print(f"Mean Kernel Output Error: {np.mean(accum_errors):.6f}")
    print(f"Std Dev Kernel Output Error: {np.std(accum_errors):.6f}")
    print(f"Max Kernel Output Error: {np.max(np.abs(accum_errors)):.6f}")

if __name__ == "__main__":
    analyze_rbf_noise()
