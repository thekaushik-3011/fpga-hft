import json
import numpy as np
import time
from sklearn.metrics import accuracy_score

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def verify_linear_svm(linear_params, test_data):
    print("\n--- Verifying Linear SVM (Q8.8 Integer Arithmetic) ---")
    weights = np.array(linear_params['weights'], dtype=np.int32)
    bias = linear_params['bias'] # int
    
    X_test = np.array(test_data['X_test'], dtype=np.int32)
    y_test = np.array(test_data['y_test'], dtype=np.int32)
    
    start_time = time.time()
    
    # Integer Dot Product: (N_samples, N_features) . (N_features,) -> (N_samples,)
    # Scaling: X is Q8.8, W is Q8.8 -> Dot is Q16.16
    # We essentially divide by 256 to bring it back to Q8.8 range if we interpret 'bias' as Q8.8
    # But usually, we just keep accumulating.
    # Linear params JSON: bias is Q8.8 (value ~ -53, float was -0.2?). 
    # If weights are Q8.8 and Input is Q8.8, product is Q16.16.
    # Bias is Q8.8. To add bias, we need to shift bias or shift product.
    # The training notebook said: dot_prod_scaled = dot_prod // 256
    
    correct = 0
    total = len(y_test)
    
    preds = []
    
    # Vectorized implementation for speed
    dot_prod = np.dot(X_test, weights)
    dot_prod_scaled = np.floor(dot_prod / 256.0).astype(np.int32) # approximating bit shift >> 8
    decision_values = dot_prod_scaled + bias
    preds = (decision_values > 0).astype(np.int32)
    
    correct = np.sum(preds == y_test)
    accuracy = correct / total * 100
    
    elapsed = time.time() - start_time
    
    print(f"Linear SVM Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"  True Label dist: {dict(zip(unique, counts))}")
    unique_p, counts_p = np.unique(preds, return_counts=True)
    print(f"  Pred Label dist: {dict(zip(unique_p, counts_p))}")

def verify_kernel_svm(kernel_params, test_data, override_gamma=None):
    print("\n--- Verifying Kernel SVM (Simulated Fixed-Point Arithmetic) ---")
    
    # Load parameters as Integers
    gamma_q = override_gamma if override_gamma is not None else int(kernel_params['gamma'])
    bias_q = int(kernel_params['bias'])
    dual_coef_q = np.array(kernel_params['dual_coef'][0], dtype=np.int32) # Shape (N_sv,)

    support_vectors_q = np.array(kernel_params['support_vectors'], dtype=np.int32) # Shape (N_sv, N_feat)
    
    X_test_q = np.array(test_data['X_test'], dtype=np.int32)   # Shape (N_test, N_feat)
    y_test = np.array(test_data['y_test'], dtype=np.int32)
    
    N_test = X_test_q.shape[0]
    N_sv = support_vectors_q.shape[0]
    
    print(f"  Test samples: {N_test}")
    print(f"  Support Vectors: {N_sv}")
    print(f"  Gamma (Q8.8): {gamma_q}")
    print(f"  Bias (Q8.8): {bias_q}")
    
    start_time = time.time()
    
    # 1. Compute Squared Euclidean Distance in Integers
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    # Note: Inputs are Q8.8 (scaled by 256).
    # d^2 will be scaled by 256*256 = 65536.
    
    print("  Computing kernel matrix (integer operations)...")
    
    # x^2 (sum over features)
    X_sq = np.sum(X_test_q**2, axis=1).astype(np.int64) # Shape (N_test,)
    
    # y^2 (sum over features)
    SV_sq = np.sum(support_vectors_q**2, axis=1).astype(np.int64) # Shape (N_sv,)
    
    # 2*x*y
    # Result is roughly Q16.16 scaled
    dot_products = np.dot(X_test_q, support_vectors_q.T).astype(np.int64) # Shape (N_test, N_sv)
    
    # Broadcasting to get distance matrix
    # dist_sq[i, j] = X_sq[i] + SV_sq[j] - 2 * dot_products[i, j]
    dist_sq = X_sq[:, np.newaxis] + SV_sq[np.newaxis, :] - 2 * dot_products
    
    # 2. Compute Kernel value: exp(-gamma * ||x-y||^2)
    # gamma_q is Q8.8 (scaled by 256)
    # dist_sq is Q16.16 (scaled by 65536)
    # product is scaled by 256 * 65536 = 16,777,216 (2^24)
    
    # Argument for exp needs to be de-scaled to valid range for the function
    # true_arg = - (gamma_q * dist_sq) / (2^24)
    # We calculate high precision argument then effectively "lookup" quantized exp
    
    gamma_dist = -(gamma_q * dist_sq) # Shape (N_test, N_sv)
    
    # Simulate Fixed Point Exp
    # Using float exp for accuracy of the function itself, but inputs derived from integers
    # This matches a high-fidelity LUT on FPGA
    
    # Scale factor compensation: 
    # We want K(u,v) which is roughly 0..1 (or 0..256 in Q8.8)
    # real_val = exp( real_gamma * real_dist_sq )
    # real_gamma = gamma_q / 256.0
    # real_dist_sq = dist_sq / 65536.0
    
    exp_arg = gamma_dist / (256.0 * 65536.0)
    print(f"  Exp Arg stats: Min={exp_arg.min():.4f}, Max={exp_arg.max():.4f}, Mean={exp_arg.mean():.4f}")
    
    K_val = np.exp(exp_arg)
    print(f"  K_val stats: Min={K_val.min():.4f}, Max={K_val.max():.4f}, Mean={K_val.mean():.4f}")
    
    # Quantize Kernel result to Q8.8 (0..256)
    # This is what the FPGA creates after the exp module
    K_q = (K_val * 256.0).astype(np.int32)
    print(f"  K_q stats: Min={K_q.min()}, Max={K_q.max()}, Mean={K_q.mean()}")

    
    # 3. Decision Function
    # decision = sum(dual_coef * K) + bias
    # dual_coef is Q8.8 ? Actually checking the json..
    # from previous `view_file`: dual_coef values are around -25600.
    # -25600 / 256 = -100. This seems reasonable.
    # So dual_coef is Q8.8.
    # K_q is Q8.8.
    # Product is Q16.16.
    
    # We perform the dot product
    decision_sum = np.dot(K_q, dual_coef_q.T) # Shape (N_test,)
    print(f"  Decision Sum (Q16.16) stats: Min={decision_sum.min()}, Max={decision_sum.max()}, Mean={decision_sum.mean()}")
    
    # decision_sum is scaled by 256*256 = 65536
    # bias_q is scaled by 256
    # We need to align them.
    # Either shift decision_sum down by 8 bits (div 256) -> Q8.8
    # Or shift bias up by 8 bits.
    # Typically we output Q8.8 scores? 
    # Let's align to Q8.8 for final check.
    
    decision_scaled = np.floor(decision_sum / 256.0).astype(np.int32)
    print(f"  Decision Scaled (Q8.8) stats: Min={decision_scaled.min()}, Max={decision_scaled.max()}, Mean={decision_scaled.mean()}")
    
    final_scores = decision_scaled + bias_q
    print(f"  Final Scores (w/ Bias) stats: Min={final_scores.min()}, Max={final_scores.max()}, Mean={final_scores.mean()}")
    
    predictions = (final_scores > 0).astype(np.int32)
    
    correct = np.sum(predictions == y_test)
    accuracy = correct / N_test * 100
    
    print(f"Kernel SVM Accuracy: {accuracy:.2f}% ({correct}/{N_test})")
    
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"  True Label dist: {dict(zip(unique, counts))}")
    unique_p, counts_p = np.unique(predictions, return_counts=True)
    print(f"  Pred Label dist: {dict(zip(unique_p, counts_p))}")

if __name__ == "__main__":
    base_path = "results/"
    
    test_data = load_json(base_path + "test_data.json")
    
    try:
        linear_params = load_json(base_path + "linear_params.json")
        verify_linear_svm(linear_params, test_data)
    except FileNotFoundError:
        print("linear_params.json not found.")

    try:
        kernel_params = load_json(base_path + "kernel_params.json")
        verify_kernel_svm(kernel_params, test_data)
    except Exception as e:
        print(f"Error in Kernel SVM: {e}")
