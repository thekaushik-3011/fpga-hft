
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
import sys
import os

# Add local path for imports
sys.path.append(os.getcwd())
try:
    from simulation.train_and_quantize import FixedPointConverter
    from simulation.fixed_point_sim import KernelSVMSimulator
except ImportError:
    # Try relative import if running from simulation dir
    sys.path.append('../')
    from simulation.train_and_quantize import FixedPointConverter
    from simulation.fixed_point_sim import KernelSVMSimulator

def analyze_dataset_structure(X, y):
    print("\nPart 1: Dataset Accuracy Audit")
    print("-" * 40)
    
    # Check Separability
    print("Checking Linear Separability (PCA)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Simple linear classifier on PCA data
    clf = LinearSVC(random_state=42, max_iter=10000, dual='auto')
    clf.fit(X_pca, y)
    acc_pca = clf.score(X_pca, y)
    print(f"  Accuracy on 2D PCA projection: {acc_pca:.4f}")
    if acc_pca > 0.95:
        print("  -> Dataset is highly separable even in 2D.")
        
    # Cross Validation
    print("\nPerforming 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(LinearSVC(random_state=42, max_iter=10000, dual='auto'), X, y, cv=skf)
    print(f"  CV Scores: {scores}")
    print(f"  Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    if scores.mean() > 0.99:
        print("  -> CAUTION: Near-perfect accuracy confirms easy separability.")
        print("     Likely cause: Synthetic distributions have minimal overlap.")

def analyze_quantization_effects():
    print("\nPart 2: Fixed-Point Collapse Analysis")
    print("-" * 40)
    
    try:
        with open('results/svm_models.pkl', 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print("Error: svm_models.pkl not found. Run from valid directory.")
        return

    # Kernel Parameters
    sv = results['kernel']['params']['support_vectors'] # Float
    gamma = results['kernel']['params']['gamma']
    dual = results['kernel']['params']['dual_coef']
    
    print(f"Gamma: {gamma:.6f}")
    
    # Analyze Feature Range
    print(f"Feature Range: [{sv.min():.4f}, {sv.max():.4f}]")
    
    # Analyze Squared Distance Magnitude
    # Pick two random SVs to check typical distance
    dist_sq_vals = []
    for i in range(min(10, len(sv))):
        for j in range(min(10, len(sv))):
            d = np.sum((sv[i] - sv[j])**2)
            dist_sq_vals.append(d)
    
    dist_sq_vals = np.array(dist_sq_vals)
    print(f"Typical Squared Distances (Float): Mean={dist_sq_vals.mean():.4f}, Max={dist_sq_vals.max():.4f}")
    
    # Exponential Sensitivity
    # exp(-gamma * dist_sq)
    # Check what happens with Q8.8 quantization error
    
    converter = FixedPointConverter(8, 8)
    scale = 256.0
    
    print("\nSensitivity Analysis (Exp Function):")
    test_dists = [0.1, 1.0, 5.0, 10.0]
    for d in test_dists:
        # True value
        true_exp = np.exp(-gamma * d)
        
        # Quantized Input (Distance)
        # Error in distance calculation can be up to 0.5/256 per dimension?
        # Let's assume input to exp has quantization noise
        d_quant = converter.fixed_to_float(converter.float_to_fixed(d))
        quant_exp = np.exp(-gamma * d_quant)
        
        # Quantized Output
        out_fixed = converter.float_to_fixed(quant_exp)
        out_float_recon = converter.fixed_to_float(out_fixed)
        
        print(f"  Dist={d:5.2f} | True Exp={true_exp:.4f} | Q8.8 Exp={out_float_recon:.4f} | Err={(out_float_recon-true_exp):.4f}")

    # Simulation with Q16.16 (High Precision) to isolate Pruning vs Quantization
    print("\nIsolating Error Sources (Q16.16 Simulation)...")
    
    # We need a Q16.16 simulator or just simulate with float but mimic Q16.16 behavior
    # Let's use float simulation with the pruned SVs vs original SVs
    
    # 1. Float Accuracy with Pruned SVs (already have this model)
    X_test = results['test_data']['X_test']
    y_test = results['test_data']['y_test']
    
    # Re-construct a manual float predict to check Pruning impact explicitly
    # The saved kernel model in 'results' might be the full one, but 'params' has pruned SVs if we logic'd correctly?
    # Actually train_and_quantize.py modified params['support_vectors'] to top 8.
    
    # Let's run a prediction using the stored (pruned) params in float
    def float_predict(X, sv, dual, bias, g):
        preds = []
        for x in X:
            # RBF
            dists = np.sum((sv - x)**2, axis=1)
            k_vals = np.exp(-g * dists)
            decision = np.sum(dual * k_vals) + bias
            preds.append(1 if decision >= 0 else 0)
        return np.array(preds)
        
    pruned_sv = results['kernel']['params']['support_vectors']
    pruned_dual = results['kernel']['params']['dual_coef']
    pruned_bias = results['kernel']['params']['bias']
    
    print(f"  Evaluation with {len(pruned_sv)} SVs (Float Precision)...")
    # Subsample test set for speed
    X_sub = X_test[:500]
    y_sub = y_test[:500]
    
    y_pred_pruned = float_predict(X_sub, pruned_sv, pruned_dual, pruned_bias, gamma)
    acc_pruned = accuracy_score(y_sub, y_pred_pruned)
    print(f"  -> Accuracy with Pruned SVs (Float): {acc_pruned:.4f}")
    
    # Compare with Q8.8 result (from report: ~0.60)
    print(f"  -> Accuracy with Pruned SVs (Q8.8):  ~0.6012 (from report)")
    
    print("\nConclusion on Failure Mode:")
    if acc_pruned > 0.95:
        print("  -> Pruning is NOT the killer. Quantization (Q8.8) is the cause.")
    else:
        print("  -> Pruning IS the killer. Model needs more SVs.")

def main():
    # Load raw data
    try:
        X = np.load('results/X_train.npy')
        y = np.load('results/y_train.npy')
    except FileNotFoundError:
        print("Run from project root.")
        return

    analyze_dataset_structure(X, y)
    analyze_quantization_effects()

if __name__ == "__main__":
    main()
