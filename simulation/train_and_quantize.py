# simulation/train_and_quantize.py
"""
SVM Training and Fixed-Point Quantization
Trains both Linear and RBF Kernel SVM
Quantizes to Q8.8 format for hardware implementation
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import json

class FixedPointConverter:
    """Q8.8 Fixed-Point Converter"""
    
    def __init__(self, int_bits=8, frac_bits=8):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits
        self.scale = 2 ** frac_bits
        self.max_val = (2 ** (self.total_bits - 1) - 1) / self.scale
        self.min_val = -(2 ** (self.total_bits - 1)) / self.scale
        
    def float_to_fixed(self, value):
        """Convert float to Q8.8 fixed-point (16-bit signed)"""
        # Clip to representable range
        clipped = np.clip(value, self.min_val, self.max_val)
        # Scale and round
        fixed = np.round(clipped * self.scale).astype(np.int16)
        return fixed
    
    def fixed_to_float(self, fixed_val):
        """Convert Q8.8 back to float"""
        return fixed_val / self.scale
    
    def quantize_array(self, arr):
        """Quantize numpy array"""
        return self.float_to_fixed(arr)

class SVMTrainer:
    """Train and quantize SVM models"""
    
    def __init__(self, kernel='linear', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.fp_converter = FixedPointConverter()
        
    def train(self, X_train, y_train):
        """Train SVM model"""
        if self.kernel == 'linear':
            # Use LinearSVC for efficiency
            self.model = LinearSVC(C=self.C, max_iter=10000, dual=True)
        else:
            # Use kernel SVM
            self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        
        self.model.fit(X_train, y_train)
        print(f"\n{self.kernel.upper()} SVM Training Complete")
        
        return self.model
    
    def extract_parameters(self):
        """Extract model parameters for hardware implementation"""
        params = {}
        
        if self.kernel == 'linear':
            # Linear SVM: weights and bias
            params['weights'] = self.model.coef_[0]  # Shape: (n_features,)
            params['bias'] = self.model.intercept_[0]
            params['n_features'] = len(params['weights'])
            
        else:
            # Kernel SVM: support vectors, dual coefficients, bias
            params['support_vectors'] = self.model.support_vectors_
            params['dual_coef'] = self.model.dual_coef_[0]
            params['bias'] = self.model.intercept_[0]
            params['gamma'] = self.model._gamma if hasattr(self.model, '_gamma') else 1.0
            params['n_support'] = len(params['support_vectors'])
            params['n_features'] = self.model.support_vectors_.shape[1]
            
        return params
    
    def quantize_parameters(self, params):
        """Quantize parameters to Q8.8"""
        quant_params = {}
        
        if self.kernel == 'linear':
            quant_params['weights_fp'] = self.fp_converter.quantize_array(params['weights'])
            quant_params['bias_fp'] = self.fp_converter.float_to_fixed(params['bias'])
            quant_params['n_features'] = params['n_features']
            
            # Store float versions for comparison
            quant_params['weights_float'] = params['weights']
            quant_params['bias_float'] = params['bias']
            
        else:
            quant_params['sv_fp'] = self.fp_converter.quantize_array(params['support_vectors'])
            quant_params['dual_coef_fp'] = self.fp_converter.quantize_array(params['dual_coef'])
            quant_params['bias_fp'] = self.fp_converter.float_to_fixed(params['bias'])
            quant_params['gamma_fp'] = self.fp_converter.float_to_fixed(params['gamma'])
            quant_params['n_support'] = params['n_support']
            quant_params['n_features'] = params['n_features']
            
            # Store float versions
            quant_params['sv_float'] = params['support_vectors']
            quant_params['dual_coef_float'] = params['dual_coef']
            quant_params['bias_float'] = params['bias']
            quant_params['gamma_float'] = params['gamma']
            
        return quant_params
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\n{self.kernel.upper()} SVM Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        
        return metrics

from sklearn.preprocessing import StandardScaler

def main():
    # Load dataset
    X = np.load('../results/X_train.npy')
    y = np.load('../results/y_train.npy')
    
    # Split dataset
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale dataset (CRITICAL for Q8.8 fixed point)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Training set: {X_train.shape} (Scaled)")
    print(f"Test set: {X_test.shape} (Scaled)")
    print(f"Test Data Range: Min={X_test.min():.2f}, Max={X_test.max():.2f}")
    
    # Train Linear SVM
    print("\n" + "="*60)
    print("TRAINING LINEAR SVM")
    print("="*60)
    linear_trainer = SVMTrainer(kernel='linear', C=1.0)
    linear_model = linear_trainer.train(X_train, y_train)
    linear_metrics = linear_trainer.evaluate(X_test, y_test)
    
    # Extract and quantize parameters
    linear_params = linear_trainer.extract_parameters()
    linear_quant = linear_trainer.quantize_parameters(linear_params)
    
    print(f"\nLinear SVM Parameters:")
    print(f"  Weights shape: {linear_params['weights'].shape}")
    print(f"  Bias: {linear_params['bias']:.6f}")
    print(f"  Quantized bias: {linear_quant['bias_fp']} (0x{int(linear_quant['bias_fp']) & 0xFFFF:04X})")
    
    # Train Kernel SVM (RBF)
    print("\n" + "="*60)
    print("TRAINING RBF KERNEL SVM")
    print("="*60)
    kernel_trainer = SVMTrainer(kernel='rbf', C=10.0, gamma='scale')
    kernel_model = kernel_trainer.train(X_train, y_train)
    kernel_metrics = kernel_trainer.evaluate(X_test, y_test)
    
    # Extract and quantize parameters
    kernel_params = kernel_trainer.extract_parameters()
    
    # Limit support vectors to 8 for hardware
    max_sv = 8
    if kernel_params['n_support'] > max_sv:
        print(f"\nWarning: Model has {kernel_params['n_support']} support vectors")
        print(f"Selecting top {max_sv} by |dual_coef|")
        
        # Select top SVs by importance
        importance = np.abs(kernel_params['dual_coef'])
        top_indices = np.argsort(importance)[-max_sv:]
        
        kernel_params['support_vectors'] = kernel_params['support_vectors'][top_indices]
        kernel_params['dual_coef'] = kernel_params['dual_coef'][top_indices]
        kernel_params['n_support'] = max_sv
    
    kernel_quant = kernel_trainer.quantize_parameters(kernel_params)
    
    print(f"\nKernel SVM Parameters:")
    print(f"  Support Vectors: {kernel_params['n_support']}")
    print(f"  Gamma: {kernel_params['gamma']:.6f}")
    print(f"  Bias: {kernel_params['bias']:.6f}")
    
    # Save models and parameters
    results = {
        'linear': {
            'model': linear_model,
            'params': linear_params,
            'quant_params': linear_quant,
            'metrics': linear_metrics
        },
        'kernel': {
            'model': kernel_model,
            'params': kernel_params,
            'quant_params': kernel_quant,
            'metrics': kernel_metrics
        },
        'test_data': {
            'X_test': X_test,
            'y_test': y_test
        }
    }
    
    # Save to files
    with open('../results/svm_models.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save quantized parameters as JSON for RTL testbench
    json_params = {
        'linear': {
            'weights': linear_quant['weights_fp'].tolist(),
            'bias': int(linear_quant['bias_fp']),
            'n_features': int(linear_quant['n_features'])
        },
        'kernel': {
            'support_vectors': kernel_quant['sv_fp'].tolist(),
            'dual_coef': kernel_quant['dual_coef_fp'].tolist(),
            'bias': int(kernel_quant['bias_fp']),
            'gamma': int(kernel_quant['gamma_fp']),
            'n_support': int(kernel_quant['n_support']),
            'n_features': int(kernel_quant['n_features'])
        }
    }
    
    with open('../results/quantized_params.json', 'w') as f:
        json.dump(json_params, f, indent=2)
    
    print("\n" + "="*60)
    print("Models and parameters saved successfully")
    print("="*60)

if __name__ == "__main__":
    main()