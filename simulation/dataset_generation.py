# simulation/dataset_generation.py
"""
HFT-Inspired Synthetic Dataset Generator
Simulates market microstructure features for binary classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

np.random.seed(42)

class HFTDatasetGenerator:
    """
    Generates synthetic high-frequency trading features
    Features:
    0-3: Price deltas (last 4 ticks)
    4-7: Volume imbalance ratios
    8-11: Bid-ask spread features
    12-15: Momentum indicators
    
    Classes:
    0: No significant price movement (hold/sell)
    1: Significant upward movement (buy signal)
    """
    
    def __init__(self, n_samples=12000, feature_dim=16):
        self.n_samples = n_samples
        self.feature_dim = feature_dim
        
    def generate(self):
        # Class 0: Market noise/consolidation
        n_class0 = self.n_samples // 2
        
        # Price deltas - small random walk
        price_deltas_0 = np.random.randn(n_class0, 4) * 0.3
        
        # Volume imbalance - near balanced
        vol_imbalance_0 = np.random.randn(n_class0, 4) * 0.2 + 0.1
        
        # Spread - moderate
        spread_0 = np.random.gamma(2, 0.5, (n_class0, 4))
        
        # Momentum - weak
        momentum_0 = np.random.randn(n_class0, 4) * 0.25
        
        X_class0 = np.hstack([price_deltas_0, vol_imbalance_0, 
                              spread_0, momentum_0])
        y_class0 = np.zeros(n_class0)
        
        # Class 1: Strong upward movement
        n_class1 = self.n_samples - n_class0
        
        # Price deltas - almost random (very noisy signal)
        price_deltas_1 = np.random.randn(n_class1, 4) * 0.5 + 0.05
        
        # Volume imbalance - slight bias (the "alpha")
        vol_imbalance_1 = np.random.randn(n_class1, 4) * 0.5 + 0.15
        
        # Spread - noisy gamma
        spread_1 = np.random.gamma(2, 0.5, (n_class1, 4)) # Same as class 0 essentially
        
        # Momentum - lagging indicator, weak signal
        momentum_1 = np.random.randn(n_class1, 4) * 0.5 + 0.1
        
        X_class1 = np.hstack([price_deltas_1, vol_imbalance_1,
                              spread_1, momentum_1])
        y_class1 = np.ones(n_class1)
        
        # Combine and shuffle
        X = np.vstack([X_class0, X_class1])
        y = np.hstack([y_class0, y_class1])
        
        # Shuffle
        indices = np.random.permutation(self.n_samples)
        X = X[indices]
        y = y[indices]
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    
    def visualize(self, X, y):
        """Generate visualization of dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Feature correlation heatmap
        df = pd.DataFrame(X, columns=[f'F{i}' for i in range(16)])
        df['Class'] = y
        
        corr = df.iloc[:, :16].corr()
        sns.heatmap(corr, ax=axes[0, 0], cmap='coolwarm', center=0)
        axes[0, 0].set_title('Feature Correlation Matrix')
        
        # Class distribution
        axes[0, 1].hist(y, bins=2, edgecolor='black')
        axes[0, 1].set_title('Class Distribution')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        
        # Feature importance via variance
        feature_names = ['P_Δ0', 'P_Δ1', 'P_Δ2', 'P_Δ3',
                        'V_Imb0', 'V_Imb1', 'V_Imb2', 'V_Imb3',
                        'Spread0', 'Spread1', 'Spread2', 'Spread3',
                        'Mom0', 'Mom1', 'Mom2', 'Mom3']
        
        variances = np.var(X, axis=0)
        axes[1, 0].bar(range(16), variances)
        axes[1, 0].set_xticks(range(16))
        axes[1, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[1, 0].set_title('Feature Variance')
        axes[1, 0].set_ylabel('Variance')
        
        # 2D projection (PCA)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        for class_val in [0, 1]:
            mask = y == class_val
            axes[1, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             label=f'Class {int(class_val)}', alpha=0.6)
        
        axes[1, 1].set_title('PCA Projection (2D)')
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    generator = HFTDatasetGenerator(n_samples=12000)
    X, y, scaler = generator.generate()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(X, axis=0)[:4]}...")
    print(f"  Std:  {np.std(X, axis=0)[:4]}...")
    
    # Save dataset
    np.save('../results/X_train.npy', X)
    np.save('../results/y_train.npy', y)
    
    # Visualize
    generator.visualize(X, y)