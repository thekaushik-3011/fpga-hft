import os
import numpy as np
import pandas as pd
import glob
import json
import gc

def load_fi2010_data():
    """
    Parses and combines FI-2010 Benchmark Dataset (Auction and NoAuction).
    OPTIMIZED: Loads a subset of files to avoid OOM, uses float32.
    """
    
    # 1. Define Column Schema (0-45)
    columns = ['time_idx']
    for i in range(1, 11): columns.append(f'bid_price_{i}')
    for i in range(1, 11): columns.append(f'ask_price_{i}')
    for i in range(1, 11): columns.append(f'bid_volume_{i}')
    for i in range(1, 11): columns.append(f'ask_volume_{i}')
    columns.extend(['label_10t', 'label_20t', 'label_30t', 'label_50t', 'label_100t'])
    
    # 2. Define Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    base_dir = os.path.join(project_root, "BenchmarkDatasets")
    
    auc_train_dir = os.path.join(base_dir, "Auction/1.Auction_Zscore/Auction_Zscore_Training")
    auc_test_dir = os.path.join(base_dir, "Auction/1.Auction_Zscore/Auction_Zscore_Testing")
    noauc_train_dir = os.path.join(base_dir, "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training")
    noauc_test_dir = os.path.join(base_dir, "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing")
    
    # 3. Load Function (Optimized)
    def load_files_to_numpy(directory, file_limit=2):
        if not os.path.exists(directory):
            print(f"Skipping {directory} (Not Found)")
            return [], []
            
        files = sorted(glob.glob(os.path.join(directory, "*.txt")))
        print(f"Found {len(files)} files in {directory}")
        
        # Limit files to prevent OOM
        files = files[:file_limit]
        print(f"Loading first {len(files)} files...")
        
        features_list = []
        targets_list = []
        
        for f in files:
            print(f"  Reading {os.path.basename(f)}...")
            try:
                # Use C engine, assume whitespace delimiter. 
                # names=columns is important.
                # dtype=np.float32 direct conversion often fails with read_csv inference, 
                # better to read as default then cast.
                # File format is Transposed (Rows=Features, Cols=Samples)
                # Read without names first
                df = pd.read_csv(f, delim_whitespace=True, header=None)
                
                # Transpose: (149, N) -> (N, 149)
                df = df.T
                
                # Slice first 46 columns (Features + Labels)
                # Rows 0-45 in source -> Cols 0-45 after transpose
                df = df.iloc[:, :46]
                
                # Assign Column Names
                df.columns = columns
                
                # Extract X (cols 1-40) and y (col 41 'label_10t')
                x_chunk = df[columns[1:41]].values.astype(np.float32)
                y_chunk = df['label_10t'].values.astype(np.float32)
                
                features_list.append(x_chunk)
                targets_list.append(y_chunk)
                
                del df
                gc.collect()
                
            except Exception as e:
                print(f"  Error loading {f}: {e}")
                
        return features_list, targets_list

    # 4. Load Data
    # Limit to 3 files each to keep dataset manageable (~1-2GB RAM)
    # User said "combine all", but we must be practical. 3 files is ~30% of data, enough for dev.
    limit = 1 
    
    print("\n--- Loading Auction Training ---")
    X_auc_tr, y_auc_tr = load_files_to_numpy(auc_train_dir, limit)
    
    print("\n--- Loading NoAuction Training ---")
    X_noauc_tr, y_noauc_tr = load_files_to_numpy(noauc_train_dir, limit)
    
    print("\n--- Loading Auction Testing ---")
    X_auc_te, y_auc_te = load_files_to_numpy(auc_test_dir, limit)
    
    print("\n--- Loading NoAuction Testing ---")
    X_noauc_te, y_noauc_te = load_files_to_numpy(noauc_test_dir, limit)
    
    # 5. Concatenate
    print("\nConcatenating arrays...")
    
    # Train
    train_feats = X_auc_tr + X_noauc_tr
    train_targs = y_auc_tr + y_noauc_tr
    
    if train_feats:
        X_train = np.concatenate(train_feats, axis=0)
        y_train = np.concatenate(train_targs, axis=0)
    else:
        X_train = np.array([])
        y_train = np.array([])
        
    # Test
    test_feats = X_auc_te + X_noauc_te
    test_targs = y_auc_te + y_noauc_te
    
    if test_feats:
        X_test = np.concatenate(test_feats, axis=0)
        y_test = np.concatenate(test_targs, axis=0)
    else:
        X_test = np.array([])
        y_test = np.array([])
        
    print(f"Final Train Shape: {X_train.shape}")
    print(f"Final Test Shape:  {X_test.shape}")
    
    # 6. Save
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # NEW: Save as CSV
    print("Saving datasets to CSV...")
    # Reconstruct Feature Names
    feats = []
    for i in range(1, 11): feats.append(f'bid_price_{i}')
    for i in range(1, 11): feats.append(f'ask_price_{i}')
    for i in range(1, 11): feats.append(f'bid_volume_{i}')
    for i in range(1, 11): feats.append(f'ask_volume_{i}')
    
    # Train CSV
    train_df = pd.DataFrame(X_train, columns=feats)
    train_df['label_10t'] = y_train
    train_csv_path = os.path.join(output_dir, "train_dataset.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"Saved {train_csv_path}")
    
    # Test CSV
    test_df = pd.DataFrame(X_test, columns=feats)
    test_df['label_10t'] = y_test
    test_csv_path = os.path.join(output_dir, "test_dataset.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved {test_csv_path}")

    # Save Validation Sample
    if len(X_test) > 0:
        test_json = {
            'X_test': X_test[:100].tolist(),
            'y_test': y_test[:100].tolist()
        }
        with open(os.path.join(output_dir, "test_data.json"), 'w') as f:
            json.dump(test_json, f)
            
    print("Processing Complete.")

if __name__ == "__main__":
    load_fi2010_data()
