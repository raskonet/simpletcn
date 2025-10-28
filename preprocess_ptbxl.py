import pandas as pd
import numpy as np
import wfdb
import os
import matplotlib.pyplot as plt

# CONFIGURATION
BASE_PATH = 'physionet.org/files/ptb-xl/1.0.3/' 
if not os.path.exists(BASE_PATH): BASE_PATH = 'physionet.org/'

# 200k points ~ 33 minutes of data at 100Hz
TOTAL_POINTS = 200000 

def process_data():
    print(f"--- Data Engineering Pipeline ---")
    
    # 1. Load Database
    try:
        df = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    except FileNotFoundError:
        print(f"Error: Could not find ptbxl_database.csv at {BASE_PATH}")
        return

    # Filter for NORM (Healthy Controls) to learn standard sinus rhythm first
    norm_df = df[df.scp_codes.str.contains('NORM')]
    print(f"Found {len(norm_df)} normal records.")

    all_signals = []
    
    print(f"Extracting {TOTAL_POINTS} data points...")
    for ecg_id, row in norm_df.iterrows():
        if len(all_signals) * 1000 >= TOTAL_POINTS: break
        
        rel_path = row['filename_lr']
        full_path = os.path.join(BASE_PATH, rel_path)
        
        try:
            # Read Lead II (Channel 1)
            signals, fields = wfdb.rdsamp(full_path, channels=[1])
            sig = signals.flatten()
            all_signals.append(sig)
        except:
            continue

    raw_data = np.concatenate(all_signals)
    raw_data = raw_data[:TOTAL_POINTS]
    
    # 2. Split Data indices
    n = len(raw_data)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # n_test is the remainder
    
    train_raw = raw_data[:n_train]
    val_raw = raw_data[n_train:n_train+n_val]
    test_raw = raw_data[n_train+n_val:]
    
    # 3. Normalization (CRITICAL STEP)
    # We fit the scaler ONLY on Training data, then apply to Val/Test.
    # This prevents "Data Leakage".
    mean = np.mean(train_raw)
    std = np.std(train_raw)
    
    train_norm = (train_raw - mean) / std
    val_norm = (val_raw - mean) / std
    test_norm = (test_raw - mean) / std
    
    print(f"\n[Split Stats]")
    print(f"Train: {len(train_norm)} points")
    print(f"Val:   {len(val_norm)} points")
    print(f"Test:  {len(test_norm)} points")
    print(f"Mean (Train): {mean:.4f}, Std (Train): {std:.4f}")

    # 4. Save Files
    np.savetxt("ecg_train.txt", train_norm, fmt='%.6f')
    np.savetxt("ecg_val.txt", val_norm, fmt='%.6f')
    np.savetxt("ecg_test.txt", test_norm, fmt='%.6f')
    print(f"\nFiles saved: ecg_train.txt, ecg_val.txt, ecg_test.txt")

    # 5. Visualization of the Split
    plt.figure(figsize=(15, 5))
    plt.plot(range(0, 1000), train_norm[:1000], label='Train (First 1k)', color='blue', alpha=0.7)
    plt.title("Training Data Sample (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("data_split_preview.png")

if __name__ == "__main__":
    process_data()
