
import os
import joblib
import numpy as np
from sda_training import CONFIG, load_source_data

def debug_contradiction():
    # 1. Load Scaler from S003 Result
    # S003/Fold_1_S003_T002_SDA_frac1.0/scaler.pkl
    # We established its mean[6] is 3.33. Let's re-verify.
    path = 'results_final_rank1/S003/Fold_1_S003_T002_SDA_frac1.0/scaler.pkl'
    if os.path.exists(path):
        scaler = joblib.load(path)
        print(f"Scaler Mean (Index 6): {scaler.mean_[6]}")
        print(f"Scaler Mean (All): {scaler.mean_}")
    else:
        print("Scaler file not found.")

    # 2. Load Source Data directly
    print("\nLoading Source Data via sda_training...")
    conf = CONFIG.copy()
    conf['feature_set'] = 'theta'
    try:
        data, _ = load_source_data(conf)
        mean_val = data[:, :, 6].mean()
        print(f"Source Data Mean (Index 6): {mean_val}")
    except Exception as e:
        print(f"Error loading source: {e}")

if __name__ == "__main__":
    debug_contradiction()
