
import joblib
import os
import numpy as np

def check_other_corruption():
    # Check S007 Scaler from CORRUPTED folder
    path = 'results_final_rank1_corrupted/S007/Fold_1_S007_T005_SDA_frac1.0/scaler.pkl'
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            print(f"S007 Corrupted Scaler Mean (Index 6): {scaler.mean_[6]}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("S007 Corrupted path not found.")

if __name__ == "__main__":
    check_other_corruption()
