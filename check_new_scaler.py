
import joblib
import os
import numpy as np

def check_new_scaler():
    # results_final_rank1/S003/Fold_1_S003_T002_SDA_frac1.0/scaler.pkl
    path = 'results_final_rank1/S003/Fold_1_S003_T002_SDA_frac1.0/scaler.pkl'
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            print(f"New Scaler Mean (Index 6): {scaler.mean_[6]}")
            if abs(scaler.mean_[6]) < 0.1:
                print("VERIFIED: Scaler Mean is correct (~0.0)")
            elif abs(scaler.mean_[6] - 3.33) < 0.1:
                print("FAILURE: Scaler Mean is still 3.33")
            else:
                print(f"Unknown Mean: {scaler.mean_[6]}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    else:
        print("Scaler not found yet.")

if __name__ == "__main__":
    check_new_scaler()
