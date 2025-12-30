
import os
import torch
import numpy as np
import joblib
from sda_training import CONFIG, load_target_data_by_trials, SDA_Dual_Model, GaitDataset
# Import normalize_fixed/independent if needed, but we rely on scaler.pkl

def verify():
    # Setup Config for S003 SDA
    subj = 'S003'
    mode = 'SDA'
    
    # Path to the suspicion
    # Fold_1_S003_T002_SDA_frac1.0
    res_dir = 'results_final_rank3/S003/Fold_1_S003_T002_SDA_frac1.0'
    model_path = os.path.join(res_dir, 'best_model.pth')
    scaler_path = os.path.join(res_dir, 'scaler.pkl')
    
    print(f"Verifying Model: {model_path}")
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    
    # Load Data (S003_T002)
    # We must set target_subject for load_target_data_by_trials to find correct file?
    # No, it uses CONFIG['target_h5'] and keys.
    # But keys must be passed.
    test_trial = 'S003_T002'
    
    # Update global config just in case
    CONFIG['target_subject'] = subj
    # CONFIG['feature_set'] = 'theta' (Default)
    
    raw_x, raw_y = load_target_data_by_trials(CONFIG, [test_trial])
    print(f"Loaded Raw X: {raw_x.shape}")
    
    # Normalize
    # We simulate 'test' normalization: transform using loaded scaler
    # src_norm logic in training:
    # tgt_test_norm = scaler.transform(tgt_test.reshape(-1, F)).reshape(...)
    N, T, F = raw_x.shape
    x_norm = scaler.transform(raw_x.reshape(-1, F)).reshape(N, T, F)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(raw_y, dtype=torch.float32)
    
    # Load Model
    model = SDA_Dual_Model(CONFIG)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Eval
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        pred, _ = model(x_tensor, domain='target')
        loss = criterion(pred, y_tensor).item()
        
    print(f"Verified MSE for S003 SDA: {loss:.4f}")

if __name__ == "__main__":
    verify()
