
import os
import torch
import numpy as np
import joblib
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials

# Override for Rank 3
RANK3_CONFIG = CONFIG.copy()
RANK3_CONFIG['encoder_layers'] = [256, 128]
RANK3_CONFIG['decoder_layers'] = [64]
RANK3_CONFIG['dropout'] = 0.3
RANK3_CONFIG['results_dir'] = 'results_final_rank3'

def calculate_rmse_pct(pred, gt):
    phi_pred = np.arctan2(pred[:, 1], pred[:, 0])
    phi_gt = np.arctan2(gt[:, 1], gt[:, 0])
    diff = phi_pred - phi_gt
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    diff_pct = diff * (100.0 / (2 * np.pi))
    return np.sqrt(np.mean(diff_pct**2))

def check_s007():
    subj = 'S007'
    # Need to find trial for S007. Usually T005 based on logs?
    # Let's verify existing file in rank1
    # results_final_rank1/S007/Fold_1_S007_T005_SDA_frac1.0
    trial = 'S007_T005'
    
    # Check Rank 3
    dir3 = f'results_final_rank3/S007/Fold_1_{trial}_SDA_frac1.0'
    model3 = os.path.join(dir3, 'best_model.pth')
    if not os.path.exists(model3):
        print("Rank 3 S007 model not found.")
        return

    # Load Data (using current logic, assuming consistent normalization method?)
    # Rank 3 used 'fixed' normalization for SDA.
    scaler3 = os.path.join(dir3, 'scaler.pkl')
    
    rank3_conf = RANK3_CONFIG.copy()
    rank3_conf['target_subject'] = subj
    # Need to set load_target_data config? 
    # It uses 'target_h5' and 'stride'. Stride 5.
    
    # Load Scaler
    scaler = joblib.load(scaler3)
    X_raw, Y_raw = load_target_data_by_trials(rank3_conf, [trial])
    N, T, F = X_raw.shape
    X_norm = scaler.transform(X_raw.reshape(-1, F)).reshape(N, T, F)
    Xt = torch.tensor(X_norm, dtype=torch.float32)
    
    # Model 3
    m3 = SDA_Dual_Model(rank3_conf)
    m3.load_state_dict(torch.load(model3, map_location='cpu'))
    m3.eval()
    
    with torch.no_grad():
        p3, _ = m3(Xt, domain='target')
        rmse3 = calculate_rmse_pct(p3.numpy(), Y_raw)
        
    print(f"S007 Rank 3 SDA RMSE: {rmse3:.2f}%")
    
    # Rank 1 Check (just to confirm my previous finding)
    dir1 = f'results_final_rank1/S007/Fold_1_{trial}_SDA_frac1.0'
    with open(os.path.join(dir1, 'final_rmse_pct.txt')) as f:
        rmse1 = float(f.read().strip())
    print(f"S007 Rank 1 SDA RMSE: {rmse1:.2f}%")

if __name__ == "__main__":
    check_s007()
