
import os
import torch
import numpy as np
import joblib
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials

def debug_s003():
    subj = 'S003'
    trial = 'S003_T002'
    
    # Path
    # results_final_rank1/S003/Fold_1_S003_T002_SDA_frac1.0
    dir1 = f'results_final_rank1/S003/Fold_1_{trial}_SDA_frac1.0'
    model_path = os.path.join(dir1, 'best_model.pth')
    scaler_path = os.path.join(dir1, 'scaler.pkl')
    
    print(f"Debug S003 Rank 1")
    print(f"Dir: {dir1}")
    
    # Config
    conf = CONFIG.copy()
    conf['target_subject'] = subj
    conf['feature_set'] = 'theta'
    conf['results_dir'] = 'results_final_rank1'
    # Rank 1 Params
    conf['encoder_layers'] = [128, 64]
    conf['decoder_layers'] = [64]

    # Check what 'theta' stats are
    conf_th = conf.copy(); conf_th['feature_set'] = 'theta'
    X_th, Y_raw = load_target_data_by_trials(conf_th, [trial])
    print(f"S003 Theta Mean: {X_th.mean(axis=(0,1))}")
    
    # Check what 'hip' stats are
    conf_hip = conf.copy(); conf_hip['feature_set'] = 'hip' # Default fallback
    X_hip, _ = load_target_data_by_trials(conf_hip, [trial])
    print(f"S003 Hip Mean: {X_hip.mean(axis=(0,1))}")
    
    # Load Source 'theta' stats (Global Scaler expectation)
    # We can't easily recalculate source stats without loading big file.
    # But we can check    # Load Scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler Mean (Source Theta): {scaler.mean_}")
    
    # Load Source 'Hip' Stats?
    # We need to call load_source_data with feature_set='hip'.
    # This might take time (loading large H5). 
    # Let's just try loading ONE subject from Source H5 if possible? 
    # load_source_data loads ALL.
    # We'll use a trick: Config with 'source_h5' pointing to same file, but just load (it's fast enough 1-2GB).
    
    print("Loading Source Hip Stats (Estimating)...")
    conf_src_hip = conf.copy(); conf_src_hip['feature_set'] = 'hip'
    # To save time, we can patch `load_source_data` to break after 100 samples?
    # Or just run it. It took ~30s in log.
    try:
        src_h, _ = load_source_data(conf_src_hip)
        print(f"Source Hip Mean: {src_h.mean(axis=(0,1))}")
    except Exception as e:
        print(f"Failed to load source hip: {e}")

    # Check Scalar Variance
    print(f"Scaler Var: {scaler.var_}")

    # Re-verify RMSE calculation
    # Only test Theta since model was trained on Rank 1 (Theta)
    N, T, F = X_th.shape
    X_acc = scaler.transform(X_th.reshape(-1, F)).reshape(N, T, F)
    Xt = torch.tensor(X_acc, dtype=torch.float32)
    
    model = SDA_Dual_Model(conf)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        pred, _ = model(Xt, domain='target')

    # Manual RMSE
    phi_pred = np.arctan2(pred.numpy()[:, 1], pred.numpy()[:, 0])
    phi_gt = np.arctan2(Y_raw[:, 1], Y_raw[:, 0]) # Y_raw is same for both
    diff = phi_pred - phi_gt
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    diff_pct = diff * (100.0 / (2 * np.pi))
    rmse = np.sqrt(np.mean(diff_pct**2))
    print(f"RMSE (Theta Input): {rmse:.4f}%")
    
    # Test Hip Input just in case model is weird
    X_acc_h = scaler.transform(X_hip.reshape(-1, F)).reshape(N, T, F)
    Xt_h = torch.tensor(X_acc_h, dtype=torch.float32)
    with torch.no_grad():
        pred_h, _ = model(Xt_h, domain='target')
    phi_pred_h = np.arctan2(pred_h.numpy()[:, 1], pred_h.numpy()[:, 0])
    diff_h = phi_pred_h - phi_gt
    diff_h = (diff_h + np.pi) % (2 * np.pi) - np.pi
    diff_pct_h = diff_h * (100.0 / (2 * np.pi))
    rmse_h = np.sqrt(np.mean(diff_pct_h**2))
    print(f"RMSE (Hip Input): {rmse_h:.4f}%")

if __name__ == "__main__":
    debug_s003()
