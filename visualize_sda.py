
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import joblib
import os
from sda_training import SDA_Dual_Model, CONFIG

def load_trial_data_and_normalize(h5_file, mode, scaler_path, key_idx=0):
    # Load Scaler
    scaler = joblib.load(scaler_path)
    
    with h5py.File(h5_file, 'r') as f:
        keys = list(f.keys())
        key = keys[min(key_idx, len(keys)-1)]
        print(f"Loading trial: {key} from {mode}")
        grp = f[key]
        
        # Determine prefix
        if mode == 'source':
            pfx_acc = 'thigh'
            pfx_angle = '' # hip_angle is top level
        else:
            pfx = 'paretic'
        
        try:
            if mode == 'source':
                acc_x = grp['thigh_acc_x'][:][0]
                acc_y = grp['thigh_acc_y'][:][0]
                acc_z = grp['thigh_acc_z'][:][0]
                gyr_x = grp['thigh_gyr_x'][:][0]
                gyr_y = grp['thigh_gyr_y'][:][0]
                gyr_z = grp['thigh_gyr_z'][:][0]
                angle = grp['hip_angle'][:][0]
                angleV = grp['hip_angleV'][:][0]
                
                label_raw = grp['gcR_hs'][:][0]
                phase = label_raw * 0.01 * 2 * np.pi 
            else:
                pfx = 'paretic'
                acc_x = grp[f'{pfx}_acc_x'][:][0]
                acc_y = grp[f'{pfx}_acc_y'][:][0]
                acc_z = grp[f'{pfx}_acc_z'][:][0]
                gyr_x = grp[f'{pfx}_gyr_x'][:][0]
                gyr_y = grp[f'{pfx}_gyr_y'][:][0]
                gyr_z = grp[f'{pfx}_gyr_z'][:][0]
                angle = grp[f'{pfx}_hip_angle'][:][0]
                angleV = grp[f'{pfx}_hip_angleV'][:][0]
                
                label_raw = grp['gc_hs'][:][0]
                phase = label_raw * 2 * np.pi 

            features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, angle, angleV], axis=1)
            labels_2d = np.stack([np.cos(phase), np.sin(phase)], axis=1)
            
            # Global Normalization
            # Reshape to (T, F) - already there
            features_norm = scaler.transform(features)
            
            return features_norm, labels_2d, key
            
        except KeyError as e:
            print(f"Error loading {key}: {e}")
            return None, None, None

def run_inference(model, features, domain, window_size=50):
    # sliding window
    length = features.shape[0]
    preds = []
    
    batch_windows = []
    
    # Simple loop
    for i in range(0, length - window_size + 1): # stride 1 for visualization
        window = features[i:i+window_size]
        batch_windows.append(window)
    
    batch_windows = np.array(batch_windows, dtype=np.float32)
    batch_tensor = torch.tensor(batch_windows).to(CONFIG['device'])
    
    start_idx = 0
    BATCH = 64
    all_preds = []
    
    with torch.no_grad():
        while start_idx < len(batch_tensor):
            batch = batch_tensor[start_idx : start_idx + BATCH]
            pred, _ = model(batch, domain=domain)
            all_preds.append(pred.cpu().numpy())
            start_idx += BATCH
            
    return np.concatenate(all_preds, axis=0)

def visualize_results():
    if not os.path.exists('sda_model_final.pth'):
        print("Model not found. Waiting for training...")
        return

    device = CONFIG['device']
    model = SDA_Dual_Model(CONFIG).to(device)
    model.load_state_dict(torch.load('sda_model_final.pth'))
    model.eval()
    print("Model loaded.")

    def process_and_plot(h5_file, mode, key_idx, title, filename):
        features, labels_gt, key_name = load_trial_data_and_normalize(h5_file, mode, 'sda_scaler.pkl', key_idx)
        if features is None: return
        
        # Predict
        preds = run_inference(model, features, mode, CONFIG['window_size'])
        
        # Align GT: Predictions correspond to the END of the window.
        valid_len = len(preds)
        gt_aligned = labels_gt[CONFIG['window_size']-1 : CONFIG['window_size']-1 + valid_len]
        
        # 1. Phase Portrait
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        step = 5
        plt.plot(gt_aligned[::step, 0], gt_aligned[::step, 1], 'g-', alpha=0.5, label='GT')
        # Preds: column 0 is Cos, column 1 is Sin (based on training)
        plt.plot(preds[::step, 0], preds[::step, 1], 'b--', alpha=0.5, label='Pred')
        plt.title(f'{title} ({key_name})\nPhase Portrait')
        plt.xlabel('Cos')
        plt.ylabel('Sin')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        
        # 2. Phase Angle (Time Series)
        # arctan2(y, x) -> arctan2(sin, cos)
        # GT is [Cos, Sin], so y=Index 1, x=Index 0
        phase_gt = np.arctan2(gt_aligned[:, 1], gt_aligned[:, 0])
        phase_pred = np.arctan2(preds[:, 1], preds[:, 0])
        
        # Normalize to 0-1
        phase_gt = (phase_gt + np.pi) / (2 * np.pi)
        phase_pred = (phase_pred + np.pi) / (2 * np.pi)

        plt.subplot(1, 2, 2)
        limit = 1000
        plt.plot(phase_gt[:limit], 'g-', label='GT')
        plt.plot(phase_pred[:limit], 'b--', label='Pred')
        plt.title('Estimated Phase (first 1000 steps)')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

    # Plot
    # Check if scaler exists
    if not os.path.exists('sda_scaler.pkl'):
        print("Scaler not found. Train first.")
        return

    process_and_plot(CONFIG['source_h5'], 'source', 0, "Source (Healthy)", 'plot_source_0.png')
    process_and_plot(CONFIG['target_h5'], 'target', 0, "Target (Stroke)", 'plot_target_0.png')
    process_and_plot(CONFIG['target_h5'], 'target', 2, "Target (Stroke)", 'plot_target_2.png')

if __name__ == "__main__":
    visualize_results()
