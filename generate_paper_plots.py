
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import h5py


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import h5py

# Import from training script to ensure consistency
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials, get_target_trials, GaitDataset

# --- No manual config or class defs needed ---
CONFIG['device'] = 'cpu' # Force CPU for plots
# CONFIG['results_dir'] = 'results_final_rank1' # Use imported one

def load_data_consistent(trial_key, scaler_path):
    # Wrapper to load one trial using sda_training logic
    if not os.path.exists(scaler_path): return None, None
    scaler = joblib.load(scaler_path)
    
    # Temporarily set batch size or other flags? No, load_target_data returns numpy.
    
    # We must ensure CONFIG matches what load_target needs
    # It needs 'target_h5' (Already in CONFIG)
    # It needs 'window_size' (50)
    # It needs 'stride_tgt' (5) -> Used for Test
    
    X_raw, Y_raw = load_target_data_by_trials(CONFIG, [trial_key])
    if len(X_raw) == 0: return None, None
    
    # Normalize
    N, T, F = X_raw.shape
    X_norm = scaler.transform(X_raw.reshape(-1, F)).reshape(N, T, F)
    return torch.tensor(X_norm, dtype=torch.float32), Y_raw

def calculate_rmse(pred, gt):
    # Convert [cos, sin] to phase angle
    phi_pred = np.arctan2(pred[:, 1], pred[:, 0])
    phi_gt = np.arctan2(gt[:, 1], gt[:, 0])
    
    # Circular Difference
    diff = phi_pred - phi_gt
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    
    # Convert diff to percentage (2pi = 100%)
    diff_percent = diff * (100.0 / (2 * np.pi))
    
    rmse = np.sqrt(np.mean(diff_percent**2))
    return rmse

def calculate_max_error(pred, gt):
    phi_pred = np.arctan2(pred[:, 1], pred[:, 0])
    phi_gt = np.arctan2(gt[:, 1], gt[:, 0])
    diff = phi_pred - phi_gt
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    diff_percent = diff * (100.0 / (2 * np.pi))
    return np.max(np.abs(diff_percent))

def to_phase_percent(pred):
    phi = np.arctan2(pred[:, 1], pred[:, 0]) # -pi to pi
    # Map to 0-100%
    # -pi -> 0 (or 50?), 0 -> 50, pi -> 100? 
    # Usually: 0 phase is HS. 
    # arctan2(sin, cos). If label was constructed as cos(phase), sin(phase) where phase 0..2pi.
    # At 0: cos=1, sin=0. atan2(0,1) = 0.
    # At pi: cos=-1, sin=0. atan2(0,-1) = pi.
    # At 2pi (close to): -0.001. atan2 -> -0.
    # We want 0..2pi mappings.
    phi_unwrap = phi % (2*np.pi) # 0 to 2pi
    return phi_unwrap * (100.0 / (2 * np.pi))

def generate_plots():
    subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    target_h5 = CONFIG['target_h5'] # Use from imported CONFIG
    
    # 1. Comparison Plot (XY Trajectory)
    # We want to show SDA vs SO vs Ground Truth for a representative subject (e.g., S003)
    
    rep_subj = 'S003'
    fold = 1
    # Find a test trial from results dir
    res_path = os.path.join(CONFIG['results_dir'], rep_subj)
    # List directories to find specific trial
    try:
        dirs = os.listdir(res_path)
    except FileNotFoundError:
        print("Results not found yet. Training might be incomplete.")
        return

    # Look for results dirs
    sda_dir = None; so_dir = None; to_dir = None; tl_dir = None
    test_trial = None
    
    for d in dirs:
        if d.startswith(f"Fold_{fold}_") and "_SDA_" in d:
            sda_dir = os.path.join(res_path, d)
            parts = d.split('_')
            idx = parts.index("SDA")
            test_trial = "_".join(parts[2:idx])
    
    if not test_trial: return
    
    # Find others
    for d in dirs:
        if d.startswith(f"Fold_{fold}_{test_trial}_SO_"): so_dir = os.path.join(res_path, d)
        elif d.startswith(f"Fold_{fold}_{test_trial}_TO_"): to_dir = os.path.join(res_path, d)
        elif d.startswith(f"Fold_{fold}_{test_trial}_TL_"): tl_dir = os.path.join(res_path, d)
            
    # Helper to load model
    def load_model_from_dir(d, mode='SDA'):
        if not d: return None
        path = os.path.join(d, 'final_model.pth')
        if not os.path.exists(path): path = os.path.join(d, 'best_model.pth')
        if not os.path.exists(path): return None
        m = SDA_Dual_Model(CONFIG)
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.eval()
        return m

    model_sda = load_model_from_dir(sda_dir, 'SDA')
    model_so = load_model_from_dir(so_dir, 'SO')
    model_to = load_model_from_dir(to_dir, 'TO')
    model_tl = load_model_from_dir(tl_dir, 'TL')
    
    # Load Scaler
    scaler_path = os.path.join(sda_dir, 'scaler.pkl')
    X, Y_true = load_data_consistent(test_trial, scaler_path)
    
    # Inference
    preds = {}
    with torch.no_grad():
        if model_sda: 
            p, _ = model_sda(X, domain='target')
            preds['SDA'] = p.numpy()
        if model_so:
            p, _ = model_so(X, domain='source')
            preds['SO'] = p.numpy()
        if model_to:
            p, _ = model_to(X, domain='target')
            preds['TO'] = p.numpy()
        if model_tl:
            p, _ = model_tl(X, domain='target')
            preds['TL'] = p.numpy()
            
    # Plot XY
    plt.figure(figsize=(6, 6))
    plt.plot(Y_true[:,0], Y_true[:,1], 'k--', label='Ground Truth', alpha=0.4)
    if 'SO' in preds: plt.plot(preds['SO'][:,0], preds['SO'][:,1], 'b:', label='Source Only', linewidth=1, alpha=0.6)
    if 'TO' in preds: plt.plot(preds['TO'][:,0], preds['TO'][:,1], 'g--', label='Target Only', linewidth=1.5, alpha=0.7)
    if 'TL' in preds: plt.plot(preds['TL'][:,0], preds['TL'][:,1], 'm-.', label='Fine-Tuning', linewidth=1.5, alpha=0.7)
    if 'SDA' in preds: plt.plot(preds['SDA'][:,0], preds['SDA'][:,1], 'r-', label='SDA (Proposed)', linewidth=2.5)
    
    plt.xlabel('Cos Phase'); plt.ylabel('Sin Phase')
    plt.title(f'Trajectory Reconstruction ({rep_subj})')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('paper_plot_trajectory.png', dpi=300)
    print("Saved paper_plot_trajectory.png")
    
    # Plot Time Series (Phase %)
    plt.figure(figsize=(10, 4))
    t = np.arange(len(Y_true))
    # Convert to %
    def plot_phase_pct(vals, **kwargs):
        pct = to_phase_percent(vals)
        plt.scatter(t, pct, s=2, **kwargs) # scatter better for discont
    
    plot_phase_pct(Y_true, color='k', label='GT', alpha=0.3)
    if 'SO' in preds: plot_phase_pct(preds['SO'], color='b', label='SO', alpha=0.3) # scatter heavy?
    # Maybe plain plot with NaN on wrap? 
    # Just simple plot for now, visual check
    if 'TO' in preds: plt.plot(t, to_phase_percent(preds['TO']), 'g--', label='TO', linewidth=1, alpha=0.7)
    if 'TL' in preds: plt.plot(t, to_phase_percent(preds['TL']), 'm-.', label='TL', linewidth=1, alpha=0.7)
    if 'SDA' in preds: plt.plot(t, to_phase_percent(preds['SDA']), 'r-', label='SDA', linewidth=1.5)
    
    plt.xlim(0, 300) # Zoom in
    plt.xlabel('Time Step'); plt.ylabel('Gait Cycle (%)')
    plt.title('Phase Estimation (Gait Cycle %)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('paper_plot_timeseries.png', dpi=300)
    print("Saved paper_plot_timeseries.png")

    # 2. Bar Chart (RMSE & Max Error Comparison)
    # We collect RMSE and Max Error of Phase across all folds
    rmse_means = {'SDA': [], 'TO': [], 'TL': []}
    rmse_stds = {'SDA': [], 'TO': [], 'TL': []}
    max_means = {'SDA': [], 'TO': [], 'TL': []}
    max_stds = {'SDA': [], 'TO': [], 'TL': []}
    
    # Helper (Integrated Loop)
    for subj in subjects:
        s_path = os.path.join(CONFIG['results_dir'], subj)
        if not os.path.exists(s_path): 
            for k in rmse_means: 
                rmse_means[k].append(0); rmse_stds[k].append(0)
                max_means[k].append(0); max_stds[k].append(0)
            continue
            
        curr_vals = {'SDA': [], 'TO': [], 'TL': []}
        curr_maxs = {'SDA': [], 'TO': [], 'TL': []}
        
        for d in os.listdir(s_path):
            full_d = os.path.join(s_path, d)
            if not os.path.isdir(full_d): continue
            
            # Determine Mode
            mode = None
            if "_SDA_" in d: mode = 'SDA'
            # elif "_SO_" in d: mode = 'SO' # Exclude SO from logic as requested
            elif "_TO_" in d: mode = 'TO'
            elif "_TL_" in d: mode = 'TL'
            else: continue
            
            val_rmse = None
            val_max = None
            
            rmse_path = os.path.join(full_d, 'final_rmse_pct.txt')
            max_path = os.path.join(full_d, 'final_max_error_pct.txt')
            
            # Check if both exist
            if os.path.exists(rmse_path) and os.path.exists(max_path):
                try: 
                    val_rmse = float(open(rmse_path).read().strip())
                    val_max = float(open(max_path).read().strip())
                except: pass
            
            if val_rmse is None or val_max is None:
                # Recalc logic
                model_path = os.path.join(full_d, 'final_model.pth')
                if not os.path.exists(model_path): model_path = os.path.join(full_d, 'best_model.pth')
                if os.path.exists(model_path):
                    try:
                        pts = d.split('_'); idx = pts.index(mode)
                        trial = "_".join(pts[2:idx])
                        # Load Data
                        scaler_path = os.path.join(full_d, 'scaler.pkl')
                        Xt, Yt = load_data_consistent(trial, scaler_path)
                        
                        if Xt is not None:
                            m = SDA_Dual_Model(CONFIG)
                            m.load_state_dict(torch.load(model_path, map_location='cpu'))
                            m.eval()
                            with torch.no_grad():
                                if mode == 'SO': param = 'source'
                                else: param = 'target'
                                p, _ = m(Xt, domain=param)
                                # Metrics
                                val_rmse = calculate_rmse(p.numpy(), Yt)
                                val_max = calculate_max_error(p.numpy(), Yt)
                                
                                with open(rmse_path, 'w') as f: f.write(str(val_rmse))
                                with open(max_path, 'w') as f: f.write(str(val_max))
                    except Exception as e: 
                        pass
            
            if val_rmse is not None: curr_vals[mode].append(val_rmse)
            if val_max is not None: curr_maxs[mode].append(val_max)
            
        # Aggregate stats
        for k in rmse_means:
            if curr_vals[k]:
                rmse_means[k].append(np.mean(curr_vals[k]))
                rmse_stds[k].append(np.std(curr_vals[k]) if len(curr_vals[k]) > 1 else 0)
                max_means[k].append(np.mean(curr_maxs[k]))
                max_stds[k].append(np.std(curr_maxs[k]) if len(curr_maxs[k]) > 1 else 0)
            else:
                rmse_means[k].append(0); rmse_stds[k].append(0)
                max_means[k].append(0); max_stds[k].append(0)

    # Calculate Overall Stats (Grand Mean over all subjects' means)
    subjects_plus = subjects + ['Overall']
    for k in rmse_means:
        # Avoid zero entries if any
        valid_rmse = [v for v in rmse_means[k] if v > 0]
        valid_max = [v for v in max_means[k] if v > 0]
        
        if valid_rmse:
            overall_rmse_mean = np.mean(valid_rmse)
            overall_rmse_std = np.std(valid_rmse)
            rmse_means[k].append(overall_rmse_mean)
            rmse_stds[k].append(overall_rmse_std)
        else:
            rmse_means[k].append(0); rmse_stds[k].append(0)
            
        if valid_max:
            overall_max_mean = np.mean(valid_max)
            overall_max_std = np.std(valid_max)
            max_means[k].append(overall_max_mean)
            max_stds[k].append(overall_max_std)
        else:
            max_means[k].append(0); max_stds[k].append(0)

    # Plot Bar (RMSE & Max Error Subplots)
    x = np.arange(len(subjects_plus))
    # Make Overall separated? Just add to end for now.
    width = 0.25 
    fig, axes = plt.subplots(1, 2, figsize=(18, 6)) # Wider for extra bar
    
    modes_to_plot = [('TO', 'Target Only', 'green'), 
                     ('TL', 'Fine-Tuning', 'orange'), 
                     ('SDA', 'SDA (Proposed)', 'firebrick')]
    
    # Subplot 1: RMSE
    for i, (m_key, m_label, m_color) in enumerate(modes_to_plot):
        offset = (i - 1) * width
        axes[0].bar(x + offset, rmse_means[m_key], width, yerr=rmse_stds[m_key], 
                label=m_label, color=m_color, alpha=0.8, capsize=5)
    
    axes[0].set_ylabel('Phase RMSE (%) - Lower is Better')
    axes[0].set_title('Mean RMSE (3-Fold CV)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subjects_plus)
    # Highlight Overall?
    # axes[0].axvline(x[-2] + 0.5, color='k', linestyle='--', alpha=0.3)
    
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Max Error
    for i, (m_key, m_label, m_color) in enumerate(modes_to_plot):
        offset = (i - 1) * width
        axes[1].bar(x + offset, max_means[m_key], width, yerr=max_stds[m_key], 
                label=m_label, color=m_color, alpha=0.8, capsize=5)
    
    axes[1].set_ylabel('Max Phase Error (%) - Lower is Better')
    axes[1].set_title('Mean Maximum Error (3-Fold CV)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subjects_plus)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('paper_plot_bar.png', dpi=300)
    print("Saved paper_plot_bar.png with RMSE and Max Error (including Overall)")

if __name__ == "__main__":
    generate_plots()
