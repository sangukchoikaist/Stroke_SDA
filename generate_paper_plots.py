
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
import scipy.stats as stats

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

    # Calculate Overall Stats (Grand Mean over all subjects' means)
    subjects_plus = subjects + ['Overall']
    
    # Collect GLOBAL raw values for statistics (SDA vs others)
    global_rmse = {'SDA': [], 'TO': [], 'TL': [], 'SO': []} 
    
    # We need to re-iterate or store them during the main loop. 
    # Let's re-build the collection logic slightly to be cleaner or just iterate again?
    # Better to store in the main loop. But variables are local.
    # Let's modify the collection part above to populate global lists.
    
    # ... Wait, I can't easily jump back to the loop in replace_file_content.
    # I will rewrite the loop section and the plotting section.
    
    # Let's restart the Main Loop logic here for clarity in Replacement
    
    rmse_means = {'SDA': [], 'TO': [], 'TL': [], 'SO': [], 'SO_TS': [], 'TL_TS': []}
    rmse_stds = {'SDA': [], 'TO': [], 'TL': [], 'SO': [], 'SO_TS': [], 'TL_TS': []}
    max_means = {'SDA': [], 'TO': [], 'TL': [], 'SO': [], 'SO_TS': [], 'TL_TS': []}
    max_stds = {'SDA': [], 'TO': [], 'TL': [], 'SO': [], 'SO_TS': [], 'TL_TS': []}
    
    # Global containers for Paired T-Test
    stats_sda_to = {'SDA': [], 'TO': []}
    stats_sda_tl = {'SDA': [], 'TL': []}
    stats_sda_so = {'SDA': [], 'SO': []}
    stats_sda_sots = {'SDA': [], 'SO_TS': []}
    stats_sda_tlts = {'SDA': [], 'TL_TS': []}
    
    for subj in subjects:
        s_path = os.path.join(CONFIG['results_dir'], subj)
        
        # Temp storage for this subject to compute mean/std
        subj_rmse = {'SDA': [], 'TO': [], 'TL': [], 'SO': [], 'SO_TS': [], 'TL_TS': []}
        subj_max = {'SDA': [], 'TO': [], 'TL': [], 'SO': [], 'SO_TS': [], 'TL_TS': []}
        
        if os.path.exists(s_path): 
             # We need to iterate folds to ensure alignment
             for fold in [1, 2, 3]:
                 found_fold = {'SDA': None, 'TO': None, 'TL': None, 'SO': None, 'SO_TS': 'Recalc', 'TL_TS': 'Recalc'}
                 
                 for d in os.listdir(s_path):
                     if d.startswith(f"Fold_{fold}_"):
                         if "_SDA_" in d and "beta" not in d: found_fold['SDA'] = d
                         elif "_TO_" in d: 
                             # Prioritize IndepNorm if duplicate exists
                             if "IndepNorm" in d:
                                 found_fold['TO'] = d
                             elif found_fold['TO'] is None:
                                 found_fold['TO'] = d
                         elif "_TL_" in d: 
                             if "IndepNorm" in d:
                                 found_fold['TL_TS'] = d
                             else:
                                 found_fold['TL'] = d
                     elif "_SO_" in d: found_fold['SO'] = d
                 
                 # Load values
                 fold_vals = {}
                 for m in ['SDA', 'TO', 'TL', 'SO', 'SO_TS', 'TL_TS']:
                     val = None
                     vmax = None
                     
                     # Determine Source Directory for Model/Results
                     target_fold_retrieval = m 
                     if m == 'SO_TS': target_fold_retrieval = 'SO'
                     
                     if found_fold[target_fold_retrieval]:
                         full_d = os.path.join(s_path, found_fold[target_fold_retrieval])
                         
                         if m == 'SO_TS':
                             # Always recalc on fly
                             pass
                         else:
                             rmse_path = os.path.join(full_d, 'final_rmse_pct.txt')
                             max_path = os.path.join(full_d, 'final_max_error_pct.txt')
                             if os.path.exists(rmse_path):
                                 try: val = float(open(rmse_path).read().strip())
                                 except: pass
                             if os.path.exists(max_path):
                                 try: vmax = float(open(max_path).read().strip())
                                 except: pass

                         if val is None or vmax is None:
                             # Recalc logic
                             model_path = os.path.join(full_d, 'final_model.pth')
                             if not os.path.exists(model_path): model_path = os.path.join(full_d, 'best_model.pth')
                             
                             if os.path.exists(model_path):
                                 try:
                                     # Infer trial from folder name: Fold_1_S003_T002_SDA_...
                                     # Folder name corresponds to target_fold_retrieval
                                     pts = found_fold[target_fold_retrieval].split('_')
                                     try:
                                         if target_fold_retrieval in pts: idx = pts.index(target_fold_retrieval)
                                         else: idx = 4 # Fallback assumption
                                         trial = "_".join(pts[2:idx])
                                     except: 
                                         trial = None
                                     
                                     if trial:
                                         if m == 'SO_TS': # Special handling for On-the-fly Mode
                                              # print(f"DEBUG: Processing {m} for {subj} {trial}")
                                              X_raw, Y_raw = load_target_data_by_trials(CONFIG, [trial])
                                              
                                              if len(X_raw) > 0:
                                                  # Fit Scaler on TEST data
                                                  N, T, F = X_raw.shape
                                                  scaler = StandardScaler()
                                                  X_flat = X_raw.reshape(-1, F)
                                                  X_norm_flat = scaler.fit_transform(X_flat)
                                                  X_norm = X_norm_flat.reshape(N, T, F)
                                                  Xt = torch.tensor(X_norm, dtype=torch.float32)
                                                  Yt = Y_raw
                                              else: 
                                                  Xt, Yt = None, None
                                                  # print(f"WARNING: No data for {m} {subj} {trial}")
                                              
                                              param = 'source'
                                         else:
                                             scaler_path = os.path.join(full_d, 'scaler.pkl')
                                             Xt, Yt = load_data_consistent(trial, scaler_path)
                                             if m == 'SO': param = 'source'
                                             else: param = 'target'
                                         
                                         if Xt is not None:
                                             model = SDA_Dual_Model(CONFIG)
                                             model.load_state_dict(torch.load(model_path, map_location='cpu'))
                                             model.eval()
                                             with torch.no_grad():
                                                 p, _ = model(Xt, domain=param)
                                                 val = calculate_rmse(p.numpy(), Yt)
                                                 vmax = calculate_max_error(p.numpy(), Yt)
                                                 
                                                 # Save only if it's a real folder based recalc
                                                 if m not in ['SO_TS', 'TL_TS']:
                                                     with open(rmse_path, 'w') as f: f.write(str(val))
                                                     with open(max_path, 'w') as f: f.write(str(vmax))
                                 except Exception as e:
                                     print(f"Error recalc {m} {subj}: {e}")

                         if val is not None:
                             subj_rmse[m].append(val)
                         if vmax is not None:
                             subj_max[m].append(vmax)
                         
                         fold_vals[m] = val
                     else:
                         fold_vals[m] = None
                 
                 # Collect Paired Data for Stats
                 if fold_vals['SDA'] is not None and fold_vals['TO'] is not None:
                     stats_sda_to['SDA'].append(fold_vals['SDA'])
                     stats_sda_to['TO'].append(fold_vals['TO'])
                     
                 if fold_vals['SDA'] is not None and fold_vals['TL'] is not None:
                     stats_sda_tl['SDA'].append(fold_vals['SDA'])
                     stats_sda_tl['TL'].append(fold_vals['TL'])

                 if fold_vals['SDA'] is not None and fold_vals['SO'] is not None:
                     stats_sda_so['SDA'].append(fold_vals['SDA'])
                     stats_sda_so['SO'].append(fold_vals['SO'])
                 
                 if fold_vals['SDA'] is not None and fold_vals['SO_TS'] is not None:
                     stats_sda_sots['SDA'].append(fold_vals['SDA'])
                     stats_sda_sots['SO_TS'].append(fold_vals['SO_TS'])
                 
                 if fold_vals['SDA'] is not None and fold_vals['TL_TS'] is not None:
                     stats_sda_tlts['SDA'].append(fold_vals['SDA'])
                     stats_sda_tlts['TL_TS'].append(fold_vals['TL_TS'])
        
        # After processing a subject, store means
        for k in ['SDA', 'TO', 'TL', 'SO', 'SO_TS', 'TL_TS']:
            if subj_rmse[k]:
                rmse_means[k].append(np.mean(subj_rmse[k]))
                rmse_stds[k].append(np.std(subj_rmse[k], ddof=1) if len(subj_rmse[k])>1 else 0)
                max_means[k].append(np.mean(subj_max[k]))
                max_stds[k].append(np.std(subj_max[k], ddof=1) if len(subj_max[k])>1 else 0)
            else:
                 rmse_means[k].append(0); rmse_stds[k].append(0)
                 max_means[k].append(0); max_stds[k].append(0)

    # Perform T-Tests on the aggregated paired data
    print("\n>>> Statistical Significance (Paired t-test) <<<")
    # SDA vs TO
    if len(stats_sda_to['SDA']) > 1:
        stat, p = stats.ttest_rel(stats_sda_to['SDA'], stats_sda_to['TO'])
        print(f"SDA vs TO (N={len(stats_sda_to['SDA'])}): t={stat:.4f}, p={p:.5e}")
    else:
        print("Not enough paired data for SDA vs TO")

    # SDA vs TL
    if len(stats_sda_tl['SDA']) > 1:
        stat, p = stats.ttest_rel(stats_sda_tl['SDA'], stats_sda_tl['TL'])
        print(f"SDA vs TL (N={len(stats_sda_tl['SDA'])}): t={stat:.4f}, p={p:.5e}")

    # SDA vs SO
    if len(stats_sda_so['SDA']) > 1:
        stat, p = stats.ttest_rel(stats_sda_so['SDA'], stats_sda_so['SO'])
        print(f"SDA vs SO (N={len(stats_sda_so['SDA'])}): t={stat:.4f}, p={p:.5e}")

    # SDA vs SO_TS
    if len(stats_sda_sots['SDA']) > 1:
        stat, p = stats.ttest_rel(stats_sda_sots['SDA'], stats_sda_sots['SO_TS'])
        print(f"SDA vs SO_TS (N={len(stats_sda_sots['SDA'])}): t={stat:.4f}, p={p:.5e}")

    # SDA vs TL_TS
    if len(stats_sda_tlts['SDA']) > 1:
        stat, p = stats.ttest_rel(stats_sda_tlts['SDA'], stats_sda_tlts['TL_TS'])
        print(f"SDA vs TL_TS (N={len(stats_sda_tlts['SDA'])}): t={stat:.4f}, p={p:.5e}")

    # Calculate Overall Mean/Std
    for m in ['SDA', 'TO', 'TL', 'SO', 'SO_TS', 'TL_TS']:
        vals = rmse_means[m]
        if len(vals) > 0:
            overall_mu = np.mean(vals)
            overall_sigma = np.std(vals)
        else:
            overall_mu = 0
            overall_sigma = 0
            
        rmse_means[m].append(overall_mu)
        rmse_stds[m].append(overall_sigma)
        
        vals_max = max_means[m]
        if len(vals_max) > 0:
            overall_mu_max = np.mean(vals_max)
            overall_sigma_max = np.std(vals_max)
        else:
            overall_mu_max = 0
            overall_sigma_max = 0
        
        max_means[m].append(overall_mu_max)
        max_stds[m].append(overall_sigma_max)
        
    # Font Settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

    # --- Plotting ---
    subjects_plus = subjects + ['Overall']
    x = np.arange(len(subjects_plus))
    
    # Updated Modes to Plot (User Request: Improved Design)
    # Professional Palette (Flat UI Colors)
    modes_to_plot = [
        ('SO_TS', 'Source Only', '#95A5A6'),   # Concrete Gray
        ('TO', 'Target Only', '#3498DB'),      # Peter River Blue
        ('TL_TS', 'Fine-Tuning', '#F39C12'),   # Orange
        ('SDA', 'SDA (Proposed)', '#E74C3C')   # Alizarin Red
    ]
    
    width = 0.18 # Wider bars for 4 categories
    
    # Common Plot Style Helper
    def apply_style(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.grid(axis='y', linestyle=':', alpha=0.6, color='gray')
        ax.set_axisbelow(True) 
        ax.tick_params(axis='both', which='major', labelsize=14, width=1.5) # Increased Tick Font

    # 1. RMSE Plot
    # Compact Size for Single Column (scaled heavily, so fonts must be huge)
    fig, ax = plt.subplots(figsize=(8, 5)) 
    
    # Adjust width for tighter spacing
    width = 0.18 
    
    for i, (m_key, m_label, m_color) in enumerate(modes_to_plot):
        offset = (i - 1.5) * width 
        rects = ax.bar(x + offset, rmse_means[m_key], width, yerr=rmse_stds[m_key], 
                label=m_label, color=m_color, alpha=0.9, capsize=4, error_kw={'elinewidth': 1.8, 'ecolor': '#404040'})
    
    ax.set_ylabel('Gait Phase Estimation RMSE (%)', fontsize=16, fontweight='bold') # Huge Label
    ax.set_xticks(x)
    ax.set_xticklabels(subjects_plus, fontsize=14, fontweight='bold')
    # Legend: remove title, make distinct
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=13) 
    
    apply_style(ax)
    
    plt.tight_layout()
    
    # Save Vector Formats
    plt.savefig('paper_plot_bar_rmse.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_plot_bar_rmse.svg', format='svg', bbox_inches='tight')
    plt.savefig('paper_plot_bar_rmse.eps', format='eps', dpi=300, bbox_inches='tight') # Added EPS
    plt.savefig('paper_plot_bar_rmse.png', dpi=300, bbox_inches='tight') 
    print("Saved paper_plot_bar_rmse in PDF, SVG, EPS, PNG (Paper-Ready)")
    plt.close()
 
    # 2. Max Error Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (m_key, m_label, m_color) in enumerate(modes_to_plot):
        offset = (i - 1.5) * width
        ax.bar(x + offset, max_means[m_key], width, yerr=max_stds[m_key], 
                label=m_label, color=m_color, alpha=0.9, capsize=4, error_kw={'elinewidth': 1.8, 'ecolor': '#404040'})
    
    ax.set_ylabel('Gait Phase Est. Max Error (%)', fontsize=16, fontweight='bold') # Shortened label for fit
    ax.set_xticks(x)
    ax.set_xticklabels(subjects_plus, fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=13)
    
    apply_style(ax)
    
    plt.tight_layout()
    plt.savefig('paper_plot_bar_max.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_plot_bar_max.svg', format='svg', bbox_inches='tight')
    plt.savefig('paper_plot_bar_max.eps', format='eps', dpi=300, bbox_inches='tight') # Added EPS
    plt.savefig('paper_plot_bar_max.png', dpi=300, bbox_inches='tight')
    print("Saved paper_plot_bar_max in PDF, SVG, EPS, PNG (Paper-Ready)")
    plt.close()
 
    # --- Print Markdown Table ---
    print("\n### Quantitative Results (RMSE %)")
    print("| Subject | Source Only | Target Only | Fine-Tuning | SDA |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    for i, subj in enumerate(subjects_plus):
        row_str = f"| **{subj}** |"
        # User requested: SO -> SO_TS data, Fine-Tuning -> TL_TS data
        for m in ['SO_TS', 'TO', 'TL_TS', 'SDA']:
            mu = rmse_means[m][i]
            sigma = rmse_stds[m][i]
            if mu == 0 and sigma == 0:
                row_str += " - |"
            else:
                row_str += f" {mu:.2f} ± {sigma:.2f} |"
        print(row_str)
    print("\n")

if __name__ == "__main__":
    generate_plots()
