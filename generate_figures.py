import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import torch
import joblib
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials

def generate_figures():
    # 1. Load Aggregated Results
    results = []
    with open('results_analysis/aggregated_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
            
    # Process for Plotting
    # Group by Subject, Mode
    # Filter modes order: SDA, TO, TL (TA), SO
    modes_order = ['SDA', 'TO', 'TL', 'SO']
    modes_label = {'SDA': 'Proposed (SDA)', 'TO': 'Target Only', 'TL': 'Target Augment', 'SO': 'Source Only'}
    colors = {'SDA': '#d62728', 'TO': '#1f77b4', 'TL': '#2ca02c', 'SO': '#7f7f7f'}
    
    subjects = sorted(list(set([r['Subject'] for r in results])))
    # Calculate Average
    
    # Store means per subject-mode
    table = {s: {m: {'rmse': [], 'r2': []} for m in modes_order} for s in subjects}
    
    for r in results:
        s = r['Subject']
        m = r['Mode']
        if m not in modes_order: continue
        table[s][m]['rmse'].append(float(r['RMSE']))
        table[s][m]['r2'].append(float(r['R2']))
        
    # Prepare data for bar plot
    # X-axis: Subjects + Average
    categories = subjects + ['Avg']
    n_cats = len(categories)
    
    bar_width = 0.2
    x = np.arange(n_cats)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot RMSE
    ax = axes[0]
    for i, mode in enumerate(modes_order):
        means = []
        errs = []
        for s in subjects:
            vals = table[s][mode]['rmse']
            if len(vals) > 0:
                means.append(np.mean(vals))
                errs.append(np.std(vals))
            else:
                means.append(0)
                errs.append(0)
        
        # Calculate overall average across all subjects
        all_vals = []
        for s in subjects:
            all_vals.extend(table[s][mode]['rmse'])
        avg_mean = np.mean(all_vals) if len(all_vals) > 0 else 0
        avg_std = np.std(all_vals) if len(all_vals) > 0 else 0
        
        means.append(avg_mean)
        errs.append(avg_std)
        
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, means, width=bar_width, label=modes_label[mode], color=colors[mode], yerr=errs, capsize=3)
        
    ax.set_ylabel('RMSE (Normalized Phase)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title('(a) RMSE Comparison')
    ax.legend(loc='upper left', ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Plot R2
    ax = axes[1]
    for i, mode in enumerate(modes_order):
        means = []
        errs = []
        for s in subjects:
            vals = table[s][mode]['r2']
            if len(vals) > 0:
                means.append(np.mean(vals))
                errs.append(np.std(vals))
            else:
                means.append(0)
                errs.append(0)
                
        all_vals = []
        for s in subjects:
            all_vals.extend(table[s][mode]['r2'])
        avg_mean = np.mean(all_vals) if len(all_vals) > 0 else 0
        avg_std = np.std(all_vals) if len(all_vals) > 0 else 0
        
        means.append(avg_mean)
        errs.append(avg_std)
        
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, means, width=bar_width, label=modes_label[mode], color=colors[mode], yerr=errs, capsize=3)
        
    ax.set_ylabel('R2 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title('(b) R2 Comparison')
    ax.set_ylim(-0.5, 1.1) # Limit Y to reasonable R2 range
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if not os.path.exists('paper_work/figure'):
        os.makedirs('paper_work/figure')
    plt.savefig('paper_work/figure/result_rmse_r2.png', dpi=300)
    print("Saved paper_work/figure/result_rmse_r2.png")
    
    # 2. Time Domain Plot (S003 SDA)
    # Re-run inference for a nice plot
    print("Generating time domain plot...")
    try:
        subj = 'S003'
        trial = 'S003_T002' # Found in previous step
        # Ideally find the folder
        subj_dir = f'results/{subj}'
        target_folder = None
        for f in os.listdir(subj_dir):
            if 'SDA' in f and 'frac1.0' in f and 'T002' in f:
                target_folder = f
                break
        
        if target_folder:
            folder_path = os.path.join(subj_dir, target_folder)
            model_path = os.path.join(folder_path, 'final_model.pth')
            scaler_path = os.path.join(folder_path, 'scaler.pkl')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            CONFIG['target_subject'] = subj
            model = SDA_Dual_Model(CONFIG).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            scaler = joblib.load(scaler_path)
            
            tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [trial])
            
            N, T, F_dim = tgt_test_data.shape
            tgt_test_norm = scaler.transform(tgt_test_data.reshape(-1, F_dim)).reshape(N, T, F_dim)
            batch_x = torch.tensor(tgt_test_norm, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                preds, _ = model(batch_x, domain='target')
                preds = preds.cpu().numpy()
                
            pred_phase = np.arctan2(preds[:, 1], preds[:, 0])
            gt_phase = np.arctan2(tgt_test_labels[:, 1], tgt_test_labels[:, 0])
            
            pred_phase_cont = np.unwrap(pred_phase) / (2*np.pi)
            gt_phase_cont = np.unwrap(gt_phase) / (2*np.pi)
            
            # Plot
            fig = plt.figure(figsize=(10, 4))
            
            # (a) Phase Portrait
            ax1 = fig.add_subplot(1, 2, 1)
            # Plot unit circle ref
            theta = np.linspace(0, 2*np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
            # Plot GT
            ax1.plot(tgt_test_labels[0:200, 0], tgt_test_labels[0:200, 1], 'g-', label='Ground Truth', linewidth=2)
            # Plot Pred
            ax1.plot(preds[0:200, 0], preds[0:200, 1], 'r--', label='Estimated', linewidth=1.5)
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.set_aspect('equal')
            ax1.set_xlabel('Cos(Phase)')
            ax1.set_ylabel('Sin(Phase)')
            ax1.set_title('(a) Phase Portrait')
            ax1.legend()
            
            # (b) Time Series
            ax2 = fig.add_subplot(1, 2, 2)
            t = np.arange(300) / 100.0 # 3 seconds
            
            # Identify first index where phase wraps to align visual if needed, but unwrapped is fine
            # Let's plot Wrapped phase for clarity like sawtooth? No, paper says "linear increase".
            # "The estimated gait phase over time... exhibit a linear increase"
            # So unwrapped is better or Sawtooth (0-1).
            # The draft says "continuous gait phase" and Fig 5b shows linear increase.
            # But usually we show 0-100% reset.
            # The text says "linearly increasing from 0 to 1... repeating periodically."
            # So it is sawtooth.
            
            # Convert continuous back to 0-1
            pred_saw = pred_phase_cont % 1.0
            gt_saw = gt_phase_cont % 1.0
            
            # To plot nicely without vertical lines at reset, we can separate segments or just plot points.
            # Or just plot unwrapped for a short duration.
            # "Both traces exhibit a linear increase... repeating periodically." implies sawtooth but visually continuous lines?
            # Actually, "linear increase from 0 to 1" IS sawtooth.
            # Let's plot sawtooth but handle wrap-around artifacts if connecting lines.
            
            # Simple workaround: Plot Unwrapped for 2-3 cycles.
            ax2.plot(t, gt_phase_cont[0:300], 'g-', label='Ground Truth', linewidth=2)
            ax2.plot(t, pred_phase_cont[0:300], 'r--', label='Estimated', linewidth=1.5)
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Gait Phase (cycles)')
            ax2.set_title('(b) Time Domain Estimation')
            ax2.legend()
            ax2.grid(linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('paper_work/figure/gaitphase_time_domain.png', dpi=300)
            print("Saved paper_work/figure/gaitphase_time_domain.png")
            
    except Exception as e:
        print(f"Error generating time domain plot: {e}")

if __name__ == "__main__":
    generate_figures()
