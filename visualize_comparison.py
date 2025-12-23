
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials

def visualize_comparison():
    subject = CONFIG['target_subject'] # 'S003'
    # Hardcoded trials for S003 based on logs ['S003_T002', 'S003_T003', 'S003_T006']
    # If generic, we should load from dataset, but hardcoding ensures we match the training folds.
    target_trials = ['S003_T002', 'S003_T003', 'S003_T006']
    modes = ['SDA', 'TO', 'SO']
    colors = {'SDA': 'blue', 'TO': 'red', 'SO': 'orange', 'GT': 'green'}
    
    for i, test_trial in enumerate(target_trials):
        fold_idx = i + 1
        print(f"\nGeneratin Comparison for Fold {fold_idx}: {test_trial}")
        
        plt.figure(figsize=(15, 6))
        
        # 1. Load Test Data & Scaler (Prefer SDA scaler)
        base_dir = f"results/{subject}/Fold_{fold_idx}_{test_trial}_SDA"
        if not os.path.exists(base_dir):
            print(f"  Warning: SDA results for {test_trial} not found at {base_dir}. Skipping.")
            continue

        scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
        tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
        N, T, F = tgt_test_data.shape
        tgt_test_norm = scaler.transform(tgt_test_data.reshape(-1, F)).reshape(N, T, F)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_x = torch.tensor(tgt_test_norm, dtype=torch.float32).to(device)
        
        # Plot GT Phase
        gt_phase = np.arctan2(tgt_test_labels[:, 1], tgt_test_labels[:, 0])
        # Unwrap for continuity? User asked for 0-1. 
        # So we map [-pi, pi] to [0, 1]
        gt_phase_01 = (gt_phase + np.pi) % (2 * np.pi) / (2 * np.pi)
        # Note: arctan2 is (-pi, pi]. +pi makes it (0, 2pi]. modulo? 
        # Wait, if we just want 0-1 for display:
        # We can just plot (gt_phase + np.pi) / (2*np.pi) (approx 0-1 but wraps)
        # But plotting wrapped phase in time series looks like sawtooth. That is fine.
        gt_phase_01 = (gt_phase % (2*np.pi)) / (2*np.pi) 
        
        ax1 = plt.subplot(1, 2, 1) # Phase Portrait
        ax2 = plt.subplot(1, 2, 2) # Time Series
        
        ax1.plot(tgt_test_labels[:, 0], tgt_test_labels[:, 1], color=colors['GT'], label='GT', alpha=0.9, linewidth=2)
        ax2.plot(gt_phase_01, color=colors['GT'], label='GT', linewidth=2, alpha=0.5)

        for mode in modes:
            mode_dir = f"results/{subject}/Fold_{fold_idx}_{test_trial}_{mode}"
            model_path = os.path.join(mode_dir, 'final_model.pth')
            
            if not os.path.exists(model_path):
                print(f"  Warning: {mode} model not found at {model_path}")
                continue
                
            print(f"  Loading {mode} model...")
            model = SDA_Dual_Model(CONFIG).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            with torch.no_grad():
                # Domain logic must match training
                domain = 'source' if mode == 'SO' else 'target'
                preds, _ = model(batch_x, domain=domain)
                preds = preds.cpu().numpy()
                
            pred_phase = np.arctan2(preds[:, 1], preds[:, 0])
            pred_phase_01 = (pred_phase % (2*np.pi)) / (2*np.pi)
            
            ax1.plot(preds[:, 0], preds[:, 1], color=colors[mode], label=mode, linestyle='--', alpha=0.7)
            ax2.plot(pred_phase_01, color=colors[mode], label=mode, linestyle='--', linewidth=1.5)
            
        ax1.set_title(f'Phase Portrait Comparison ({test_trial})')
        ax1.legend()
        ax1.axis('equal')
        
        ax2.set_title(f'Phase Estimation (0-1 Cycle) ({test_trial})')
        ax2.legend()
        
        outfile = f'plot_{subject}_comparison_{test_trial}.png'
        plt.tight_layout()
        plt.savefig(outfile)
        print(f"  Saved {outfile}")

if __name__ == "__main__":
    visualize_comparison()
