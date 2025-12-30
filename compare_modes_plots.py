
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data, normalize_fixed
from sda_training_indep import normalize_independent

# --- Configuration ---
# Match parameters from the run
CONFIG['results_dir'] = 'results_grid_search_theta'
CONFIG['batch_size'] = 64
CONFIG['input_dim'] = 8
CONFIG['window_size'] = 50 
CONFIG['stride'] = 5
CONFIG['stride_tgt'] = 5
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG['encoder_layers'] = [64, 32] 
CONFIG['decoder_layers'] = [32]
CONFIG['dropout'] = 0.3
CONFIG['output_dim'] = 2
CONFIG['feature_set'] = 'theta'
CONFIG['max_folds'] = 3

# Specific Run Params
LAMBDA_str = "L1.0_LS0.3_LR0.0001_E64_D32_Dr0.3"

def to_phase(vals):
    # vals: (N, 2) -> (N,) phase in 0~100
    phase = np.arctan2(vals[:, 1], vals[:, 0]) / (2*np.pi)
    phase = np.mod(phase, 1.0)
    return phase * 100.0

def process_subject_fold(subj, fold_idx, test_trial, result_base_path):
    print(f"Processing {subj} Fold {fold_idx} Trial {test_trial}...")
    
    # 1. Load Raw Data
    CONFIG['target_subject'] = subj
    src_data, _ = load_source_data(CONFIG)
    
    target_trials = get_target_trials(CONFIG)
    train_trials = [t for t in target_trials if t != test_trial]
    
    tgt_train_data, _ = load_target_data_by_trials(CONFIG, train_trials)
    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
    
    # 2. Prepare Normalized Inputs
    # Fixed Norm (for SDA, SO, TL)
    _, _, test_fixed, _ = normalize_fixed(src_data, tgt_train_data, tgt_test_data)
    
    # Independent Norm (for TO)
    _, _, test_indep, _ = normalize_independent(src_data, tgt_train_data, tgt_test_data)
    
    test_fixed_t = torch.tensor(test_fixed, dtype=torch.float32).to(CONFIG['device'])
    test_indep_t = torch.tensor(test_indep, dtype=torch.float32).to(CONFIG['device'])
    
    # 3. Load Models and Infer
    modes = ['SDA', 'SO', 'TO', 'TL']
    predictions = {}
    
    for mode in modes:
        # Construct folder name
        # TO has suffix _IndepNorm
        # Format: Fold_{i}_{trial}_{mode}_frac1.0[_IndepNorm]
        suffix = "_IndepNorm" if mode == 'TO' else ""
        folder_name = f"Fold_{fold_idx}_{test_trial}_{mode}_frac1.0{suffix}"
        
        mode_dir = os.path.join(result_base_path, folder_name)
        model_path = os.path.join(mode_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"  Model missing for {mode}: {model_path}")
            predictions[mode] = None
            continue
            
        print(f"  Loading {mode}...")
        model = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
        try:
            model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        except:
             print(f"  Failed to load {model_path}")
             predictions[mode] = None
             continue

        model.eval()
        
        # Select Input
        inp = test_indep_t if mode == 'TO' else test_fixed_t
        
        with torch.no_grad(): 
            if mode == 'SO':
                pred, _ = model(inp, domain='source')
            else:
                pred, _ = model(inp, domain='target')
        
        predictions[mode] = pred.cpu().numpy()

    # 4. Visualization
    
    # A. Phase Portrait Comparison
    plt.figure(figsize=(8, 8))
    plt.scatter(tgt_test_labels[:, 0], tgt_test_labels[:, 1], c='black', alpha=0.1, label='Ground Truth', s=10)
    
    colors = {'SDA': 'red', 'SO': 'blue', 'TO': 'green', 'TL': 'orange'}
    
    for mode in modes:
        pred = predictions[mode]
        if pred is not None:
             plt.scatter(pred[:, 0], pred[:, 1], c=colors[mode], alpha=0.4, label=mode, s=5)
             
    plt.title(f"Phase Portrait Comparison - {subj} Fold {fold_idx}")
    plt.xlabel("Cos")
    plt.ylabel("Sin")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    save_pp = os.path.join(result_base_path, f"Compare_PhasePortrait_{subj}_Fold{fold_idx}.png")
    plt.savefig(save_pp)
    plt.close()
    
    # B. Gait Phase Comparison (Time Series)
    # Convert to phase
    true_phase = to_phase(tgt_test_labels)
    pred_phases = {}
    for mode in modes:
        if predictions[mode] is not None:
            pred_phases[mode] = to_phase(predictions[mode])
            
    # Plotting first 500 samples
    N = 600
    plt.figure(figsize=(15, 6))
    plt.plot(true_phase[:N], label='Ground Truth', color='black', linewidth=2, alpha=0.6)
    
    for mode in modes:
        if mode in pred_phases:
            plt.plot(pred_phases[mode][:N], label=mode, color=colors[mode], linestyle='--', linewidth=1.5)
            
    plt.title(f"Gait Phase Estimation Comparison - {subj} Fold {fold_idx}")
    plt.xlabel("Time Step")
    plt.ylabel("Gait Phase (%)")
    plt.legend()
    plt.grid(True)
    save_ts = os.path.join(result_base_path, f"Compare_GaitPhase_{subj}_Fold{fold_idx}.png")
    plt.savefig(save_ts)
    plt.close()
    
    print(f"Saved plots to {result_base_path}")
    
    # Return metrics for aggregation
    metrics = {}
    for mode in modes:
        if mode in pred_phases:
             p_pred = pred_phases[mode]
             p_true = true_phase
             
             # Circular Error
             diff = np.abs(p_pred - p_true)
             diff = np.minimum(diff, 100.0 - diff)
             rmse = np.sqrt(np.mean(diff**2))
             metrics[mode] = rmse
             
             # Amplitude (Radius) Comparison
             # true radius is ~1.0
             pred_vectors = predictions[mode]
             radius = np.sqrt(pred_vectors[:, 0]**2 + pred_vectors[:, 1]**2)
             metrics[f"{mode}_amp"] = np.mean(radius)
        else:
             metrics[mode] = None
             metrics[f"{mode}_amp"] = None
             
    return metrics
             
    return metrics


def run_comparison():
    subjects = ['S003', 'S007']
    
    # Accumulate results
    # summary[subj][mode] = [rmse_fold1, rmse_fold2, ...]
    summary = {s: {} for s in subjects}
    for s in subjects:
        for m in ['SDA', 'SO', 'TO', 'TL']:
            summary[s][m] = []
            summary[s][f"{m}_amp"] = []
    
    for subj in subjects:
        # result_base_path = results_grid_search_theta/S003_L1.0_LS0.3_LR0.0001_E64_D32_Dr0.3/S003
        run_folder = f"{subj}_{LAMBDA_str}"
        result_base_path = os.path.join(CONFIG['results_dir'], run_folder, subj)
        
        if not os.path.exists(result_base_path):
             print(f"Directory not found: {result_base_path}")
             continue
             
        # Identify Folds
        subdirs = [d for d in os.listdir(result_base_path) if os.path.isdir(os.path.join(result_base_path, d))]
        
        # Determine (Fold, Trial) pairs present
        pairs = set()
        for d in subdirs:
            parts = d.split('_')
            if len(parts) >= 4 and parts[0] == 'Fold':
                fold = int(parts[1])
                trial = f"{parts[2]}_{parts[3]}"
                pairs.add((fold, trial))
        
        sorted_pairs = sorted(list(pairs))
        
        for fold, trial in sorted_pairs:
            metrics = process_subject_fold(subj, fold, trial, result_base_path)
            for m, val in metrics.items():
                if val is not None:
                    summary[subj][m].append(val)

    # Print Summary Table
    print("\n\n========================================================")
    print("             Phase RMSE Comparison (Circular)           ")
    print("========================================================")
    print(f"{'Subject':<10} {'Mode':<10} {'Mean RMSE (%)':<15} {'Std Dev':<10} {'Folds':<10}")
    print("-" * 60)
    
    for subj in subjects:
        for mode in ['SDA', 'SO', 'TO', 'TL']:
            vals = summary[subj][mode]
            if vals:
                mean_rmse = np.mean(vals)
                std_rmse = np.std(vals)
                n = len(vals)
                print(f"{subj:<10} {mode:<10} {mean_rmse:<15.4f} {std_rmse:<10.4f} {n:<10}")
            else:
                print(f"{subj:<10} {mode:<10} {'N/A':<15} {'N/A':<10} {0:<10}")
        print("-" * 60)
        
    print("\n\n========================================================")
    print("             Mean Amplitude Comparison (Target ~1.0)    ")
    print("========================================================")
    print(f"{'Subject':<10} {'Mode':<10} {'Mean Radius':<15} {'Std Dev':<10}")
    print("-" * 60)
    
    for subj in subjects:
        for mode in ['SDA', 'SO', 'TO', 'TL']:
            vals = summary[subj].get(f"{mode}_amp", [])
            if vals:
                mean_amp = np.mean(vals)
                std_amp = np.std(vals)
                print(f"{subj:<10} {mode:<10} {mean_amp:<15.4f} {std_amp:<10.4f}")
            else:
                print(f"{subj:<10} {mode:<10} {'N/A':<15} {'N/A':<10}")
        print("-" * 60)

if __name__ == "__main__":
    run_comparison()
