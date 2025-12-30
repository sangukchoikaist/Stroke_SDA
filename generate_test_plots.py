
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data, normalize_fixed
import h5py

# --- Specific Config for the Model to Visualize ---
# Path: results_grid_search_theta/S003_L2.0_LR0.0001_E64_D64_Dr0.3/S003/Fold_1_S003_T002_SDA_frac1.0
CONFIG['results_dir'] = 'results_grid_search_theta/S003_L1.0_LS0.3_LR0.0001_E64_D32_Dr0.3'
CONFIG['target_subject'] = 'S003'
CONFIG['batch_size'] = 64
CONFIG['input_dim'] = 8
CONFIG['window_size'] = 50 
CONFIG['stride'] = 5
CONFIG['stride_tgt'] = 5
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Hyperparameters (Must match filename/training)
CONFIG['encoder_layers'] = [64, 32] # 64 -> 32
CONFIG['decoder_layers'] = [32]
CONFIG['dropout'] = 0.3
CONFIG['output_dim'] = 2
CONFIG['feature_set'] = 'theta'

# Run Details
test_trial = 'S003_T002'
fold_idx = 1
mode = 'SDA'
suffix = "" 


def visualize_folder(folder_name):
    print(f"\nProcessing {folder_name}...")
    
    # Parse metadata from folder name
    # Format: Fold_{fold_idx}_{test_trial}_{mode}_frac...
    parts = folder_name.split('_')
    # Fold_1_S003_T002_SDA_frac1.0
    # parts[0] = Fold
    # parts[1] = 1
    # parts[2] = S003
    # parts[3] = T002
    # parts[4] = SDA
    
    fold_idx =  int(parts[1])
    test_trial = f"{parts[2]}_{parts[3]}"
    mode = parts[4]
    
    print(f"  Fold: {fold_idx}, Trial: {test_trial}, Mode: {mode}")

    out_dir = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'], folder_name)
    model_path = os.path.join(out_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(out_dir, 'final_model.pth')
        if not os.path.exists(model_path):
            print(f"  Model not found at {model_path}")
            return

    # Load Data
    src_data, _ = load_source_data(CONFIG)
    
    target_trials = get_target_trials(CONFIG)
    train_trials = [t for t in target_trials if t != test_trial]
    
    tgt_train_data, _ = load_target_data_by_trials(CONFIG, train_trials)
    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
    
    # Normalize (Fixed)
    _, _, tgt_test_norm, scaler = normalize_fixed(src_data, tgt_train_data, tgt_test_data)
    
    # Load Model
    model = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
    model.eval()
    
    # Inference
    X_test = torch.tensor(tgt_test_norm, dtype=torch.float32).to(CONFIG['device'])
    with torch.no_grad():
        if mode == 'SO':
            pred, _ = model(X_test, domain='source')
        else:
            pred, _ = model(X_test, domain='target')
            
    pred_np = pred.cpu().numpy()
    
    # Visualization
    # 1. Phase Portrait
    plt.figure(figsize=(6, 6))
    plt.scatter(tgt_test_labels[:, 0], tgt_test_labels[:, 1], alpha=0.1, label='True Phase', color='black', s=5)
    plt.scatter(pred_np[:, 0], pred_np[:, 1], alpha=0.3, label='Predicted', color='red', s=5)
    plt.title(f"Phase Portrait ({mode}) - Fold {fold_idx}\nTrial: {test_trial}")
    plt.xlabel("Output 1 (Cos)")
    plt.ylabel("Output 2 (Sin)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    save_path_pp = os.path.join(out_dir, "plot_phase_portrait.png")
    plt.savefig(save_path_pp)
    plt.close()
    print(f"  Saved Phase Portrait")
    
    # 2. Output vs Label
    N = 500
    if len(pred_np) > N:
        pred_sub = pred_np[:N]
        label_sub = tgt_test_labels[:N]
    else:
        pred_sub = pred_np
        label_sub = tgt_test_labels

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(label_sub[:, 0], label='True Cos', color='black', alpha=0.7)
    plt.plot(pred_sub[:, 0], label='Pred Cos', color='blue', linestyle='--')
    plt.title(f"Output 1 (Cos) - Fold {fold_idx}")
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(label_sub[:, 1], label='True Sin', color='black', alpha=0.7)
    plt.plot(pred_sub[:, 1], label='Pred Sin', color='red', linestyle='--')
    plt.title(f"Output 2 (Sin) - Fold {fold_idx}")
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    save_path_ts = os.path.join(out_dir, "plot_outputs_time_series.png")
    plt.savefig(save_path_ts)
    plt.close()
    print(f"  Saved Time Series Plot")

def run_all():
    base_path = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'])
    if not os.path.exists(base_path):
        print(f"Result path not found: {base_path}")
        return

    subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("Fold_")])
    
    if not subdirs:
        print("No Fold directories found.")
        return
        
    for d in subdirs:
        visualize_folder(d)

if __name__ == "__main__":
    run_all()
    
