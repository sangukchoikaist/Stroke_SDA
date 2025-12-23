import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data, normalize_independent
import h5py

# Override Config to match the saved run
CONFIG['results_dir'] = 'results_analysis/Ablation_NoMMD'
CONFIG['target_subject'] = 'S013'
CONFIG['batch_size'] = 64
CONFIG['input_dim'] = 8
CONFIG['window_size'] = 100
CONFIG['stride_tgt'] = 5
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Target Trial (Test)
test_trial = 'S013_T002'
fold_idx = 1
mode = 'SO'
normalization_mode = 'fixed' # The one we want to visualize
suffix = "" # Fixed norm has no suffix

def visualize():
    print(f"Visualizing {mode} with {normalization_mode} normalization...")
    
    # Path to saved model
    out_dir = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'], f"Fold_{fold_idx}_{test_trial}_{mode}_frac1.0{suffix}")
    model_path = os.path.join(out_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Load Data
    # We need to re-load and normalize exactly as in training
    # 1. Load Source (for reference, though unused for Target Norm)
    src_data, _ = load_source_data(CONFIG)
    
    # 2. Get Train Trials (for fitting scaler)
    target_trials = get_target_trials(CONFIG)
    train_trials = [t for t in target_trials if t != test_trial]
    
    # 3. Load Target Train & Test
    tgt_train_data, _ = load_target_data_by_trials(CONFIG, train_trials)
    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
    
    # 4. Normalize (Independent)
    # Note: We need to fit scaler on Target Train and apply to Test
    _, _, tgt_test_norm, scaler_tgt = normalize_independent(src_data, tgt_train_data, tgt_test_data)
    
    # Load Model
    model = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Inference
    X_test = torch.tensor(tgt_test_norm, dtype=torch.float32).to(CONFIG['device'])
    with torch.no_grad():
        if mode == 'SO':
            pred, _ = model(X_test, domain='source') # Source Decoder
        else:
            pred, _ = model(X_test, domain='target')
            
    pred_np = pred.cpu().numpy()
    
    # Convert to Phase (0~100%)
    # pred is (cos, sin)
    def to_phase(vals):
        # vals: (N, 2) -> (N,) phase in 0~1
        phase = np.arctan2(vals[:, 1], vals[:, 0]) / (2*np.pi)
        phase = np.mod(phase, 1.0)
        return phase * 100.0
        
    pred_phase = to_phase(pred_np)
    true_phase = to_phase(tgt_test_labels)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    # Plot a specific range
    start_idx = 0
    end_idx = 1000 # 10 seconds (if 100Hz) -> actually window stride matters. 
    # Data is windowed. Plotting concatenated windows? 
    # Just plotting the predictions as sequence. 
    # Since stride=5, they overlap? No, we just plot the sequence of predictions.
    # Prediction is for the last point of the window.
    
    plt.plot(true_phase[start_idx:end_idx], label='True Phase', color='black', alpha=0.7)
    plt.plot(pred_phase[start_idx:end_idx], label='Predicted Phase (SO)', color='red', linestyle='--')
    plt.title(f"Gait Phase Estimation (Source Only, Independent Norm)\nTest Trial: {test_trial}")
    plt.ylabel("Gait Phase (%)")
    plt.xlabel("Sample Index (Stride=5)")
    plt.legend()
    plt.grid(True)
    
    save_path = "plot_prediction_SO_FixedNorm_S013.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    visualize()
