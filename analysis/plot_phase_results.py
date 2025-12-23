import os
import sys
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import joblib

# Add parent directory to path to possibly import config or util if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redefine Model Classes to ensure standalone execution without side effects
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dims[1]) 

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        _, (h_n, _) = self.lstm2(out) 
        z = h_n[-1]
        z = self.norm(z) 
        return z

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32], output_dim=2, dropout=0.0):
        super(Decoder, self).__init__()
        layers_list = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers_list.append(nn.Linear(in_d, h_d))
            layers_list.append(nn.ReLU())
            if dropout > 0: layers_list.append(nn.Dropout(dropout))
            in_d = h_d
        layers_list.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers_list)
    def forward(self, x): return self.net(x)

class SDA_Dual_Model(nn.Module):
    def __init__(self, config):
        super(SDA_Dual_Model, self).__init__()
        self.encoder = Encoder(config['input_dim'], config['encoder_layers'], dropout=config.get('dropout', 0.3))
        self.decoder_src = Decoder(config['encoder_layers'][-1], config['decoder_layers'], config['output_dim'])
        self.decoder_tgt = Decoder(config['encoder_layers'][-1], config['decoder_layers'], config['output_dim'])
    def forward(self, x, domain='target'):
        z = self.encoder(x)
        if domain == 'source': out = self.decoder_src(z)
        else: out = self.decoder_tgt(z)
        return out, z

# Configuration matches training
CONFIG = {
    'input_dim': 8,
    'encoder_layers': [128, 64],
    'decoder_layers': [128],
    'dropout': 0.3,
    'output_dim': 2,
    'window_size': 50,
    # Paths
    'results_dir': 'd:/RSC lab/Codes/Stroke_SDA/results',
    'data_path': 'd:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5'
}

def load_model(path, device='cpu'):
    # Try Config A: Standard (decoder 128)
    try:
        config = CONFIG.copy()
        model = SDA_Dual_Model(config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except RuntimeError as e:
        if 'size mismatch' in str(e):
            print(f"Size mismatch detected. Trying decoder_layers=[32] for {path}")
            config = CONFIG.copy()
            config['decoder_layers'] = [32]
            model = SDA_Dual_Model(config).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            return model
        else:
            raise e

def load_test_data(subject, trial, scaler_path):
    # Load Scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # Load Data
    data = []
    labels = []
    
    with h5py.File(CONFIG['data_path'], 'r') as f:
        # Construct key (e.g., S003_02)
        # Assuming trial passes as "T002", we map it to "S003_02" ?
        # Wait, get_target_trials in sda_training returns keys like "S003_01", "S003_02"
        # The folder name uses T002 (likely corresponding to index or key)
        # Let's assume input trial is "T002", we need to find "S003_02"
        # Actually sda_training.py: trials.append(key) -> keys are "S003_01"
        # The result folder: "Fold_1_S003_T002" -> likely "T" + last 3 digits or just matched?
        # Let's verify standard naming. "S003_02" in H5 might correspond to T002.
        
        # Trial format: "T002" -> "S003_02"
        trial_suffix = trial[1:] # "002"
        # Try to find matching key
        key_candidate = f"{subject}_{trial_suffix[-2:]}" # "S003_02"
        
        if key_candidate not in f:
             print(f"Warning: Key {key_candidate} not in H5. Trying all keys.")
             found = False
             for k in f.keys():
                 if k.endswith(trial_suffix[-2:]) and subject in k:
                     key_candidate = k
                     found = True
                     break
             if not found:
                 raise ValueError(f"Could not find data for {subject} {trial}")

        print(f"Loading data from Key: {key_candidate}")
        grp = f[key_candidate]
        pfx = 'paretic'
        
        try:
            acc_x = grp[f'{pfx}_acc_x'][:][0]
            acc_y = grp[f'{pfx}_acc_y'][:][0]
            acc_z = grp[f'{pfx}_acc_z'][:][0]
            gyr_x = grp[f'{pfx}_gyr_x'][:][0]
            gyr_y = grp[f'{pfx}_gyr_y'][:][0]
            gyr_z = grp[f'{pfx}_gyr_z'][:][0]
            
            # Check feature set - default to 'angle' if not specified, but check sda_training logic
            # sda_training defaults to angle/angleV unless feature_set=='theta'
            # We will stick to angle/angleV as default for now unless we see 'theta' in folder name?
            # User didn't specify, but I should probably check what was trained. 
            # Assuming standard features:
            angle = grp[f'{pfx}_hip_angle'][:][0]
            angleV = grp[f'{pfx}_hip_angleV'][:][0]
            features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, angle, angleV], axis=1)
            
            label_raw = grp['gc_hs'][:][0] # 0-1
            phase = label_raw * 2 * np.pi 
            labels_2d = np.stack([np.cos(phase), np.sin(phase)], axis=1)
            
            # Windowing
            # Test data is usually processed sequentially without overlap? 
            # sda_training uses stride_tgt (5) for training. For test, usually stride=1 or same?
            # sda_training: test_loader uses SAME sliding window logic.
            # load_target_data_by_trials uses config['stride_tgt'] (5).
            # For visualization, we want continuous stream?
            # If we want to recreate the time series, stride=1 is best, but model was trained on slices.
            # However, sda_training test_loader uses GaitDataset which uses the Windows created by load_target_data_by_trials.
            # IF I want to plot the continuous phase, I should probably use stride=1 for inference to get nice smooth partial overlaps, 
            # OR just take the last point of each window if stride=1.
            # Let's use stride=CONFIG['stride_tgt'] to match "Test MSE" evaluation exactness,
            # BUT for "Phase Portrait" and "Gait Phase plot" (continuous), stride=5 might be gaps?
            # stride=5 means we skip 4 samples (40ms). 
            # Let's stick to stride=1 for better visualization resolution, 
            # knowing the model might only care about window context.
            stride_vis = 1 
            win_size = CONFIG['window_size']
            
            for i in range(0, len(features) - win_size + 1, stride_vis):
                window = features[i : i + win_size]
                label = labels_2d[i + win_size - 1] # Label is at the END of window
                data.append(window)
                labels.append(label)
                
        except KeyError as e:
            print(f"KeyError loading data: {e}")
            return None, None

    data = np.array(data)
    labels = np.array(labels)
    
    # Normalize
    # Shape (N, T, F) -> Flat -> Transform -> Reshape
    N, T, F = data.shape
    data_flat = data.reshape(-1, F)
    data_norm_flat = scaler.transform(data_flat)
    data_norm = data_norm_flat.reshape(N, T, F)
    
    # Convert to Tensor
    return torch.tensor(data_norm, dtype=torch.float32), labels

def compute_phase(predictions):
    # predictions: (N, 2) -> cos, sin
    # phase = atan2(sin, cos)
    # Map to 0-100%
    # atan2 returns -pi to pi
    cos = predictions[:, 0]
    sin = predictions[:, 1]
    phase_rad = np.arctan2(sin, cos) # -pi to pi
    
    # Adjust to 0-2pi (0-1)
    # usually gait phase 0% is HS. 
    # gt was created as cos(2*pi*phase), sin(2*pi*phase)
    # so phase = atan2(...) should be correct.
    # wraparound: -pi -> pi
    # if phase < 0: phase += 2pi
    
    phase_rad = np.mod(phase_rad, 2*np.pi)
    phase_pct = (phase_rad / (2*np.pi)) * 100.0
    return phase_pct

def run_comparison(subject='S003', trial='T002'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths to models
    # Base: SO (Source Only)
    # Proposed: SDA (Supervised Domain Adaptation)
    # Need to find the folder. Assumes standard naming structure from sda_training.
    # "Fold_X_Subject_Trial_Mode_frac1.0"
    
    base_dir = None
    sda_dir = None
    
    sub_res_dir = os.path.join(CONFIG['results_dir'], subject)
    for d in os.listdir(sub_res_dir):
        if trial in d and 'frac1.0' in d:
            if '_SO_' in d:
                base_dir = os.path.join(sub_res_dir, d)
            elif '_SDA_' in d:
                sda_dir = os.path.join(sub_res_dir, d)
    
    if not base_dir or not sda_dir:
        print("Could not find result directories for SO and SDA.")
        print(f"Looking in {sub_res_dir} for {trial} ...")
        return

    print(f"Base Dir: {base_dir}")
    print(f"SDA Dir: {sda_dir}")
    
    # Load Scaler (Use SDA scaler? or SO scaler? Should be same method 'fixed' usually)
    # sda_training uses 'fixed' normalization by default which fits on Source.
    # So the scaler should be identical. Let's load from SDA dir.
    scaler_path = os.path.join(sda_dir, 'scaler.pkl')
    
    # Load Data
    print("Loading Test Data...")
    X, y_gt = load_test_data(subject, trial, scaler_path)
    X = X.to(device)
    
    # Load Models
    print("Loading Models...")
    model_base = load_model(os.path.join(base_dir, 'best_model.pth'), device)
    model_sda = load_model(os.path.join(sda_dir, 'best_model.pth'), device)
    
    # Inference
    print("Inference...")
    with torch.no_grad():
        # SO: Domain Source (Dec_src) ? Or Domain Target (Dec_tgt)?
        # In sda_training SO mode: "pred, _ = model(bx, domain='source')" during test!
        pred_base, _ = model_base(X, domain='source')
        
        # SDA: Domain Target
        pred_sda, _ = model_sda(X, domain='target')
        
    pred_base = pred_base.cpu().numpy()
    pred_sda = pred_sda.cpu().numpy()
    
    # Compute Phases
    phase_gt = compute_phase(y_gt)
    phase_base = compute_phase(pred_base)
    phase_sda = compute_phase(pred_sda)
    
    # --- Plotting ---
    print("Plotting...")
    
    # 1. Phase Portrait (Unit Circle)
    # Plot Sin vs Cos
    plt.figure(figsize=(6, 6))
    
    # Plot a subset of points to avoid clutter? Or full trajectory?
    # Let's plot line
    plt.plot(y_gt[:, 0], y_gt[:, 1], 'k--', label='Ground Truth', alpha=0.6)
    plt.plot(pred_base[:, 0], pred_base[:, 1], 'b-', label='Base (SO)', alpha=0.7)
    plt.plot(pred_sda[:, 0], pred_sda[:, 1], 'r-', label='Proposed (SDA)', alpha=0.7)
    
    plt.xlim([-1.5, 1.5]); plt.ylim([-1.5, 1.5])
    plt.xlabel('Cosine'); plt.ylabel('Sine')
    plt.title(f'Phase Portrait - {subject} {trial}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'analysis/phase_portrait_{subject}_{trial}.png')
    plt.close()

    # 2. Gait Phase (Time Series)
    plt.figure(figsize=(10, 5))
    # Pick a range (e.g. 0 to 500 samples)
    limit = 500 if len(phase_gt) > 500 else len(phase_gt)
    t = np.arange(limit)
    
    plt.plot(t, phase_gt[:limit], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(t, phase_base[:limit], 'b-', label='Base (SO)', alpha=0.8)
    plt.plot(t, phase_sda[:limit], 'r-', label='Proposed (SDA)', alpha=0.8)
    
    plt.ylabel('Gait Phase (%)')
    plt.xlabel('Time (samples)')
    plt.title(f'Gait Phase Estimation - {subject} {trial}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'analysis/gait_phase_{subject}_{trial}.png')
    plt.close()
    
    
    # 3. Cosine/Sine vs Time
    plt.figure(figsize=(10, 8))
    
    # Cosine
    plt.subplot(2, 1, 1)
    plt.plot(t, y_gt[:limit, 0], 'k--', label='GT Cos', alpha=0.6)
    plt.plot(t, pred_base[:limit, 0], 'b-', label='Base (SO)', alpha=0.8)
    plt.plot(t, pred_sda[:limit, 0], 'r-', label='Proposed (SDA)', alpha=0.8)
    plt.ylabel('Cosine')
    plt.title(f'Cosine Component - {subject} {trial}')
    plt.legend()
    plt.grid(True)
    
    # Sine
    plt.subplot(2, 1, 2)
    plt.plot(t, y_gt[:limit, 1], 'k--', label='GT Sin', alpha=0.6)
    plt.plot(t, pred_base[:limit, 1], 'b-', label='Base (SO)', alpha=0.8)
    plt.plot(t, pred_sda[:limit, 1], 'r-', label='Proposed (SDA)', alpha=0.8)
    plt.ylabel('Sine')
    plt.xlabel('Time (samples)')
    plt.title(f'Sine Component - {subject} {trial}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'analysis/components_time_{subject}_{trial}.png')
    plt.close()

    print("Done! Saved plots.")

if __name__ == "__main__":
    # Create analysis dir if not exists
    os.makedirs('analysis', exist_ok=True)
    
    # Run for S008 T001 as requested
    run_comparison('S008', 'T001')
