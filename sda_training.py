
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# --- Configuration ---
CONFIG = {
    'source_h5': 'd:/RSC lab/Codes/Stroke_SDA/output/all_subjects_dataset_ds_hip.h5',
    'target_h5': 'd:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5',
    'target_subject': 'S003', 
    'window_size': 50,    
    'stride': 5,          # User requested 5 (More Source Data)
    'stride_tgt': 5,      # User requested 5 (Less Target Data -> Natural Scarcity)
    'batch_size': 64,     
    'learning_rate': 0.0001,
    'epochs': 50,          # User requested 10 with Infinite Loading
    'lambda_mmd': 0.5,     # User requested 1.5 (3.0 diverged) -> Grid Search found 0.5 best
    'lambda_src': 0.3,    
    'lambda_tgt': 1.0,    
    'input_dim': 8,
    'hidden_dim': 64,
    'encoder_layers': [128, 64], 
    'decoder_layers': [128],      # Grid Search found 128 best (was 32)
    'dropout': 0.3,               # Default 0.3
    'output_dim': 2,        
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'results_dir': 'results',
    'patience': 5,        # Early Stopping Patience
    'min_delta': 0.0005,   # Min improvements
    'data_fraction': 1.0   # Back to 100% Data
}

# --- Config Override for Analysis ---
config_file = os.environ.get('SDA_CONFIG_FILE', 'config_override.json')
if os.path.exists(config_file):
    import json
    with open(config_file, 'r') as f:
        override = json.load(f)
        print(f"!!! CONFIG OVERRIDE LOADED from {config_file}: {override} !!!")
        CONFIG.update(override)

# --- 1. Data Loading Helper ---

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pth', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

class GaitDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def load_source_data(config):
    # Load Source (Healthy) - All subjects
    print("Loading Source Data (All Healthy)...")
    src_data = [] 
    src_labels = []
    
    with h5py.File(config['source_h5'], 'r') as f:
        for key in f.keys():
            grp = f[key]
            if 'walking_speed' in grp:
                speed = np.mean(grp['walking_speed'])
                if speed > 0.7: continue
            
            try:
                acc_x = grp['thigh_acc_x'][:][0]
                acc_y = grp['thigh_acc_y'][:][0]
                acc_z = grp['thigh_acc_z'][:][0]
                gyr_x = grp['thigh_gyr_x'][:][0]
                gyr_y = grp['thigh_gyr_y'][:][0]
                gyr_z = grp['thigh_gyr_z'][:][0]
                if config.get('feature_set') == 'theta':
                    # Use theta_est and its derivative
                    # theta_est is (1, T) -> [0]
                    # numerical derivative with dt=0.01
                    th = grp['theta_est'][:][0]
                    
                    # Compute Derivative
                    # Simple gradient or diff? User said "numerical derivative (dt=10ms)"
                    # np.gradient is good for centered diff.
                    th_vel = np.gradient(th, 0.01)
                    
                    features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, th, th_vel], axis=1)
                else:
                    # Original: hip_angle, hip_angleV
                    angle = grp['hip_angle'][:][0]
                    angleV = grp['hip_angleV'][:][0]
                    features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, angle, angleV], axis=1)
                label_raw = grp['gcR_hs'][:][0]
                phase = label_raw * 0.01 * 2 * np.pi 
                labels_2d = np.stack([np.cos(phase), np.sin(phase)], axis=1)
                
                if len(features) < config['window_size']: continue

                for i in range(0, len(features) - config['window_size'] + 1, config['stride']):
                    window = features[i : i + config['window_size']]
                    label = labels_2d[i + config['window_size'] - 1]
                    src_data.append(window)
                    src_labels.append(label)
            except KeyError: continue
    
    return np.array(src_data), np.array(src_labels)

def get_target_trials(config):
    target_subj = config['target_subject']
    trials = []
    with h5py.File(config['target_h5'], 'r') as f:
        for key in f.keys():
            if key.startswith(target_subj):
                trials.append(key)
    trials.sort()
    return trials

def load_target_data_by_trials(config, trial_keys):
    data = []
    labels = []
    
    with h5py.File(config['target_h5'], 'r') as f:
        for key in trial_keys:
            if key not in f: continue
            grp = f[key]
            pfx = 'paretic'
            try:
                acc_x = grp[f'{pfx}_acc_x'][:][0]
                acc_y = grp[f'{pfx}_acc_y'][:][0]
                acc_z = grp[f'{pfx}_acc_z'][:][0]
                gyr_x = grp[f'{pfx}_gyr_x'][:][0]
                gyr_y = grp[f'{pfx}_gyr_y'][:][0]
                gyr_z = grp[f'{pfx}_gyr_z'][:][0]
                if config.get('feature_set') == 'theta':
                   # Use paretic_theta_est and its derivative
                   p_th = grp[f'{pfx}_theta_est'][:][0]
                   p_th_vel = np.gradient(p_th, 0.01)
                   features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, p_th, p_th_vel], axis=1)
                else:
                    angle = grp[f'{pfx}_hip_angle'][:][0]
                    angleV = grp[f'{pfx}_hip_angleV'][:][0]
                    features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, angle, angleV], axis=1)
                label_raw = grp['gc_hs'][:][0]
                phase = label_raw * 2 * np.pi 
                labels_2d = np.stack([np.cos(phase), np.sin(phase)], axis=1)
                
                if len(features) < config['window_size']: continue
                
                for i in range(0, len(features) - config['window_size'] + 1, config['stride_tgt']):
                    window = features[i : i + config['window_size']]
                    label = labels_2d[i + config['window_size'] - 1]
                    data.append(window)
                    labels.append(label)
            except KeyError: continue
            
    return np.array(data), np.array(labels)

def normalize_fixed(src_data, tgt_train, tgt_test):
    scaler = StandardScaler()
    N_s, T, F = src_data.shape
    src_flat = src_data.reshape(-1, F)
    scaler.fit(src_flat)
    
    src_norm = scaler.transform(src_flat).reshape(N_s, T, F)
    tgt_train_norm = scaler.transform(tgt_train.reshape(-1, F)).reshape(tgt_train.shape[0], T, F)
    
    if len(tgt_test) > 0:
        tgt_test_norm = scaler.transform(tgt_test.reshape(-1, F)).reshape(tgt_test.shape[0], T, F)
    else:
        tgt_test_norm = np.empty((0, T, F))
        
    return src_norm, tgt_train_norm, tgt_test_norm, scaler

def normalize_independent(src_data, tgt_train, tgt_test):
    """Normalize Source and Target independently (User's approach)"""
    # 1. Source Norm
    scaler_src = StandardScaler()
    N_s, T, F = src_data.shape
    src_flat = src_data.reshape(-1, F)
    scaler_src.fit(src_flat)
    src_norm = scaler_src.transform(src_flat).reshape(N_s, T, F)
    
    # 2. Target Norm (Fit on Target Train)
    scaler_tgt = StandardScaler()
    N_t, T, F = tgt_train.shape
    tgt_train_flat = tgt_train.reshape(-1, F)
    scaler_tgt.fit(tgt_train_flat) # Fit on Target Train!
    tgt_train_norm = scaler_tgt.transform(tgt_train_flat).reshape(N_t, T, F)
    
    if len(tgt_test) > 0:
        # Transform Test using Target Train scaler
        tgt_test_norm = scaler_tgt.transform(tgt_test.reshape(-1, F)).reshape(tgt_test.shape[0], T, F)
    else:
        tgt_test_norm = np.empty((0, T, F))
        
    return src_norm, tgt_train_norm, tgt_test_norm, scaler_tgt

# --- 3. LOTO CV Logic ---


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dims[1]) # Normalization for MMD stability

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        _, (h_n, _) = self.lstm2(out) 
        z = h_n[-1]
        z = self.norm(z) # Normalize feature
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

def mmd_loss_multiscale(x, y):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    x_sq = xx.diag().unsqueeze(1) # (N, 1)
    y_sq = yy.diag().unsqueeze(0) # (1, M)
    
    dxx = x_sq + x_sq.t() - 2.*xx
    dyy = y_sq.t() + y_sq - 2.*yy
    dxy = x_sq + y_sq - 2.*xy
    
    # Robust Median Bandwidth
    dist_cat = torch.cat([dxx.view(-1), dyy.view(-1), dxy.view(-1)], dim=0)
    dist_cat = dist_cat[dist_cat > 0]
    
    if len(dist_cat) > 0:
        bandwidth = torch.median(dist_cat).detach() # Detach is critical
    else:
        bandwidth = torch.tensor(1.0).to(x.device)
        
    loss = 0
    scales = [0.1, 0.5, 1.0, 2.0, 10.0] # Wider range
    for s in scales: 
        bw = bandwidth * s
        loss += torch.exp( -dxx / (bw + 1e-8)).mean() + torch.exp( -dyy / (bw + 1e-8)).mean() - 2*torch.exp( -dxy / (bw + 1e-8)).mean()
    return loss

# --- 3. LOTO CV Logic ---

def train_and_evaluate(fold_idx, train_trials, test_trial, src_data, src_labels, mode='SDA', normalization='fixed'):
    print(f"\n[Fold {fold_idx} - {mode}] Train: {train_trials}, Test: {test_trial}, Norm: {normalization}")
    
    # Setup Output Dir
    suffix = "_IndepNorm" if normalization == 'independent' else ""
    out_dir = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'], f"Fold_{fold_idx}_{test_trial}_{mode}_frac{CONFIG['data_fraction']}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Data
    tgt_train_data, tgt_train_labels = load_target_data_by_trials(CONFIG, train_trials)
    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
    
    # Validation Split (70/30) as requested
    if len(tgt_train_data) > 0:
        tgt_train_data, tgt_val_data, tgt_train_labels, tgt_val_labels = train_test_split(
            tgt_train_data, tgt_train_labels, test_size=0.3, random_state=42, shuffle=True
        )
        
        # --- Data Subsampling (Data Efficiency Experiment) ---
        if CONFIG['data_fraction'] < 1.0:
            n_samples = int(len(tgt_train_data) * CONFIG['data_fraction'])
            print(f"  [Subsampling] Reducing Target Train {len(tgt_train_data)} -> {n_samples} ({CONFIG['data_fraction']*100}%)")
            # We already shuffled in split, so just take the first N
            tgt_train_data = tgt_train_data[:n_samples]
            tgt_train_labels = tgt_train_labels[:n_samples]
            
    else:
        tgt_val_data, tgt_val_labels = np.empty((0, CONFIG['window_size'], CONFIG['input_dim'])), np.empty((0, CONFIG['output_dim']))
    
    # Normalize
    if normalization == 'independent':
         src_norm, tgt_train_norm, tgt_test_norm, scaler = normalize_independent(src_data, tgt_train_data, tgt_test_data)
         if len(tgt_val_data) > 0:
            # Transform Val using the same scaler (Target Train scaler)
            tgt_val_norm = scaler.transform(tgt_val_data.reshape(-1, CONFIG['input_dim'])).reshape(tgt_val_data.shape)
         else: tgt_val_norm = np.empty((0, CONFIG['window_size'], CONFIG['input_dim']))
         
    else: # Fixed (Global Source Scaler)
        if mode == 'TO': # Target Only - Fit scaler only on Target Train? Or Global?
            src_norm, tgt_train_norm, tgt_test_norm, scaler = normalize_fixed(src_data, tgt_train_data, tgt_test_data)
            if len(tgt_val_data) > 0:
                tgt_val_norm = scaler.transform(tgt_val_data.reshape(-1, CONFIG['input_dim'])).reshape(tgt_val_data.shape)
            else: tgt_val_norm = np.empty((0, CONFIG['window_size'], CONFIG['input_dim']))
        else:
            src_norm, tgt_train_norm, tgt_test_norm, scaler = normalize_fixed(src_data, tgt_train_data, tgt_test_data)
            if len(tgt_val_data) > 0:
                tgt_val_norm = scaler.transform(tgt_val_data.reshape(-1, CONFIG['input_dim'])).reshape(tgt_val_data.shape)
            else: tgt_val_norm = np.empty((0, CONFIG['window_size'], CONFIG['input_dim']))
        
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    
    # Dataloaders
    src_loader = DataLoader(GaitDataset(src_norm, src_labels), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    if len(tgt_train_norm) > 0:
        tgt_loader = DataLoader(GaitDataset(tgt_train_norm, tgt_train_labels), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        if len(tgt_val_norm) > 0:
            val_loader = DataLoader(GaitDataset(tgt_val_norm, tgt_val_labels), batch_size=CONFIG['batch_size'], shuffle=False)
        else: val_loader = None
    else:
        tgt_loader = None 
        val_loader = None
        
    # Source Validation Split (Global for SDA and SO now)
    if mode in ['SDA', 'SO']:
        # Split Source into Train/Val (80/20)
        s_train_x, s_val_x, s_train_y, s_val_y = train_test_split(src_norm, src_labels, test_size=0.2, random_state=42, shuffle=True)
        # Re-create src_loader for training
        src_loader = DataLoader(GaitDataset(s_train_x, s_train_y), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        # Create src_val_loader
        src_val_loader = DataLoader(GaitDataset(s_val_x, s_val_y), batch_size=CONFIG['batch_size'], shuffle=False)
    else:
        src_val_loader = None    
        
    test_loader = DataLoader(GaitDataset(tgt_test_norm, tgt_test_labels), batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Model & Opt
    model = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    criterion = nn.MSELoss()
    
    # Transfer Learning (TL) Special Logic: Load Pre-trained SO Model
    if mode == 'TL':
        # Construct path to the corresponding SO model (assuming frac1.0 for Source training)
        # Note: We rely on the folder naming structure: Fold_{fold_idx}_{test_trial}_SO_frac1.0
        # If config is different, this might fail, but for this context it's correct.
        so_dir = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'], f"Fold_{fold_idx}_{test_trial}_SO_frac1.0")
        so_path = os.path.join(so_dir, 'best_model.pth')
        
        if os.path.exists(so_path):
            print(f"  [TL] Loading Pre-trained SO Model from: {so_path}")
            # Load weights
            # Note: SO model has same architecture (SDA_Dual_Model)
            checkpoint = torch.load(so_path)
            model.load_state_dict(checkpoint)
        else:
            print(f"  [TL] WARNING: SO Model not found at {so_path}. Falling back to random init (NOT PRE-TRAINED).")
        
        print("  [TL] Freezing Encoder and initializing Decoder Tgt...")
        # Copy weights: Src Decoder -> Tgt Decoder (Transfer Knowledge)
        model.decoder_tgt.load_state_dict(model.decoder_src.state_dict())
        
        # Freeze Encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # Optimizer for Fine-Tuning (Decoder Tgt Only)
        opt_enc = None 
        opt_dec_src = None 
        opt_dec_tgt = optim.Adam(model.decoder_tgt.parameters(), lr=CONFIG['learning_rate'])
        
    else:
        opt_enc = optim.Adam(model.encoder.parameters(), lr=CONFIG['learning_rate'])
        opt_dec_src = optim.Adam(model.decoder_src.parameters(), lr=CONFIG['learning_rate'])
        opt_dec_tgt = optim.Adam(model.decoder_tgt.parameters(), lr=CONFIG['learning_rate'])
    
    # Early Stopping
    stopper = EarlyStopping(patience=CONFIG['patience'], min_delta=CONFIG['min_delta'], 
                            path=os.path.join(out_dir, 'best_model.pth'), verbose=True)
    
    # Early Stopping
    stopper = EarlyStopping(patience=CONFIG['patience'], min_delta=CONFIG['min_delta'], 
                            path=os.path.join(out_dir, 'best_model.pth'), verbose=True)
    
    # History
    history = {'train_loss': [], 'val_mse': [], 'test_mse': [], 'loss_s': [], 'loss_t': [], 'loss_mmd': []}
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0; total_s = 0; total_t = 0; total_mmd = 0
        
        # --- Mode Switching ---
        if mode == 'SDA':
            # Infinite Loading: Loop over Source (Large), cycle Target (Small)
            num_batches = len(src_loader)
            
            src_iter = iter(src_loader)
            tgt_iter = iter(tgt_loader)
            
            for _ in range(num_batches):
                try: batch_s = next(src_iter)
                except StopIteration: break
                
                try: batch_t = next(tgt_iter)
                except StopIteration: 
                    tgt_iter = iter(tgt_loader)
                    batch_t = next(tgt_iter)
                
                xs, ys = batch_s[0].to(CONFIG['device']), batch_s[1].to(CONFIG['device'])
                xt, yt = batch_t[0].to(CONFIG['device']), batch_t[1].to(CONFIG['device'])
                
                opt_enc.zero_grad(); opt_dec_src.zero_grad(); opt_dec_tgt.zero_grad()
                pred_s, z_s = model(xs, domain='source')
                loss_s = criterion(pred_s, ys)
                pred_t, z_t = model(xt, domain='target')
                loss_t = criterion(pred_t, yt)
                loss_mmd = mmd_loss_multiscale(z_s, z_t)
                loss = (CONFIG['lambda_tgt']*loss_t) + (CONFIG['lambda_src']*loss_s) + (CONFIG['lambda_mmd']*loss_mmd)
                loss.backward()
                opt_enc.step(); opt_dec_src.step(); opt_dec_tgt.step()
                
                total_loss += loss.item(); total_s += loss_s.item(); total_t += loss_t.item(); total_mmd += loss_mmd.item()
                
        elif mode == 'SO': # Source Only
            # Use ALL source data (Original Logic)
            num_batches = len(src_loader)
            src_iter = iter(src_loader)
            
            for _ in range(num_batches):
                try: batch_s = next(src_iter)
                except StopIteration: break
                
                xs, ys = batch_s[0].to(CONFIG['device']), batch_s[1].to(CONFIG['device'])
                opt_enc.zero_grad(); opt_dec_src.zero_grad()
                pred_s, _ = model(xs, domain='source')
                loss = criterion(pred_s, ys)
                loss.backward()
                opt_enc.step(); opt_dec_src.step()
                total_loss += loss.item(); total_s += loss.item()
                
        elif mode == 'TO': # Target Only
            # Train only on Target, minimize Target Loss
            if tgt_loader is None or len(tgt_loader) == 0:
                print("Error: No Target Train Data for TO mode.")
                break
            num_batches = len(tgt_loader) # Finite loop for Target
            for batch_t in tgt_loader:
                xt, yt = batch_t[0].to(CONFIG['device']), batch_t[1].to(CONFIG['device'])
                opt_enc.zero_grad(); opt_dec_tgt.zero_grad() # No Source Decoder update
                pred_t, _ = model(xt, domain='target')
                loss = criterion(pred_t, yt)
                loss.backward()
                opt_enc.step(); opt_dec_tgt.step()
                total_loss += loss.item(); total_t += loss.item()

        elif mode == 'TL': # Transfer Learning (Target Fine Tuning)
            if tgt_loader is None or len(tgt_loader) == 0: break
            num_batches = len(tgt_loader)
            for batch_t in tgt_loader:
                xt, yt = batch_t[0].to(CONFIG['device']), batch_t[1].to(CONFIG['device'])
                opt_dec_tgt.zero_grad() # Encoder is frozen, only Decoder Tgt
                pred_t, _ = model(xt, domain='target')
                loss = criterion(pred_t, yt)
                loss.backward()
                opt_dec_tgt.step()
                total_loss += loss.item(); total_t += loss.item()

        # Validation Eval
        val_mse = 0
        
        # SO Mode Validation (Validate on Source)
        if mode == 'SO' and src_val_loader:
            model.eval()
            total_v_loss = 0
            with torch.no_grad():
                for xv, yv in src_val_loader:
                    xv, yv = xv.to(CONFIG['device']), yv.to(CONFIG['device'])
                    # SO uses Source Decoder (conceptually) or Target?
                    # "SO" usually means "Train on Source, Validate on Source" to pick best Source model.
                    # Then Test on Target.
                    pred, _ = model(xv, domain='source')
                    total_v_loss += criterion(pred, yv).item()
            val_mse = total_v_loss / len(src_val_loader) if len(src_val_loader) else 0
            
            
        # SDA Mode Validation (Composite Loss)
        elif mode == 'SDA' and val_loader:
            model.eval()
            total_v_loss = 0
            with torch.no_grad():
                # We need to iterate both loaders. Zip them?
                # Handle different lengths by cycling or just zip (shortest). 
                # Validation should be deterministic. Let's use zip.
                # If src_val_loader is missing for some reason, fallback.
                if not src_val_loader: src_val_iter = []
                else: src_val_iter = iter(src_val_loader)
                
                # We iterate val_loader (Target)
                for xt, yt in val_loader:
                    xt, yt = xt.to(CONFIG['device']), yt.to(CONFIG['device'])
                    
                    try: 
                        batch_s = next(src_val_iter)
                    except StopIteration:
                        src_val_iter = iter(src_val_loader)
                        batch_s = next(src_val_iter)
                    
                    xs, ys = batch_s[0].to(CONFIG['device']), batch_s[1].to(CONFIG['device'])
                    
                    # Compute Component Losses
                    pred_t, z_t = model(xt, domain='target')
                    v_loss_t = criterion(pred_t, yt)
                    
                    pred_s, z_s = model(xs, domain='source')
                    v_loss_s = criterion(pred_s, ys)
                    
                    v_loss_mmd = mmd_loss_multiscale(z_s, z_t)
                    
                    # Weighted Sum
                    v_loss = (CONFIG['lambda_tgt']*v_loss_t) + (CONFIG['lambda_src']*v_loss_s) + (CONFIG['lambda_mmd']*v_loss_mmd)
                    total_v_loss += v_loss.item()
                    
            val_mse = total_v_loss / len(val_loader) if len(val_loader) else 0

        # TO Mode Validation (Target Only)
        elif mode in ['TO', 'TL'] and val_loader:
            model.eval()
            total_v_loss = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(CONFIG['device']), yv.to(CONFIG['device'])
                    pred, _ = model(xv, domain='target')
                    total_v_loss += criterion(pred, yv).item()
            val_mse = total_v_loss / len(val_loader) if len(val_loader) else 0

        # Test Eval
        model.eval()
        test_mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
                # Inference Domain Logic
                if mode == 'SO':
                     # SO: Use Source Decoder for Target Data
                    pred, _ = model(bx, domain='source')
                else:
                    # SDA/TO: Use Target Decoder
                    pred, _ = model(bx, domain='target') 
                test_mse += criterion(pred, by).item()
        test_mse /= len(test_loader)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        history['train_loss'].append(avg_loss)
        history['val_mse'].append(val_mse)
        history['test_mse'].append(test_mse)
        if mode == 'SDA':
             history['loss_s'].append(total_s/num_batches if num_batches else 0)
             history['loss_t'].append(total_t/num_batches if num_batches else 0)
             history['loss_mmd'].append(total_mmd/num_batches if num_batches else 0)
        
        print(f"  Ep {epoch+1}: Train {avg_loss:.4f} | Val {val_mse:.4f} | Test {test_mse:.4f}")

        # Check Early Stopping
        stopper(val_mse, model)
        if stopper.early_stop:
            print("Early stopping triggered!")
            break

    # Load Best Model for Final Eval/Save
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pth')))
    
    # Final Test Eval with Best Model
    model.eval()
    test_mse = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
            if mode == 'SO': pred, _ = model(bx, domain='source')
            else: pred, _ = model(bx, domain='target') 
            test_mse += criterion(pred, by).item()
    test_mse /= len(test_loader)
    print(f"Final Test MSE (Best Model): {test_mse:.4f}")

    # Save Model
    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model.pth'))
    
    # Save Result to Text File for Easy Aggregation
    with open(os.path.join(out_dir, 'final_mse.txt'), 'w') as f:
        f.write(str(test_mse))
    
    # Plot Loss Dynamics
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Total Loss', linewidth=2, color='black')
    if mode == 'SDA':
        plt.plot(history['loss_s'], label='Src', linestyle='--'); plt.plot(history['loss_t'], label='Tgt', linestyle='--'); plt.plot(history['loss_mmd'], label='MMD', linestyle='--')
    plt.plot(history['test_mse'], label='Test MSE', linewidth=2, color='red')
    plt.title(f'Loss Dynamics ({mode}) - Fold {fold_idx}')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'loss_dynamics.png'))
    plt.close()
    
    return test_mse # Return Best Model's Test MSE

def process_all_subjects():
    # Full Subject List (Excluding S002 as requested)
    subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    
    # Check if override limited the subject


    # Modes to run
    # Check if override limited the subject
    if 'target_subject' in CONFIG:
        print(f"!!! Restricting process_all_subjects to {CONFIG['target_subject']} due to override !!!")
        subjects = [CONFIG['target_subject']]

    # Modes to run (Default: SDA, TO, SO, TL)
    modes = CONFIG.get('modes', ['SDA', 'TO', 'SO', 'TL']) 
    
    print(f"Starting Full Experiment on Subjects: {subjects}")
    print(f"Modes: {modes}")
    
    print(f"\nChecking GPU Availability:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    
    src_data, src_labels = load_source_data(CONFIG)
    
    for subj in subjects:
        print(f"\n\n================================================")
        print(f"Processing Target Subject: {subj}")
        print(f"================================================")
        
        # Update Config for current subject
        CONFIG['target_subject'] = subj
        
        target_trials = get_target_trials(CONFIG)
        print(f"Trials: {target_trials}")
        
        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            for i, test_trial in enumerate(target_trials):
                # Max Folds Check
                if 'max_folds' in CONFIG and (i + 1) > CONFIG['max_folds']:
                    print(f"Skipping Fold {i+1} due to max_folds limit.")
                    continue

                # Check if result already exists (Skip if done to save time?)
                # Construct expected output dir
                out_dir = os.path.join(CONFIG['results_dir'], subj, f"Fold_{i+1}_{test_trial}_{mode}_frac{CONFIG['data_fraction']}")
                flag_file = os.path.join(out_dir, 'final_model.pth')
                if os.path.exists(os.path.join(out_dir, 'best_model.pth')):
                    print(f"  [Skipping] Model already exists at {out_dir}")
                    continue
                
                train_trials = [t for t in target_trials if t != test_trial]
                try:
                    train_and_evaluate(i+1, train_trials, test_trial, src_data, src_labels, mode=mode)
                    break # Run only the first fold for efficiency
                except Exception as e:
                    print(f"Error processing {subj} {mode} {test_trial}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    process_all_subjects()
