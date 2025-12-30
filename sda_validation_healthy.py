
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

# --- Configuration (Best Params) ---
CONFIG = {
    'source_h5': os.path.join('output', 'all_subjects_dataset_ds_hip.h5'),
    'target_subject': 'AB06', # Simulating AB06 as the "Stroke" patient
    'window_size': 50,    
    'stride': 5,          
    'stride_tgt': 5,      
    'batch_size': 64,     
    'learning_rate': 0.0001,
    'epochs': 50,          
    'lambda_mmd': 2.0,    # Best
    'lambda_src': 0.5,    # Best
    'lambda_tgt': 1.0,    
    'input_dim': 8,
    'hidden_dim': 64,
    'encoder_layers': [64, 32], # Adjusted based on Best Params (Enc=64)
    'decoder_layers': [128],    # Best Dec=128
    'dropout': 0.3,             
    'output_dim': 2,        
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'results_dir': 'results_validation_healthy',
    'patience': 10,        
    'min_delta': 0.0001,
    'data_fraction': 1.0,
    'feature_set': 'theta'
}

# --- Data Loading Helpers ---

class GaitDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def extract_features(grp, config):
    # Same logic as load_source_data
    acc_x = grp['thigh_acc_x'][:][0]
    acc_y = grp['thigh_acc_y'][:][0]
    acc_z = grp['thigh_acc_z'][:][0]
    gyr_x = grp['thigh_gyr_x'][:][0]
    gyr_y = grp['thigh_gyr_y'][:][0]
    gyr_z = grp['thigh_gyr_z'][:][0]
    
    if config.get('feature_set') == 'theta':
        th = grp['theta_est'][:][0]
        th_vel = np.gradient(th, 0.01)
        features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, th, th_vel], axis=1)
    else:
        angle = grp['hip_angle'][:][0]
        angleV = grp['hip_angleV'][:][0]
        features = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, angle, angleV], axis=1)
    
    label_raw = grp['gcR_hs'][:][0]
    phase = label_raw * 0.01 * 2 * np.pi 
    labels_2d = np.stack([np.cos(phase), np.sin(phase)], axis=1)
    
    return features, labels_2d

def load_source_data_exclude_target(config):
    # Load Source (Healthy) - All subjects EXCEPT target_subject
    print(f"Loading Source Data (All Healthy EXCEPT {config['target_subject']})...")
    src_data = [] 
    src_labels = []
    
    target_subj = config['target_subject']
    
    with h5py.File(config['source_h5'], 'r') as f:
        for key in f.keys():
            # Exclude Target Subject Data
            if key.startswith(target_subj): continue
            
            grp = f[key]
            if 'walking_speed' in grp:
                speed = np.mean(grp['walking_speed'])
                if speed > 0.7: continue
            
            try:
                features, labels_2d = extract_features(grp, config)
                
                if len(features) < config['window_size']: continue

                for i in range(0, len(features) - config['window_size'] + 1, config['stride']):
                    window = features[i : i + config['window_size']]
                    label = labels_2d[i + config['window_size'] - 1]
                    src_data.append(window)
                    src_labels.append(label)
            except KeyError: continue
    
    return np.array(src_data), np.array(src_labels)

def load_healthy_target_data(config):
    # Load ONLY Target Subject Data (Simulating Patient)
    print(f"Loading Target Data (Healthy {config['target_subject']} as Patient)...")
    data = [] 
    labels = []
    
    target_subj = config['target_subject']
    
    with h5py.File(config['source_h5'], 'r') as f:
        for key in f.keys():
            if not key.startswith(target_subj): continue # ONLY Target
            
            grp = f[key]
            # Speed filter? Maybe simulate stroke by picking slow speeds?
            # Let's keep all speeds for 'Healthy' validation
            
            try:
                features, labels_2d = extract_features(grp, config)
                
                if len(features) < config['window_size']: continue

                # Use stride_tgt
                for i in range(0, len(features) - config['window_size'] + 1, config['stride_tgt']):
                    window = features[i : i + config['window_size']]
                    label = labels_2d[i + config['window_size'] - 1]
                    data.append(window)
                    labels.append(label)
            except KeyError: continue
    
    return np.array(data), np.array(labels)

# --- Models ---
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
        # Ensure encoder_layers length matches logic
        if len(config['encoder_layers']) < 2: config['encoder_layers'] = [64, 32]
            
        self.encoder = Encoder(config['input_dim'], config['encoder_layers'], dropout=config.get('dropout', 0.3))
        self.decoder_src = Decoder(config['encoder_layers'][-1], config['decoder_layers'], config['output_dim'])
        self.decoder_tgt = Decoder(config['encoder_layers'][-1], config['decoder_layers'], config['output_dim'])
    def forward(self, x, domain='target'):
        z = self.encoder(x)
        if domain == 'source': out = self.decoder_src(z)
        else: out = self.decoder_tgt(z)
        return out, z

# --- MMD Loss ---
def mmd_loss_multiscale(x, y):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    x_sq = xx.diag().unsqueeze(1)
    y_sq = yy.diag().unsqueeze(0)
    
    dxx = x_sq + x_sq.t() - 2.*xx
    dyy = y_sq.t() + y_sq - 2.*yy
    dxy = x_sq + y_sq - 2.*xy
    
    dist_cat = torch.cat([dxx.view(-1), dyy.view(-1), dxy.view(-1)], dim=0)
    dist_cat = dist_cat[dist_cat > 0]
    
    if len(dist_cat) > 0:
        bandwidth = torch.median(dist_cat).detach()
    else:
        bandwidth = torch.tensor(1.0).to(x.device)
        
    loss = 0
    scales = [0.1, 0.5, 1.0, 2.0, 10.0]
    for s in scales: 
        bw = bandwidth * s
        loss += torch.exp( -dxx / (bw + 1e-8)).mean() + torch.exp( -dyy / (bw + 1e-8)).mean() - 2*torch.exp( -dxy / (bw + 1e-8)).mean()
    return loss

# --- Main Validation Logic ---
def run_validation():
    # 1. Load Data
    src_data, src_labels = load_source_data_exclude_target(CONFIG)
    tgt_data, tgt_labels = load_healthy_target_data(CONFIG)
    
    print(f"Source Data: {src_data.shape}")
    print(f"Target Data: {tgt_data.shape}")
    
    # 2. Split Target (Train/Test)
    # Since we are simulating, let's just do a random split 50/50
    tgt_train_data, tgt_test_data, tgt_train_labels, tgt_test_labels = train_test_split(
        tgt_data, tgt_labels, test_size=0.5, random_state=42, shuffle=True
    )
    # Further split Train into Train/Val
    tgt_train_data, tgt_val_data, tgt_train_labels, tgt_val_labels = train_test_split(
        tgt_train_data, tgt_train_labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 3. Normalize (Fixed)
    scaler = StandardScaler()
    N_s, T, F = src_data.shape
    src_flat = src_data.reshape(-1, F)
    scaler.fit(src_flat)
    
    src_norm = scaler.transform(src_flat).reshape(N_s, T, F)
    tgt_train_norm = scaler.transform(tgt_train_data.reshape(-1, F)).reshape(tgt_train_data.shape)
    tgt_val_norm = scaler.transform(tgt_val_data.reshape(-1, F)).reshape(tgt_val_data.shape)
    tgt_test_norm = scaler.transform(tgt_test_data.reshape(-1, F)).reshape(tgt_test_data.shape)
    
    # 4. Loaders
    batch_size = CONFIG['batch_size']
    src_loader = DataLoader(GaitDataset(src_norm, src_labels), batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(GaitDataset(tgt_train_norm, tgt_train_labels), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GaitDataset(tgt_val_norm, tgt_val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GaitDataset(tgt_test_norm, tgt_test_labels), batch_size=batch_size, shuffle=False)
    
    # 5. Model Setup
    model = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    opt_enc = optim.Adam(model.encoder.parameters(), lr=CONFIG['learning_rate'])
    opt_dec_src = optim.Adam(model.decoder_src.parameters(), lr=CONFIG['learning_rate'])
    opt_dec_tgt = optim.Adam(model.decoder_tgt.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    print("\nStarting Training (SDA Mode)...")
    best_loss = float('inf')
    
    # 6. Training Loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        num_batches = 100 # Fixed Iterations per epoch for test
        
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        
        total_loss = 0
        
        for _ in range(num_batches):
            try: batch_s = next(src_iter)
            except: src_iter = iter(src_loader); batch_s = next(src_iter)
            
            try: batch_t = next(tgt_iter)
            except: tgt_iter = iter(tgt_loader); batch_t = next(tgt_iter)
            
            xs, ys = batch_s[0].to(CONFIG['device']), batch_s[1].to(CONFIG['device'])
            xt, yt = batch_t[0].to(CONFIG['device']), batch_t[1].to(CONFIG['device'])
            
            opt_enc.zero_grad(); opt_dec_src.zero_grad(); opt_dec_tgt.zero_grad()
            
            p_s, z_s = model(xs, domain='source')
            l_s = criterion(p_s, ys)
            
            p_t, z_t = model(xt, domain='target')
            l_t = criterion(p_t, yt)
            
            l_mmd = mmd_loss_multiscale(z_s, z_t)
            
            loss = (CONFIG['lambda_tgt']*l_t) + (CONFIG['lambda_src']*l_s) + (CONFIG['lambda_mmd']*l_mmd)
            loss.backward()
            opt_enc.step(); opt_dec_src.step(); opt_dec_tgt.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(CONFIG['device']), vy.to(CONFIG['device'])
                vp, _ = model(vx, domain='target')
                val_mse += criterion(vp, vy).item()
        val_mse /= len(val_loader)
        
        print(f"Ep {epoch+1}: Train Loss {total_loss/num_batches:.4f} | Val MSE {val_mse:.4f}")
        
        if val_mse < best_loss:
            best_loss = val_mse
            torch.save(model.state_dict(), 'best_val_model.pth')
            
    # 7. Final Evaluation
    print("\nEvaluation on Test Set...")
    model.load_state_dict(torch.load('best_val_model.pth'))
    model.eval()
    
    preds_all = []
    targets_all = []
    
    with torch.no_grad():
        for tx, ty in test_loader:
            tx = tx.to(CONFIG['device'])
            tp, _ = model(tx, domain='target')
            preds_all.append(tp.cpu().numpy())
            targets_all.append(ty.numpy())
            
    preds = np.concatenate(preds_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    
    # Calculate MSE
    mse = np.mean((preds - targets)**2)
    print(f"Test MSE: {mse:.4f}")
    
    # Calculate Amplitude
    pred_amp = np.sqrt(preds[:,0]**2 + preds[:,1]**2)
    avg_amp = np.mean(pred_amp)
    print(f"Average Amplitude (Radius): {avg_amp:.4f} (Expected ~1.0)")
    
    # Calculate Phase RMSE
    # Atan2(sin, cos) -> Phase in radians (-pi, pi)
    # We predicted [cos, sin], so y=sin, x=cos
    pred_phase = np.arctan2(preds[:,1], preds[:,0])
    target_phase = np.arctan2(targets[:,1], targets[:,0])
    
    # Circular diff
    diff = pred_phase - target_phase
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    mse_rad = np.mean(diff**2)
    rmse_rad = np.sqrt(mse_rad)
    rmse_deg = np.degrees(rmse_rad)
    rmse_percent = (rmse_deg / 360.0) * 100
    
    print(f"Phase RMSE: {rmse_rad:.4f} rad")
    print(f"Phase RMSE: {rmse_deg:.2f} deg")
    print(f"Phase RMSE: {rmse_percent:.2f} %")
    
if __name__ == "__main__":
    run_validation()
