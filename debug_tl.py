
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sda_training import SDA_Dual_Model, CONFIG, load_source_data, load_target_data_by_trials, get_target_trials, GaitDataset, normalize_fixed
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib

def debug_tl_init():
    fold_idx = 1
    subj = 'S003'
    CONFIG['target_subject'] = subj
    target_trials = get_target_trials(CONFIG)
    test_trial = target_trials[fold_idx-1] # First one
    
    print(f"Debugging TL Init for {subj} Fold {fold_idx} Trial {test_trial}")
    
    # Load SO Path
    so_dir = os.path.join(CONFIG['results_dir'], subj, f"Fold_{fold_idx}_{test_trial}_SO_frac1.0")
    so_path = os.path.join(so_dir, 'best_model.pth')
    
    if not os.path.exists(so_path):
        print(f"CRITICAL: SO Path {so_path} does not exist!")
        return 
        
    print("SO Path Found.")
    
    # Load Model
    model = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    checkpoint = torch.load(so_path)
    model.load_state_dict(checkpoint)
    print("Loaded SO weights into Model.")
    
    # Check SO Performance (Decoder Src)
    # We need data...
    src_data, src_labels = load_source_data(CONFIG)
    tgt_train_data, tgt_train_labels = load_target_data_by_trials(CONFIG, [t for t in target_trials if t != test_trial])
    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
    
    src_norm, tgt_train_norm, tgt_test_norm, scaler = normalize_fixed(src_data, tgt_train_data, tgt_test_data)
    
    test_loader = DataLoader(GaitDataset(tgt_test_norm, tgt_test_labels), batch_size=CONFIG['batch_size'])
    
    criterion = nn.MSELoss()
    model.eval()
    
    # Eval Decoder Src (SO Mode)
    mse_src = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            pred, _ = model(x, domain='source')
            mse_src += criterion(pred, y).item()
    print(f"SO Performance (Dec Src): {mse_src/len(test_loader):.4f}")
    
    # Eval Decoder Tgt (Before Copy) -> Should be random
    mse_tgt_bad = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            pred, _ = model(x, domain='target')
            mse_tgt_bad += criterion(pred, y).item()
    print(f"Dec Tgt (Random): {mse_tgt_bad/len(test_loader):.4f}")
    
    # COPY
    model.decoder_tgt.load_state_dict(model.decoder_src.state_dict())
    print("Copied Src -> Tgt")
    
    # Eval Decoder Tgt (After Copy) -> Should match SO
    mse_tgt_good = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            pred, _ = model(x, domain='target')
            mse_tgt_good += criterion(pred, y).item()
    print(f"Dec Tgt (After Copy): {mse_tgt_good/len(test_loader):.4f}")

if __name__ == "__main__":
    debug_tl_init()
