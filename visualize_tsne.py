import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import joblib
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data, normalize_fixed

# Configuration (Ensure it matches training)
CONFIG['results_dir'] = 'results_analysis/SDA_Effect'
CONFIG['target_subject'] = 'S003'
CONFIG['batch_size'] = 64
CONFIG['input_dim'] = 8
CONFIG['window_size'] = 100
CONFIG['stride_tgt'] = 5
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config for visualization
fold_idx = 1
test_trial = 'S003_T002'

def extract_features(model, data, domain='target'):
    """Extract latent features (z) from the model."""
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(CONFIG['device'])
    features = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            _, z = model(batch, domain=domain)
            features.append(z.cpu().numpy())
    return np.concatenate(features, axis=0)

def visualize_tsne():
    print("Visualizing SDA Effect via t-SNE...")
    
    # 1. Load Data
    print("Loading Data...")
    src_data, _ = load_source_data(CONFIG) # (N_s, T, F)
    
    target_trials = get_target_trials(CONFIG)
    # We use Target Test Data for visualization (unseen data)
    _, _ = load_target_data_by_trials(CONFIG, [t for t in target_trials if t != test_trial]) # train dummy
    tgt_test_data, _ = load_target_data_by_trials(CONFIG, [test_trial])
    
    # Normalize (Fixed - Source Norm)
    # Note: We must use the SAME scaler as training.
    # Ideally, load the scaler from the result dir.
    # Let's try to load scaler from SDA folder (it should be same as SO if using fixed norm)
    sda_dir = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'], f"Fold_{fold_idx}_{test_trial}_SDA_frac1.0")
    if os.path.exists(os.path.join(sda_dir, 'scaler.pkl')):
        scaler = joblib.load(os.path.join(sda_dir, 'scaler.pkl'))
        # Transform
        N_s, T, F = src_data.shape
        src_norm = scaler.transform(src_data.reshape(-1, F)).reshape(N_s, T, F)
        
        N_t, T, F = tgt_test_data.shape
        tgt_norm = scaler.transform(tgt_test_data.reshape(-1, F)).reshape(N_t, T, F)
    else:
        print("Scaler not found, modifying manually (Risk of mismatch!)")
        # Fallback: re-fit
        _, _, _, scaler = normalize_fixed(src_data, tgt_test_data, tgt_test_data) # dummy
        N_s, T, F = src_data.shape
        src_norm = scaler.transform(src_data.reshape(-1, F)).reshape(N_s, T, F)
        N_t, T, F = tgt_test_data.shape
        tgt_norm = scaler.transform(tgt_test_data.reshape(-1, F)).reshape(N_t, T, F)

    # Subsample Source for t-SNE (Source is huge)
    # Take random 1000 samples from Source, and use all Target Test (usually small, ~1000)
    np.random.seed(42)
    idx_s = np.random.choice(len(src_norm), min(1000, len(src_norm)), replace=False)
    src_sample = src_norm[idx_s]
    tgt_sample = tgt_norm 
    
    # 2. Before Adaptation (Source Only Model)
    print("Extracting features from SO model...")
    so_dir = os.path.join(CONFIG['results_dir'], CONFIG['target_subject'], f"Fold_{fold_idx}_{test_trial}_SO_frac1.0")
    model_so = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    try:
        model_so.load_state_dict(torch.load(os.path.join(so_dir, 'best_model.pth')))
    except FileNotFoundError:
        print("SO Model not found! Using random init (Validation only)")
    
    z_s_so = extract_features(model_so, src_sample, domain='source')
    z_t_so = extract_features(model_so, tgt_sample, domain='source') # SO treats target as source-domain input
    
    # 3. After Adaptation (SDA Model)
    print("Extracting features from SDA model...")
    # sda_dir defined above
    model_sda = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
    try:
        model_sda.load_state_dict(torch.load(os.path.join(sda_dir, 'best_model.pth')))
    except FileNotFoundError:
        print("SDA Model not found! Using random init")

    z_s_sda = extract_features(model_sda, src_sample, domain='source')
    z_t_sda = extract_features(model_sda, tgt_sample, domain='target') # SDA encodes target with shared encoder (which is domain-agnostic ideally)
    
    # 4. t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Combined for Before
    X_so = np.concatenate([z_s_so, z_t_so], axis=0)
    y_so = np.concatenate([np.zeros(len(z_s_so)), np.ones(len(z_t_so))], axis=0) # 0: Source, 1: Target
    emb_so = tsne.fit_transform(X_so)
    
    # Combined for After
    X_sda = np.concatenate([z_s_sda, z_t_sda], axis=0)
    y_sda = y_so
    emb_sda = tsne.fit_transform(X_sda)
    
    # 5. Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(emb_so[y_so==0, 0], emb_so[y_so==0, 1], c='blue', alpha=0.5, label='Source', s=10)
    plt.scatter(emb_so[y_so==1, 0], emb_so[y_so==1, 1], c='red', alpha=0.5, label='Target', s=10)
    plt.title("Before SDA (Source Only)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(emb_sda[y_sda==0, 0], emb_sda[y_sda==0, 1], c='blue', alpha=0.5, label='Source', s=10)
    plt.scatter(emb_sda[y_sda==1, 0], emb_sda[y_sda==1, 1], c='red', alpha=0.5, label='Target', s=10)
    plt.title("After SDA (Domain Adaptation)")
    plt.legend()
    
    save_path = "tsne_sda_effect.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved t-SNE plot to {save_path}")

if __name__ == "__main__":
    visualize_tsne()
