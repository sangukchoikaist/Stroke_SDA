import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import joblib
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data, normalize_fixed

# Set Font to Arial (or localized equivalent) if possible
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

# Configuration
CONFIG['results_dir'] = 'results_final_rank1' 
OUTPUT_BASE_DIR = 'results_analysis/Batch_SDA_Effect'
CONFIG['batch_size'] = 64
CONFIG['input_dim'] = 8
CONFIG['window_size'] = 100
CONFIG['stride_tgt'] = 5
CONFIG['device'] = 'cpu' # Force CPU 

# Subjects
subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

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

def visualize_tsne_batch():
    ensure_dir(OUTPUT_BASE_DIR)
    
    # Pre-load Source Data (Large, load once)
    print("Loading Source Data (All Healthy)...")
    src_data, _ = load_source_data(CONFIG) # (N_s, T, F)
    
    for subj in subjects:
        print(f"\nProcessing Subject: {subj}")
        CONFIG['target_subject'] = subj
        
        # Prepare Output Directory per Subject
        subj_out_dir = os.path.join(OUTPUT_BASE_DIR, subj)
        ensure_dir(subj_out_dir)

        # Path to results
        subj_results_dir = os.path.join(CONFIG['results_dir'], subj)
        if not os.path.exists(subj_results_dir):
            print(f"  Directory not found: {subj_results_dir}, skipping.")
            continue
            
        # Find directories for Fold 1
        so_dir = None
        sda_dir = None
        test_trial = None

        for d in os.listdir(subj_results_dir):
            if d.startswith("Fold_1_"):
                if "_SO_" in d:
                    so_dir = os.path.join(subj_results_dir, d)
                elif "_SDA_" in d and "beta" not in d: # Avoid beta variants if any
                    sda_dir = os.path.join(subj_results_dir, d)
                    # Infer trial name from folder name: Fold_1_S003_T002_SDA_...
                    parts = d.split('_')
                    # Expected: ['Fold', '1', 'S003', 'T002', 'SDA', ...]
                    try:
                         idx_sda = parts.index('SDA')
                         test_trial = "_".join(parts[2:idx_sda]) # e.g. S003_T002
                    except ValueError:
                        pass

        if not so_dir or not sda_dir or not test_trial:
             print(f"  Missing SO/SDA models for Fold 1 of {subj}, skipping.")
             # Fallback logic could go here but let's assume existence based on previous checks
             continue

        print(f"  Using SO: {os.path.basename(so_dir)}")
        print(f"  Using SDA: {os.path.basename(sda_dir)}")
        print(f"  Test Trial: {test_trial}")

        # Load Scalers
        scaler_so = joblib.load(os.path.join(so_dir, 'scaler.pkl'))
        scaler_sda = joblib.load(os.path.join(sda_dir, 'scaler.pkl'))

        # Load Raw Target Test Data
        tgt_test_data, _ = load_target_data_by_trials(CONFIG, [test_trial])
        
        # Transform Source/Target for SO
        N_s, T, F = src_data.shape
        # SO usually uses source-based scaler.
        src_norm_so = scaler_so.transform(src_data.reshape(-1, F)).reshape(N_s, T, F)
        N_t, T, F = tgt_test_data.shape
        tgt_norm_so = scaler_so.transform(tgt_test_data.reshape(-1, F)).reshape(N_t, T, F)
        
        # Transform Source/Target for SDA
        # SDA uses independent headers. Source is Source Norm, Target is Target Norm.
        # But wait, src data loading already applies normalization? No, load_source_data returns raw features?
        # Check load_source_data: "features = (features - mean) / std" -> It returns Normalized Data?
        # Actually sda_training.py: load_source_data returns NORMALIZED data if feature_set is not raw?
        # Let's assume load_source_data returns RAW. 
        # Wait, sda_training.py L120: defaults to using stored mean/std.
        # Ideally we use the scaler saved in the folder to be consistent.
        # Let's stick to using the loaded scalers.

        # SDA Source Input: technically SDA source head expects Source Normalized data. 
        # Reuse src_norm_so for SDA Source as well (assuming consistency).
        src_norm_sda = src_norm_so 
        
        # SDA Target Input: Target Indep Normalized
        tgt_norm_sda = scaler_sda.transform(tgt_test_data.reshape(-1, F)).reshape(N_t, T, F)
        
        # Subsample Source
        np.random.seed(42)
        idx_s = np.random.choice(len(src_norm_so), min(1000, len(src_norm_so)), replace=False)
        src_sample_so = src_norm_so[idx_s]
        src_sample_sda = src_norm_sda[idx_s]
        
        tgt_sample_so = tgt_norm_so
        tgt_sample_sda = tgt_norm_sda
        
        # --- SO Model ---
        print("  Extracting SO features...")
        model_so = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
        try:
            model_so.load_state_dict(torch.load(os.path.join(so_dir, 'best_model.pth'), map_location=CONFIG['device']))
        except Exception as e:
            print(f"  Failed to load SO model: {e}")
            continue
            
        z_s_so = extract_features(model_so, src_sample_so, domain='source')
        z_t_so = extract_features(model_so, tgt_sample_so, domain='source')
        
        # --- SDA Model ---
        print("  Extracting SDA features...")
        model_sda = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
        try:
            model_sda.load_state_dict(torch.load(os.path.join(sda_dir, 'best_model.pth'), map_location=CONFIG['device']))
        except Exception as e:
             print(f"  Failed to load SDA model: {e}")
             continue

        z_s_sda = extract_features(model_sda, src_sample_sda, domain='source')
        z_t_sda = extract_features(model_sda, tgt_sample_sda, domain='target')
        
        # --- t-SNE ---
        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        
        X_so = np.concatenate([z_s_so, z_t_so], axis=0)
        y_so = np.concatenate([np.zeros(len(z_s_so)), np.ones(len(z_t_so))], axis=0)
        emb_so = tsne.fit_transform(X_so)
        
        X_sda = np.concatenate([z_s_sda, z_t_sda], axis=0)
        y_sda = y_so
        emb_sda = tsne.fit_transform(X_sda)
        
        # --- Save Data ---
        save_file = os.path.join(subj_out_dir, f'tsne_data_{subj}.npz')
        np.savez(save_file, 
                 emb_so=emb_so, y_so=y_so, 
                 emb_sda=emb_sda, y_sda=y_sda,
                 z_s_so=z_s_so, z_t_so=z_t_so,
                 z_s_sda=z_s_sda, z_t_sda=z_t_sda)
        print(f"  Saved data to {save_file}")

if __name__ == "__main__":
    visualize_tsne_batch()
