import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import joblib
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data, normalize_fixed

# Configuration
CONFIG['results_dir'] = 'results_analysis/Batch_SDA_Effect'
CONFIG['batch_size'] = 64
CONFIG['input_dim'] = 8
CONFIG['window_size'] = 100
CONFIG['stride_tgt'] = 5
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Subjects
subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']

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
    
    # Pre-load Source Data (Large, load once)
    print("Loading Source Data (All Healthy)...")
    src_data, _ = load_source_data(CONFIG) # (N_s, T, F)
    
    for subj in subjects:
        print(f"\nProcessing Subject: {subj}")
        CONFIG['target_subject'] = subj
        
        target_trials = get_target_trials(CONFIG)
        # Assuming Fold 1, Test Trial is usually the first one or we pick one.
        # In sda_training, target_trials[0] is often used as test for Fold 1 if sorted?
        # Let's dynamically find the folder. Assumes standardized naming.
        
        # Determine Folder Name by looking at directory
        subj_dir = os.path.join(CONFIG['results_dir'], subj)
        if not os.path.exists(subj_dir):
            print(f"  Directory not found: {subj_dir}, skipping.")
            continue
            
        # Find SO folders in current batch directory
        so_folders = [d for d in os.listdir(subj_dir) if '_SO_' in d and 'frac1.0' in d]
        
        if not so_folders:
            print(f"  No SO results found for {subj} in batch dir, skipping.")
            continue
            
        # Use first SO folder found (Fold 1 preferably)
        so_folder_name = so_folders[0] # e.g. Fold_1_S003_T002_SO_frac1.0
        
        # Parse info
        parts = so_folder_name.split('_')
        fold_idx = parts[1]
        # Folder format: Fold_{idx}_{subj}_{trial}_SO_...
        # Valid parts: Fold, 1, S003, T002, SO...
        
        full_trial_name = f"{parts[2]}_{parts[3]}" # S003_T002
        
        print(f"  Using Fold {fold_idx}, Test Trial {full_trial_name}")
        
        so_dir = os.path.join(subj_dir, so_folder_name)
        
        # --- Find SDA Model in LEGACY results folder ---
        legacy_root = 'results' # Relative to run dir
        legacy_subj_dir = os.path.join(legacy_root, subj)
        
        # Find corresponding SDA folder
        # Pattern: Fold_{fold_idx}_{full_trial_name}_SDA_frac1.0
        sda_target_name = so_folder_name.replace('_SO_', '_SDA_')
        
        sda_dir = os.path.join(legacy_subj_dir, sda_target_name)
        
        # Check if exists
        if not os.path.exists(sda_dir):
            print(f"  SDA folder not found in legacy results: {sda_dir}")
            # Try searching loosely
            if os.path.exists(legacy_subj_dir):
                candidates = [d for d in os.listdir(legacy_subj_dir) if f'Fold_{fold_idx}' in d and '_SDA_' in d]
                if candidates:
                    sda_dir = os.path.join(legacy_subj_dir, candidates[0])
                    print(f"  Found alternative SDA folder: {candidates[0]}")
                else:
                    print(f"  No SDA candidate found in {legacy_subj_dir}, skipping.")
                    continue
            else:
                 print(f"  Legacy subject dir {legacy_subj_dir} not found, skipping.")
                 continue

        # Load Scaler from SO (Source Norm)
        scaler_path = os.path.join(so_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            print("  SO Scaler not found, skipping.")
            continue  
        scaler_so = joblib.load(scaler_path)

        # Load Scaler from SDA (Target/Indep Norm probably)
        scaler_sda_path = os.path.join(sda_dir, 'scaler.pkl')
        if not os.path.exists(scaler_sda_path):
            print("  SDA Scaler not found (Legacy), skipping.")
            continue
        scaler_sda = joblib.load(scaler_sda_path)

        # Load Raw Target Test Data
        tgt_test_data, _ = load_target_data_by_trials(CONFIG, [full_trial_name])
        
        # Transform for SO (Use SO scaler - SourceFixed)
        N_s, T, F = src_data.shape
        src_norm_so = scaler_so.transform(src_data.reshape(-1, F)).reshape(N_s, T, F)
        N_t, T, F = tgt_test_data.shape
        tgt_norm_so = scaler_so.transform(tgt_test_data.reshape(-1, F)).reshape(N_t, T, F)
        
        # Transform for SDA (Use SDA scaler - Independent)
        # Note: SDA model was trained with Independent Norm for Target.
        # But Source was trained with Source Norm?
        # In normalize_independent:
        #   src_scaler = StandardScaler().fit(src) -> src_norm
        #   tgt_scaler = StandardScaler().fit(tgt_train) -> tgt_norm
        #   Returns src_norm, ... tgt_scaler.
        # So SDA Source input should be Source Normalized.
        # SDA Target input should be Target Normalized.
        # We can reuse src_norm_so for SDA Source input if it's the same scaling.
        # Yes, normalize_fixed(src) and normalize_independent(src) do the same thing for SRC.
        
        tgt_norm_sda = scaler_sda.transform(tgt_test_data.reshape(-1, F)).reshape(N_t, T, F)
        
        # Subsample Source
        np.random.seed(42)
        idx_s = np.random.choice(len(src_norm_so), min(1000, len(src_norm_so)), replace=False)
        src_sample_so = src_norm_so[idx_s]
        src_sample_sda = src_sample_so 
        
        tgt_sample_so = tgt_norm_so
        tgt_sample_sda = tgt_norm_sda
        
        # --- SO Model ---
        print("  Extracting SO features...")
        model_so = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
        try:
            model_so.load_state_dict(torch.load(os.path.join(so_dir, 'best_model.pth')))
        except Exception:
            print("  Failed to load SO model, skipping.")
            continue
            
        z_s_so = extract_features(model_so, src_sample_so, domain='source')
        z_t_so = extract_features(model_so, tgt_sample_so, domain='source')
        
        # --- SDA Model ---
        print("  Extracting SDA features...")
        model_sda = SDA_Dual_Model(CONFIG).to(CONFIG['device'])
        try:
            model_sda.load_state_dict(torch.load(os.path.join(sda_dir, 'best_model.pth')))
        except Exception:
             print("  Failed to load SDA model, skipping.")
             continue


        z_s_sda = extract_features(model_sda, src_sample_sda, domain='source')
        z_t_sda = extract_features(model_sda, tgt_sample_sda, domain='target')
        
        # --- t-SNE ---
        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # Combine all 4 sets? Or run separately for Before/After?
        # Typically better to run separately to see independent structural alignment?
        # Or run together to ensure same scale? 
        # Usually separate t-SNEs are fine as we compare the *separation*, not the coordinate values.
        
        # Run Separately
        X_so = np.concatenate([z_s_so, z_t_so], axis=0)
        y_so = np.concatenate([np.zeros(len(z_s_so)), np.ones(len(z_t_so))], axis=0)
        emb_so = tsne.fit_transform(X_so)
        
        X_sda = np.concatenate([z_s_sda, z_t_sda], axis=0)
        y_sda = y_so
        emb_sda = tsne.fit_transform(X_sda)
        
        # --- Save Data ---
        save_file = os.path.join(subj_dir, f'tsne_data_{subj}.npz')
        np.savez(save_file, 
                 emb_so=emb_so, y_so=y_so, 
                 emb_sda=emb_sda, y_sda=y_sda,
                 z_s_so=z_s_so, z_t_so=z_t_so,
                 z_s_sda=z_s_sda, z_t_sda=z_t_sda)
        print(f"  Saved data to {save_file}")
        
        # --- Plot ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(emb_so[y_so==0, 0], emb_so[y_so==0, 1], c='blue', alpha=0.5, label='Source', s=10)
        plt.scatter(emb_so[y_so==1, 0], emb_so[y_so==1, 1], c='red', alpha=0.5, label='Target', s=10)
        plt.title(f"{subj}: Before SDA")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(emb_sda[y_sda==0, 0], emb_sda[y_sda==0, 1], c='blue', alpha=0.5, label='Source', s=10)
        plt.scatter(emb_sda[y_sda==1, 0], emb_sda[y_sda==1, 1], c='red', alpha=0.5, label='Target', s=10)
        plt.title(f"{subj}: After SDA")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, f'tsne_{subj}.png'))
        plt.close()

if __name__ == "__main__":
    visualize_tsne_batch()
