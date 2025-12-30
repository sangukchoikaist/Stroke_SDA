
import sda_training as sda
import numpy as np

# Replicate run_final_parallel.py config setup
RANK1_CONFIG = {
    'lambda_mmd': 1.5,
    'lambda_src': 0.7,
    'lambda_tgt': 1.0, 
    'learning_rate': 0.0001,
    'encoder_layers': [128, 64],
    'decoder_layers': [64],
    'dropout': 0.3,
    'input_dim': 8,
    'hidden_dim': 64,
    'output_dim': 2,
    'patience': 10,
    'min_delta': 0.0001,
    'epochs': 50,
    'max_folds': 1,
    'data_fraction': 1.0,
    'feature_set': 'theta',
    'modes': ['SDA', 'TO', 'SO', 'TL'], # All modes
    'results_dir': 'results_final_rank1'
}

def debug_loading():
    print("Simulating run_final_parallel loading...")
    
    # Logic from run_final_parallel.py lines 84-87
    dummy_config = RANK1_CONFIG.copy()
    dummy_config.update(sda.CONFIG)
    
    print(f"Final Config feature_set: {dummy_config.get('feature_set')}")
    
    # Load
    try:
        src_data, src_labels = sda.load_source_data(dummy_config)
        print(f"Loaded Shape: {src_data.shape}")
        
        # Check Mean of Theta (Index 6)
        theta_mean = src_data[:, :, 6].mean()
        print(f"Theta Mean (Index 6): {theta_mean}")
        
        # Check Mean of all features
        print(f"All Means: {src_data.mean(axis=(0,1))}")
        
        if abs(theta_mean - 3.33) < 0.1:
            print("BUG REPRODUCED: Mean is ~3.33")
        elif abs(theta_mean) < 0.1:
            print("NO BUG: Mean is ~0.0")
        else:
            print(f"UNKNOWN STATE: Mean is {theta_mean}")
            
    except Exception as e:
        print(f"Loading Failed: {e}")

if __name__ == "__main__":
    debug_loading()
