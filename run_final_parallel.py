
import os
import sys
import time
import multiprocessing
import torch
import sda_training as sda

# Rank 1 Configuration
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
    'max_folds': 3,
    'data_fraction': 1.0,
    'feature_set': 'theta',
    'modes': ['SDA', 'TO', 'SO', 'TL'], # All modes
    'results_dir': 'results_final_rank1'
}

# Globals for Workers
SRC_DATA_GLOBAL = None
SRC_LABELS_GLOBAL = None
GPU_QUEUE = None

def worker_init(src_data, src_labels, gpu_queue):
    global SRC_DATA_GLOBAL, SRC_LABELS_GLOBAL, GPU_QUEUE
    SRC_DATA_GLOBAL = src_data
    SRC_LABELS_GLOBAL = src_labels
    GPU_QUEUE = gpu_queue

def run_subject(subj):
    gpu_id = GPU_QUEUE.get()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        print(f"[{subj}] Starting on GPU {gpu_id}")
        
        # Log redirection per subject
        log_dir = 'logs/rank1_parallel'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{subj}.log")
        
        with open(log_file, 'w') as f:
            # Redirect
            sys.stdout = f
            sys.stderr = f
            
            # Setup Config
            config = RANK1_CONFIG.copy()
            config['target_subject'] = subj
            config['device'] = f'cuda:{gpu_id}'
            
            # Run
            try:
                sda.train_single_run(config, SRC_DATA_GLOBAL, SRC_LABELS_GLOBAL)
                print(f"[{subj}] Finished successfully.")
            except Exception as e:
                print(f"[{subj}] Failed: {e}")
                import traceback
                traceback.print_exc()

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        GPU_QUEUE.put(gpu_id)

def main():
    subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    
    # Load Source Data Once
    print("Loading Source Data...")
    # Create a dummy config to load source data
    dummy_config = RANK1_CONFIG.copy()
    # Ensure source_h5 logic works
    dummy_config.update(sda.CONFIG) # Update with paths from original config
    src_data, src_labels = sda.load_source_data(dummy_config)
    print(f"Source Data Loaded: {src_data.shape}")
    
    # Setup GPU Queue
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1 # Fallback
    print(f"Available GPUs: {num_gpus}")
    
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)
        
    # Pool
    # We can run len(subjects) tasks. 
    # With 4 GPUs, 4 run, 2 wait.
    with multiprocessing.Pool(processes=min(len(subjects), num_gpus), 
                              initializer=worker_init, 
                              initargs=(src_data, src_labels, gpu_queue)) as pool:
        pool.map(run_subject, subjects)
        
    print("All subjects completed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
