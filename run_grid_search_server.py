
import os
import subprocess # Still used? Maybe not, but good for backup
import itertools
import json
import torch
import sys
import uuid
import time
import multiprocessing
import numpy as np
import sda_training as sda # Import the module directly

# Global variables for worker processes (inherited via fork)
# Or set via initializer
SRC_DATA_GLOBAL = None
SRC_LABELS_GLOBAL = None
GPU_QUEUE = None

def worker_init(src_data, src_labels, gpu_queue):
    global SRC_DATA_GLOBAL, SRC_LABELS_GLOBAL, GPU_QUEUE
    SRC_DATA_GLOBAL = src_data
    SRC_LABELS_GLOBAL = src_labels
    GPU_QUEUE = gpu_queue

def run_single_experiment_optimized(params):
    """
    Worker function executed by Pool.
    Uses pre-loaded global data to save I/O.
    """
    subj = params['subj']
    p = params['p']
    results_base_dir = params['results_base_dir']
    
    # helper for clean logging
    run_name = f"L{p['lambda_mmd']}_LS{p['lambda_src']}_LR{p['learning_rate']}_E{p['encoder_base_dim']}_D{p['decoder_dim']}_Dr{p['dropout']}"
    target_results_dir = os.path.join(results_base_dir, f"{subj}_{run_name}")
    
    # Acquire GPU
    gpu_id = GPU_QUEUE.get()
    
    # Redirect Stdout/Stderr to a distinct log file per run
    log_file_path = os.path.join("logs_v3", f"{subj}_{run_name}.out")
    os.makedirs("logs_v3", exist_ok=True)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class Unbuffered(object):
       def __init__(self, stream):
           self.stream = stream
       def write(self, data):
           self.stream.write(data)
           self.stream.flush()
       def writelines(self, datas):
           self.stream.writelines(datas)
           self.stream.flush()
       def __getattr__(self, attr):
           return getattr(self.stream, attr)

    status = 'Failed'
    mse_results = {}
    
    try:
        with open(log_file_path, 'w') as f_log:
            sys.stdout = Unbuffered(f_log)
            sys.stderr = Unbuffered(f_log)
            
            print(f"Starting Run on GPU {gpu_id}")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Setup Config dictionary
            enc_layers = [p['encoder_base_dim'], p['encoder_base_dim'] // 2]
            dec_layers = [p['decoder_dim']]
            
            config = {
                'target_subject': subj,
                'lambda_mmd': p['lambda_mmd'],
                'lambda_src': p['lambda_src'],
                'learning_rate': p['learning_rate'],
                'encoder_layers': enc_layers,
                'decoder_layers': dec_layers,
                'dropout': p['dropout'],
                'results_dir': target_results_dir, 
                'max_folds': 1,
                'epochs': 100,
                'patience': 5,
                'batch_size': 64,
                'modes': ['SDA'],
                'feature_set': 'theta',
                'device': f'cuda:{gpu_id}', # Direct device assignment
                'run_name': run_name
            }
            
            # CALL TRAINING DIRECTLY
            # Use global SRC Data
            try:
                # sda.train_single_run expects (config, src_data, src_labels)
                results_dict = sda.train_single_run(config, SRC_DATA_GLOBAL, SRC_LABELS_GLOBAL)
                
                # Parse Results
                # results_dict = {'SDA_Fold1': 0.023, ...}
                # We need to average or just take the one we ran
                modes = ['SDA']
                for m in modes:
                    vals = [v for k, v in results_dict.items() if k.startswith(m)]
                    if vals:
                        mse_results[m] = sum(vals) / len(vals)
                    else:
                        mse_results[m] = None
                
                status = 'Success'
                
            except Exception as e:
                print(f"CRITICAL ERROR: {e}")
                import traceback
                traceback.print_exc()
                status = 'Error'
                
            print(f"Finished. Status: {status}")

    except Exception as e:
        status = 'Error'
        # Can't log to file if open failed, but try printing to original
        original_stderr.write(f"Worker Error {run_name}: {e}\n")
        
    finally:
        # Restore Streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        GPU_QUEUE.put(gpu_id)
        
    return {
        'subj': subj,
        'p': p,
        'mse_results': mse_results,
        'status': status,
        'run_name': run_name
    }

def run_grid_search_parallel():
    # 1. GPU Setup (AVOID CUDA INIT IN PARENT)
    # We assume 4 GPUs are available on this server.
    # calling torch.cuda here causes "Cannot re-initialize CUDA in forked subprocess" error.
    num_gpus = 4
    print(f"Assuming {num_gpus} GPUs available.")
    
    # 2. LOAD DATA ONCE (The Optimization)
    print("Loading Source Data into System Memory (Shared by Workers)...")
    t0 = time.time()
    # Dummy config for loading
    dummy_config = sda.CONFIG.copy()
    dummy_config['feature_set'] = 'theta'
    src_data_loaded, src_labels_loaded = sda.load_source_data(dummy_config)
    print(f"Data Loaded in {time.time()-t0:.2f}s. Shape: {src_data_loaded.shape}")
    
    # 3. Setup Multiprocessing
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)
        
    # 4. Define Search Space
    subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    
    # Extended Grid Search (Resume v3)
    # Merged Old + New Candidates
    param_grid = {
        'lambda_mmd': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 
        'lambda_src': [0.3, 0.5, 0.7, 1.0],
        'learning_rate': [0.0001],
        'encoder_base_dim': [64, 128, 256],
        'decoder_dim': [32, 64, 128],
        'dropout': [0.3]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    total_runs = len(combinations) * len(subjects)
    print(f"Total Combinations: {len(combinations)} x {len(subjects)} = {total_runs} runs")
    
    results_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_grid_search_theta_v3")
    os.makedirs(results_base_dir, exist_ok=True)
    
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_log_theta_v3.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Subject,Mode,LambdaMMD,LambdaSrc,LR,EncDim,DecDim,Dropout,MSE\n")
            
    tasks = []
    for subj in subjects:
        for comb in combinations:
            p = dict(zip(keys, comb))
            tasks.append({
                'subj': subj,
                'p': p,
                'results_base_dir': results_base_dir
            })
            
    # 5. Execute with Pool
    # We use 'fork' which is default on Linux.
    # We pass the large data via initializer to set globals in workers without pickling per task?
    # Actually, passing arguments to 'worker_init' allows pickling ONCE per process, 
    # rather than once per task. This is efficient.
    
    max_workers = 4 
    print(f"Starting Optimized Parallel Execution with {max_workers} workers...")
    
    with multiprocessing.Pool(processes=max_workers, initializer=worker_init, initargs=(src_data_loaded, src_labels_loaded, gpu_queue)) as pool:
        # Use imap_unordered for progress tracking
        results_iter = pool.imap_unordered(run_single_experiment_optimized, tasks)
        
        count = 0
        for res in results_iter:
            count += 1
            subj = res.get('subj', '-')
            name = res.get('run_name', '-')
            
            if res['status'] == 'Success':
                mse_results = res.get('mse_results', {})
                print(f"[{count}/{total_runs}] {subj} {name} - DONE")
                
                with open(log_file, 'a') as f:
                    for m, val in mse_results.items():
                        # Param lookup from run name is hard, let's use the 'p' returned
                        p = res['p']
                        if val is not None:
                             # Subject,Mode,LambdaMMD,LambdaSrc,LR,EncDim,DecDim,Dropout,MSE
                             f.write(f"{subj},{m},{p['lambda_mmd']},{p['lambda_src']},{p['learning_rate']},{p['encoder_base_dim']},{p['decoder_dim']},{p['dropout']},{val}\n")
                        else:
                             # Maybe mode failed or wasn't run
                             pass
            else:
                 print(f"[{count}/{total_runs}] {subj} {name} - FAILED (Check logs/{subj}_{name}.out)")

if __name__ == "__main__":
    # Required for safe multiprocessing (though fork is default)
    # torch.multiprocessing.set_start_method('spawn') # NO! We want 'fork' for data sharing if possible.
    # Default is fork.
    run_grid_search_parallel()
