import os
import subprocess
import itertools
import json
import torch
import sys
import uuid
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_experiment(params):
    """
    Worker function to run a single experiment.
    params: dict containing all config and run details
    """
    subj = params['subj']
    p = params['p']
    python_exe = params['python_exe']
    training_script = params['training_script']
    results_base_dir = params['results_base_dir']
    
    # Map Dimensions
    enc_layers = [p['encoder_base_dim'], p['encoder_base_dim'] // 2]
    dec_layers = [p['decoder_dim']]
    
    # Run ID
    run_name = f"L{p['lambda_mmd']}_LR{p['learning_rate']}_E{p['encoder_base_dim']}_D{p['decoder_dim']}_Dr{p['dropout']}"
    target_results_dir = os.path.join(results_base_dir, f"{subj}_{run_name}")
    
    # Unique Config File to avoid race conditions
    unique_id = str(uuid.uuid4())
    config_filename = f"config_override_{unique_id}.json"
    
    override = {
        'target_subject': subj,
        'lambda_mmd': p['lambda_mmd'],
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
        'feature_set': 'theta'
    }
    
    try:
        with open(config_filename, 'w') as f:
            json.dump(override, f)
            
        # Run with specific environment variable
        env = os.environ.copy()
        env['SDA_CONFIG_FILE'] = config_filename
        
        # Capture output
        result = subprocess.run([python_exe, training_script], env=env, capture_output=True, text=True)
        
        # Parse MSE
        mse = None
        if os.path.exists(os.path.join(target_results_dir, subj)):
            subdirs = os.listdir(os.path.join(target_results_dir, subj))
            for d in subdirs:
                if "Fold_1_" in d and "_SDA_" in d:
                    mse_path = os.path.join(target_results_dir, subj, d, "final_mse.txt")
                    if os.path.exists(mse_path):
                        with open(mse_path, 'r') as f:
                            mse = f.read().strip()
                    break
        
        return {
            'subj': subj,
            'p': p,
            'mse': mse,
            'status': 'Success' if mse else 'Failed',
            'run_name': run_name,
            'output': result.stdout,
            'error': result.stderr
        }
        
    except Exception as e:
        return {'status': 'Error', 'error': str(e)}
        
    finally:
        # Cleanup
        if os.path.exists(config_filename):
            os.remove(config_filename)

def run_grid_search_parallel():
    # 1. Force GPU
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA (GPU) is NOT available. Exiting.")
        sys.exit(1)
        
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # 2. Define Search Space
    subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    
    # Params
    param_grid = {
        'lambda_mmd': [0.5, 1.0, 1.5, 2.0],
        'learning_rate': [0.001, 0.0001],
        'encoder_base_dim': [64, 128, 256],
        'decoder_dim': [32, 64, 128],
        'dropout': [0.2, 0.3]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    total_runs = len(combinations) * len(subjects)
    print(f"Total Combinations: {len(combinations)} x {len(subjects)} = {total_runs} runs")
    
    # Config
    python_exe = "d:/rsc lab/anaconda3/envs/torch_gpu/python.exe"
    training_script = "d:/RSC lab/Codes/Stroke_SDA/sda_training.py"
    # Separate Result Directory
    results_base_dir = "d:/RSC lab/Codes/Stroke_SDA/results_grid_search_theta"
    # Separate Log File
    log_file = "grid_search_log_theta.csv"
    
    # Init Log
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Subject,Lambda,LR,EncDim,DecDim,Dropout,MSE\n")
            
    # Prepare Tasks
    tasks = []
    for subj in subjects:
        for comb in combinations:
            p = dict(zip(keys, comb))
            tasks.append({
                'subj': subj,
                'p': p,
                'python_exe': python_exe,
                'training_script': training_script,
                'results_base_dir': results_base_dir
            })
            
    # Execute Parallel
    max_workers = 4 
    
    print(f"Starting Parallel Execution with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all (This list might be huge if we are not careful)
        # 144 * 6 = 864 tasks.
        future_to_task = {executor.submit(run_single_experiment, t): t for t in tasks}
        
        count = 0
        for future in as_completed(future_to_task):
            count += 1
            res = future.result()
            task = future_to_task[future]
            
            p = res.get('p', {})
            mse = res.get('mse')
            subj = res.get('subj', 'Unknown')
            
            if res['status'] == 'Success':
                print(f"[{count}/{total_runs}] {subj} {res['run_name']} -> MSE: {mse}")
                with open(log_file, 'a') as f:
                    f.write(f"{subj},{p['lambda_mmd']},{p['learning_rate']},{p['encoder_base_dim']},{p['decoder_dim']},{p['dropout']},{mse}\n")
            else:
                print(f"[{count}/{total_runs}] FAILED: {task['subj']} - {res.get('error')}")
                if count <= 5: # Debug print first few failures
                    print("Debug Info:")
                    print(res.get('output', ''))
                    print(res.get('error', ''))

if __name__ == "__main__":
    run_grid_search_parallel()
