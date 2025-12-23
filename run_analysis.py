
import os
import subprocess
import time

def run_analysis():
    # Helper to run sda_training.py with modified globals via command line or modification?
    # sda_training.py doesn't have argparse.
    # To avoid rewriting it, I will use a simple trick: 
    # Modify sda_training.py temporarily or pass config via environment variables?
    # Environment variables are cleaner if supported, but Config is hardcoded.
    # I will modify sda_training.py to read overrides from a json file 'config_override.json' if it exists.
    
    # 1. Modify sda_training.py to support overrides
    print("Please modify sda_training.py to support 'config_override.json' first if not done.")
    # Actually, I'll assume I do that next.
    
    # 2. Experiments
    # S003, Fold 1 (to be fast) or all folds? Just Fold 1 for trend.
    # User asked for analysis, so specific runs are needed.
    
    # List of experiments
    experiments = [
        {'name': 'Ablation_NoMMD', 'lambda_mmd': 0.0},
        {'name': 'Sensitivity_L0.1', 'lambda_mmd': 0.1},
        {'name': 'Sensitivity_L0.5', 'lambda_mmd': 0.5},
        {'name': 'Sensitivity_L1.0', 'lambda_mmd': 1.0},
        # {'name': 'Sensitivity_L3.0', 'lambda_mmd': 3.0}, # Skip, we know it diverges
    ]
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(scripts_dir, 'sda_training.py')
    python_exe = "d:/rsc lab/anaconda3/envs/torch_gpu/python.exe"
    
    for exp in experiments:
        print(f"\n>>> Running Experiment: {exp['name']} <<<")
        
        # Write Override
        import json
        override = {
            'target_subject': 'S003',
            'lambda_mmd': exp['lambda_mmd'],
            'results_dir': f"results_analysis/{exp['name']}",
            'epochs': 20, # Shorter epochs for analysis
            'batch_size': 64,
            'patience': 5
        }
        
        with open('config_override.json', 'w') as f:
            json.dump(override, f)
            
        # Run
        subprocess.run([python_exe, training_script])
        
        # Read Result (MSE)
        # We need to find where it saved. 
        # sda_training saves to results_analysis/{exp}/S003/...
        # We will parse later.

if __name__ == "__main__":
    run_analysis()
