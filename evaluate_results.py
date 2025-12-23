import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

CONFIG = {
    'results_dir': 'd:/RSC lab/Codes/Stroke_SDA/results',
}

def evaluate_all_subjects():
    subjects = ['S002', 'S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    modes = ['SDA', 'TO', 'SO', 'TL']
    
    # Store all MSEs for statistical test
    # Structure: {'SDA': [mse1, mse2, ...], 'TO': [...], ...}
    global_results = {m: [] for m in modes}
    
    print(f"\n{'Subject':<10} | {'SDA':<10} {'TO':<10} {'TL':<10} {'SO':<10}")
    print("-" * 60)
    
    for subj in subjects:
        subj_dir = os.path.join(CONFIG['results_dir'], subj)
        if not os.path.exists(subj_dir):
            print(f"{subj:<10} | Not Found")
            continue
            
        folders = os.listdir(subj_dir)
        
        # Group by Prefix (Fold_Trial) to ensure alignment
        data_map = {} 
        
        for f in folders:
            # Filter for frac1.0
            if 'frac1.0' not in f: continue
            
            # Find Mode
            found_mode = None
            for m in modes:
                if f"_{m}_" in f: 
                    found_mode = m
                    break
            if not found_mode: continue
            
            # Prefix: Fold_1_S003_T002
            prefix = f.split(f"_{found_mode}_")[0]
            
            # Read MSE
            mse_path = os.path.join(subj_dir, f, 'final_mse.txt')
            if os.path.exists(mse_path):
                try:
                    with open(mse_path, 'r') as file:
                        mse = float(file.read().strip())
                        if prefix not in data_map: data_map[prefix] = {}
                        data_map[prefix][found_mode] = mse
                except: pass
        
        # Calculate Means for Display
        means = []
        for m in modes:
            vals = []
            for k in data_map:
                if m in data_map[k]: vals.append(data_map[k][m])
            
            if len(vals) > 0: means.append(f"{np.mean(vals):.4f}")
            else: means.append("-")
            
        print(f"{subj:<10} | {means[0]:<10} {means[1]:<10} {means[2]:<10} {means[3]:<10}")

        # Align for Global T-Test (SDA vs others)
        for k in data_map:
            # Need SDA to exist to be a valid pair base
            if 'SDA' in data_map[k]:
                # Pairs
                if 'TO' in data_map[k]:
                    global_results['SDA'].append(data_map[k]['SDA'])
                    global_results['TO'].append(data_map[k]['TO'])
                
                    # Add others if they exist (independent pairs? or strict intersection?)
                    # For simplicity, let's just collect all pairs (SDA-TO, SDA-TL, etc.) independently?
                    # But global_results dict approach implies consistent index.
                    # Actually, ttest_rel needs aligned arrays.
                    # Let's create separate paired lists for robustness.
                    pass 

    print("-" * 60)
    
    # Reload logic for T-tests to be precise
    # Collect PAIRS
    sda_to_pairs = []
    sda_tl_pairs = []
    sda_so_pairs = []
    
    for subj in subjects:
        subj_dir = os.path.join(CONFIG['results_dir'], subj)
        if not os.path.exists(subj_dir): continue
        folders = os.listdir(subj_dir)
        data_map = {}
        for f in folders:
            if 'frac1.0' not in f: continue
            found_mode = None
            for m in modes:
                if f"_{m}_" in f: 
                    found_mode = m
                    break
            if not found_mode: continue
            prefix = f.split(f"_{found_mode}_")[0]
            mse_path = os.path.join(subj_dir, f, 'final_mse.txt')
            if os.path.exists(mse_path):
                with open(mse_path, 'r') as file:
                    mse = float(file.read().strip())
                    if prefix not in data_map: data_map[prefix] = {}
                    data_map[prefix][found_mode] = mse
        
        for k in data_map:
            if 'SDA' in data_map[k]:
                if 'TO' in data_map[k]: sda_to_pairs.append((data_map[k]['SDA'], data_map[k]['TO']))
                if 'TL' in data_map[k]: sda_tl_pairs.append((data_map[k]['SDA'], data_map[k]['TL']))
                if 'SO' in data_map[k]: sda_so_pairs.append((data_map[k]['SDA'], data_map[k]['SO']))
    
    print("\nXXX Statistical Analysis (Paired T-Test) XXX")
    
    def print_result(name, pairs):
        if len(pairs) < 2:
            print(f"{name}: Not enough samples (N={len(pairs)})")
            return
        a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
        t, p = stats.ttest_rel(a, b)
        mean_a = np.mean(a); mean_b = np.mean(b)
        print(f"{name} (N={len(pairs)}):")
        print(f"  SDA Mean: {mean_a:.4f} | Other Mean: {mean_b:.4f}")
        print(f"  t-stat: {t:.4f} | p-value: {p:.5f} {'*' if p<0.05 else ''}")

    print_result("SDA vs TO", sda_to_pairs)
    print_result("SDA vs TL", sda_tl_pairs)
    print_result("SDA vs SO", sda_so_pairs)

if __name__ == "__main__":
    evaluate_all_subjects()
