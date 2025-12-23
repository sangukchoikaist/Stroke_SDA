
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_analysis():
    base_dir = "d:/RSC lab/Codes/Stroke_SDA/results_analysis"
    main_results_dir = "d:/RSC lab/Codes/Stroke_SDA/results"
    
    # 1. Ablation: Effect of MMD (S003)
    # Compare "SDA (1.5)" from Main Results vs "Ablation_NoMMD (0.0)"
    
    # We need to average across folds for S003
    def get_avg_mse(root_dir, subject='S003', mode='SDA'):
        subj_dir = os.path.join(root_dir, subject)
        if not os.path.exists(subj_dir): return None
        
        folders = os.listdir(subj_dir)
        mses = []
        for f in folders:
            if f"_{mode}_" in f and "frac1.0" in f:
                mse_path = os.path.join(subj_dir, f, 'final_mse.txt')
                if os.path.exists(mse_path):
                    with open(mse_path, 'r') as file:
                        mses.append(float(file.read().strip()))
        
        if len(mses) == 0: return None
        return np.mean(mses)
    
    # Get SDA Baseline (Lambda=1.5)
    sda_baseline = get_avg_mse(main_results_dir, 'S003', 'SDA')
    
    # Get NoMMD (Lambda=0.0)
    no_mmd = get_avg_mse(os.path.join(base_dir, 'Ablation_NoMMD'), 'S003', 'SDA')
    
    print(f"SDA (1.5): {sda_baseline}")
    print(f"No MMD (0.0): {no_mmd}")
    
    if sda_baseline is not None and no_mmd is not None:
        plt.figure(figsize=(6, 5))
        plt.bar(['Without MMD (Joint)', 'With MMD (SDA)'], [no_mmd, sda_baseline], color=['gray', 'blue'])
        plt.ylabel('MSE')
        plt.title('Ablation Study: Effect of MMD Loss')
        plt.savefig('d:/RSC lab/Codes/Stroke_SDA/plot_ablation.png')
        print("Saved plot_ablation.png")
        
    # 2. Sensitivity: Lambda 0.1, 0.5, 1.0, 1.5, 3.0?
    lambdas = [0.0, 0.1, 0.5, 1.0, 1.5]
    mses = []
    
    # 0.0
    mses.append(no_mmd if no_mmd else np.nan)
    
    # 0.1
    l01 = get_avg_mse(os.path.join(base_dir, 'Sensitivity_L0.1'), 'S003', 'SDA')
    mses.append(l01 if l01 else np.nan)
    
    # 0.5
    l05 = get_avg_mse(os.path.join(base_dir, 'Sensitivity_L0.5'), 'S003', 'SDA')
    mses.append(l05 if l05 else np.nan)
    
    # 1.0
    l10 = get_avg_mse(os.path.join(base_dir, 'Sensitivity_L1.0'), 'S003', 'SDA')
    mses.append(l10 if l10 else np.nan)
    
    # 1.5 (Baseline)
    mses.append(sda_baseline if sda_baseline else np.nan)
    
    print(f"Lambdas: {lambdas}")
    print(f"MSEs: {mses}")
    
    # Filter Nans
    valid_l = []
    valid_m = []
    for l, m in zip(lambdas, mses):
        if not np.isnan(m):
            valid_l.append(l)
            valid_m.append(m)
            
    if len(valid_l) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot(valid_l, valid_m, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Lambda MMD')
        plt.ylabel('MSE')
        plt.title('Sensitivity Analysis: Lambda MMD')
        plt.grid(True)
        plt.savefig('d:/RSC lab/Codes/Stroke_SDA/plot_sensitivity.png')
        print("Saved plot_sensitivity.png")

if __name__ == "__main__":
    visualize_analysis()
