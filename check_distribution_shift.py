
import os
import numpy as np
import h5py
from sda_training import CONFIG, load_source_data, get_target_trials, load_target_data_by_trials

def check_distributions():
    # 1. Source Stats
    print("Loading Source Data (Theta)...")
    src_conf = CONFIG.copy()
    src_conf['feature_set'] = 'theta'
    # load_source_data returns (Data, Labels). Data shape (N, T, F). 
    # Theta is index 6 (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, theta, theta_vel)
    try:
        src_data, _ = load_source_data(src_conf)
        src_theta_mean = src_data[:, :, 6].mean()
        print(f"Source Theta Mean: {src_theta_mean:.4f}")
    except Exception as e:
        print(f"Failed to load source: {e}")
        return

    # 2. Target Stats per Subject
    subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    
    print("\n--- Target Subjects Theta Means ---")
    print(f"{'Subject':<10} | {'Theta Mean':<12} | {'Diff vs Source':<15}")
    print("-" * 45)
    
    for subj in subjects:
        subj_conf = CONFIG.copy()
        subj_conf['target_subject'] = subj
        subj_conf['feature_set'] = 'theta'
        
        # Get trials
        trials = get_target_trials(subj_conf)
        if not trials:
            print(f"{subj:<10} | {'No Data':<12} | -")
            continue
            
        # Load all trials for robust stat
        # To save time, maybe just first 3 trials? Use slice.
        trials_to_load = trials
        
        try:
            tgt_data, _ = load_target_data_by_trials(subj_conf, trials_to_load)
            if len(tgt_data) == 0:
                print(f"{subj:<10} | {'Empty':<12} | -")
                continue
                
            tgt_theta_mean = tgt_data[:, :, 6].mean()
            diff = tgt_theta_mean - src_theta_mean
            
            print(f"{subj:<10} | {tgt_theta_mean:<12.4f} | {diff:<15.4f}")
            
        except Exception as e:
            print(f"{subj:<10} | Error: {e}")

if __name__ == "__main__":
    check_distributions()
