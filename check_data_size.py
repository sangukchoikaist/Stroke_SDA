from sda_training import CONFIG, load_source_data, load_target_data_by_trials
import numpy as np

print("Loading Source Data...")
src_data, src_labels = load_source_data(CONFIG)
print(f"Source Data (All Healthy): {src_data.shape[0]} samples")

print("Loading Target Data (S003)...")
target_trials = ['S003_T002', 'S003_T003', 'S003_T006']
tgt_data, tgt_labels = load_target_data_by_trials(CONFIG, target_trials)
print(f"Target Data (S003 Total): {tgt_data.shape[0]} samples")

print("Breakdown by Trial:")
for trial in target_trials:
    t_data, _ = load_target_data_by_trials(CONFIG, [trial])
    print(f"  - {trial}: {t_data.shape[0]} samples")
