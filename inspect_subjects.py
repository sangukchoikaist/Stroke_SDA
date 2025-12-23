
import h5py
import numpy as np

source_path = 'd:/RSC lab/Codes/Stroke_SDA/output/all_subjects_dataset_ds_hip.h5'
target_path = 'd:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5'

print("--- Source Data (Healthy) ---")
try:
    with h5py.File(source_path, 'r') as f:
        subjects = set()
        skipped = 0
        total = 0
        for key in f.keys():
            total += 1
            grp = f[key]
            # Speed filter
            if 'walking_speed' in grp:
                speed = np.mean(grp['walking_speed'])
                if speed > 0.7:
                    skipped += 1
                    continue
            
            # Extract Subject ID (ABxx)
            parts = key.split('_')
            subj_id = parts[0]
            subjects.add(subj_id)
            
        print(f"Total Trials: {total}")
        print(f"Skipped (Speed > 0.7): {skipped}")
        print(f"Used Subjects ({len(subjects)}): {sorted(list(subjects))}")

except Exception as e:
    print(f"Error reading source: {e}")

print("\n--- Target Data (Stroke) ---")
try:
    with h5py.File(target_path, 'r') as f:
        subjects = set()
        total = 0
        for key in f.keys():
            total += 1
            # Extract Subject ID (Sxxx)
            # Format usually S002_T003
            parts = key.split('_')
            subj_id = parts[0]
            subjects.add(subj_id)
            
        print(f"Total Trials: {total}")
        print(f"Used Subjects ({len(subjects)}): {sorted(list(subjects))}")

except Exception as e:
    print(f"Error reading target: {e}")
