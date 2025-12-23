
import h5py
import numpy as np

target_h5 = 'd:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5'

def list_subjects():
    subjects = set()
    with h5py.File(target_h5, 'r') as f:
        for key in f.keys():
            # Keys are like 'S003_T002'
            parts = key.split('_')
            if len(parts) >= 1:
                subj = parts[0]
                subjects.add(subj)
    
    print(f"Found {len(subjects)} subjects: {sorted(list(subjects))}")

if __name__ == "__main__":
    list_subjects()
