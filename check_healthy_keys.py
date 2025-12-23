
import h5py

path = 'd:/RSC lab/Codes/Stroke_SDA/output/all_subjects_dataset_ds_hip.h5'
try:
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        print(f"Total keys: {len(keys)}")
        print(f"First 5 keys: {keys[:5]}")
        if keys:
            first_group = f[keys[0]]
            print(f"Keys in {keys[0]}: {list(first_group.keys())}")
except Exception as e:
    print(f"Error: {e}")
