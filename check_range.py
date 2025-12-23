
import h5py
import numpy as np

path = 'd:/RSC lab/Codes/Stroke_SDA/output/all_subjects_dataset_ds_hip.h5'

try:
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        k = keys[0]
        data = f[k]['gcR_hs'][:]
        print(f"Key: {k}")
        print(f"Range: {np.min(data)} to {np.max(data)}")
        print(f"Mean: {np.mean(data)}")
        
        path_s = 'd:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5'
        with h5py.File(path_s, 'r') as f2:
            ks = list(f2.keys())
            k2 = ks[0]
            data2 = f2[k2]['gc_hs'][:]
            print(f"Stroke Key: {k2}")
            print(f"Stroke Range: {np.min(data2)} to {np.max(data2)}")

except Exception as e:
    print(e)
