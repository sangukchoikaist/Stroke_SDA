import h5py

source_path = 'd:/RSC lab/Codes/Stroke_SDA/output/all_subjects_dataset_ds_hip.h5'
target_path = 'd:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5'

def check_key(path, key_suffix):
    print(f"Checking {path} for keys ending in '{key_suffix}'...")
    found = False
    with h5py.File(path, 'r') as f:
        # Check first valid group
        for grp_name in f.keys():
            grp = f[grp_name]
            if key_suffix in grp:
                print(f"  FOUND: '{key_suffix}' in group '{grp_name}'")
                found = True
                break
    if not found:
        print(f"  NOT FOUND: '{key_suffix}'")

if __name__ == "__main__":
    check_key(source_path, 'theta_est')
    check_key(target_path, 'paretic_theta_est')
