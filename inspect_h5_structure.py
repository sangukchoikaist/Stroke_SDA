
import h5py

def inspect_h5(path):
    print(f"--- Inspecting {path} ---")
    try:
        with h5py.File(path, 'r') as f:
            def print_attrs(name, obj):
                print(name)
                if hasattr(obj, 'shape'):
                    print(f"  Shape: {obj.shape}")
                    print(f"  Type: {obj.dtype}")
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error reading {path}: {e}")

inspect_h5('d:/RSC lab/Codes/Stroke_SDA/output/all_subjects_dataset_ds_hip.h5')
inspect_h5('d:/RSC lab/Codes/Stroke_SDA/output/stroke_dataset_hip.h5')
