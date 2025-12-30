
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.manifold import TSNE
from PIL import Image

# Import shared modules
from sda_training import SDA_Dual_Model, CONFIG, load_target_data_by_trials, get_target_trials, load_source_data

# Config
RESULTS_DIR = 'results_final_rank1'
OUTPUT_DIR = 'paper_work/figure'
CONFIG['device'] = 'cpu' # Use CPU for inference/plotting to avoid OOM or conflicts

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def copy_loss_curve():
    print(">>> Generating Loss Curve...")
    # Find a representative S003 SDA Fold 1 loss plot
    subj = 'S003'
    subj_dir = os.path.join(RESULTS_DIR, subj)
    if not os.path.exists(subj_dir):
        print(f"Subject {subj} not found.")
        return

    # Look for Fold 1 SDA
    found = False
    for d in os.listdir(subj_dir):
        if "Fold_1" in d and "_SDA_" in d and "beta" not in d: # Standard SDA
            src = os.path.join(subj_dir, d, 'loss_dynamics.png')
            if os.path.exists(src):
                dst = os.path.join(OUTPUT_DIR, 'loss_curve.png')
                shutil.copy(src, dst)
                print(f"Copied {src} to {dst}")
                found = True
                break
    if not found:
        print("Loss curve not found.")

def extract_features(model, data, domain='target'):
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    features = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            _, z = model(batch, domain=domain)
            features.append(z.numpy())
    return np.concatenate(features, axis=0)

import subprocess

def generate_tsne_plot():
    print(">>> Generating t-SNE Plot (using external script)...")
    # Make sure data is generated first? visualize_tsne_batch should be run before this script or this script should run it.
    # The user instruction was just to use the "previous" graph.
    # Assuming visualize_tsne_batch was run separately or we explicitly run it here?
    # Let's run data generation first just in case?
    # But visualize_tsne_batch takes time. Let's assume I run it manually or add it here.
    
    # Run plot_tsne_combined.py
    try:
        subprocess.run(["/home/sangukchoi/.conda/envs/torch_gpu/bin/python", "plot_tsne_combined.py"], check=True)
        # Copy to destination if not already there? 
        # plot_tsne_combined saves to 'results_analysis/Batch_SDA_Effect/combined_tsne_effect.png'
        # We need to move it to OUTPUT_DIR
        src = 'results_analysis/Batch_SDA_Effect/combined_tsne_effect.png'
        dst = os.path.join(OUTPUT_DIR, 'combined_tsne_effect.png')
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print("External script output not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running plot_tsne_combined.py: {e}")

def copy_bar_plot():
    print(">>> Copying Bar Plots...")
    
    # RMSE
    src_rmse = 'paper_plot_bar_rmse.png'
    dst_rmse = os.path.join(OUTPUT_DIR, 'result_rmse.png')
    if os.path.exists(src_rmse):
        shutil.copy(src_rmse, dst_rmse)
        print(f"Copied {src_rmse} to {dst_rmse}")
    else:
        print("paper_plot_bar_rmse.png not found.")

    # Max Error
    src_max = 'paper_plot_bar_max.png'
    dst_max = os.path.join(OUTPUT_DIR, 'result_max_error.png')
    if os.path.exists(src_max):
        shutil.copy(src_max, dst_max)
        print(f"Copied {src_max} to {dst_max}")
    else:
        print("paper_plot_bar_max.png not found.")

def combine_time_domain_plots():
    print(">>> Combining Time Domain Plots...")
    f1 = 'paper_plot_trajectory.png'
    f2 = 'paper_plot_timeseries.png'
    
    if not os.path.exists(f1) or not os.path.exists(f2):
        print("Source plots not found.")
        return

    # Use PIL to combine images simply
    img1 = Image.open(f1)
    img2 = Image.open(f2)
    
    # Resize to match width? 
    # img1 is 6x6 (squared), img2 is 10x4.
    # Let's stack them vertically or place them side by side?
    # Paper (a) Portrait, (b) Time Series. 
    # Let's stack vertically for now or adjust.
    # Actually, matplotlib is better for labeling (a) and (b).
    
    # Just stitch them vertically
    # Resize img2 to match img1 width if needed, or vice versa
    # Let's target a fixed width, say 1800 px
    base_width = 1800
    
    def resize_width(img, w):
        wpercent = (w / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        return img.resize((w, hsize), Image.Resampling.LANCZOS)
    
    img1 = resize_width(img1, base_width // 2) # Trajectory is usually square
    img2 = resize_width(img2, base_width)      # Time series is wide
    
    # Actually, side-by-side might be tight.
    # Trajectory (Square) | TimeSeries (Rect)
    # Let's put Trajectory Left, TimeSeries Right.
    # But TimeSeries is wide.
    # Let's stitch: [Traj] [   TimeSeries   ]
    
    # Re-read: Traj is 6x6, TS is 10x4.
    # H ratios: 6 vs 4.
    # If we force same height?
    
    tot_h = 1200
    def resize_height(img, h):
        wpercent = (h / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(wpercent)))
        return img.resize((wsize, h), Image.Resampling.LANCZOS)
    
    # Resize match height
    i1 = resize_height(img1, tot_h)
    i2 = resize_height(img2, tot_h)
    
    combined = Image.new('RGB', (i1.width + i2.width, tot_h), (255, 255, 255))
    combined.paste(i1, (0, 0))
    combined.paste(i2, (i1.width, 0))
    
    out_path = os.path.join(OUTPUT_DIR, 'gaitphase_time_domain.png')
    combined.save(out_path)
    print(f"Saved combined plot to {out_path}")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    
    try: copy_loss_curve()
    except Exception as e: print(f"Error copying loss curve: {e}")
    
    try: copy_bar_plot()
    except Exception as e: print(f"Error copying bar plot: {e}")
    
    try: combine_time_domain_plots()
    except Exception as e: print(f"Error combining time plots: {e}")
    
    # Run t-SNE last as it takes time
    try: generate_tsne_plot()
    except Exception as e: print(f"Error generating t-SNE: {e}")

    print("Done generating paper figures.")
