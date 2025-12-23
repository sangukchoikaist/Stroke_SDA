import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
results_dir = 'd:/RSC lab/Codes/Stroke_SDA/results_analysis/Batch_SDA_Effect'
subjects = ['S003', 'S004', 'S006', 'S007', 'S008', 'S013']
display_names = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006'] # User requested Mapping
output_file = os.path.join(results_dir, 'combined_tsne_effect.png')

def plot_combined_tsne():
    fig, axes = plt.subplots(2, 6, figsize=(20, 7)) # 6 cols, 2 rows
    # Adjust layout to make room for top legend and left labels
    plt.subplots_adjust(wspace=0.1, hspace=0.2, left=0.08, right=0.98, top=0.88, bottom=0.1)
    
    # 1. Determine Global Axis Limits
    all_x = []
    all_y = []
    
    # Pre-load to find limits
    for subj in subjects:
        data_path = os.path.join(results_dir, subj, f'tsne_data_{subj}.npz')
        if os.path.exists(data_path):
            data = np.load(data_path)
            all_x.extend(data['emb_so'][:, 0])
            all_x.extend(data['emb_sda'][:, 0])
            all_y.extend(data['emb_so'][:, 1])
            all_y.extend(data['emb_sda'][:, 1])
            
    if not all_x:
        print("No data found!")
        return

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    # Add some padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # 3. Plotting
    rows = ["Before SDA", "After SDA"]

    handles = []
    labels = []

    for col_idx, (subj, disp_name) in enumerate(zip(subjects, display_names)):
        data_path = os.path.join(results_dir, subj, f'tsne_data_{subj}.npz')
        
        if not os.path.exists(data_path):
            print(f"Data for {subj} not found at {data_path}")
            axes[0, col_idx].axis('off')
            axes[1, col_idx].axis('off')
            continue
            
        data = np.load(data_path)
        
        emb_so = data['emb_so']
        y_so = data['y_so']
        emb_sda = data['emb_sda']
        y_sda = data['y_sda']
        
        # --- Row 0: Before SDA ---
        ax0 = axes[0, col_idx]
        sc1 = ax0.scatter(emb_so[y_so==0, 0], emb_so[y_so==0, 1], c='blue', alpha=0.3, s=5, label='Non-disabled')
        sc2 = ax0.scatter(emb_so[y_so==1, 0], emb_so[y_so==1, 1], c='red', alpha=0.3, s=5, label='Post-stroke')
        
        ax0.set_xlim(xlim)
        ax0.set_ylim(ylim)
        
        if col_idx == 0:
            ax0.set_ylabel("t-SNE 2", fontsize=10) # Normal font
            # Row Label (Bold) placed outside manually later
        else:
             # Hide Y ticks for S002 onwards
             ax0.set_yticks([])
             ax0.set_ylabel("")

        ax0.set_title(disp_name, fontsize=14, fontweight='bold')
        
        # Collect handles for legend
        if col_idx == 0:
            handles = [sc1, sc2]
            labels = ['Non-disabled', 'Post-stroke']

        # --- Row 1: After SDA ---
        ax1 = axes[1, col_idx]
        ax1.scatter(emb_sda[y_sda==0, 0], emb_sda[y_sda==0, 1], c='blue', alpha=0.3, s=5, label='Non-disabled')
        ax1.scatter(emb_sda[y_sda==1, 0], emb_sda[y_sda==1, 1], c='red', alpha=0.3, s=5, label='Post-stroke')
        
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        
        if col_idx == 0:
             ax1.set_ylabel("t-SNE 2", fontsize=10) # Normal font
        else:
             ax1.set_yticks([])
             ax1.set_ylabel("")
        
        # Add x labels to bottom row
        ax1.set_xlabel("t-SNE 1", fontsize=10)

    # 4. Global Row Labels (Bold)
    # Get positions of first axes in each row to align text
    pos0 = axes[0, 0].get_position()
    pos1 = axes[1, 0].get_position()
    
    # Place text to the left of the ylabel
    # x approx 0.02 (left margin is 0.08)
    fig.text(0.02, (pos0.y0 + pos0.y1) / 2, "Before SDA", 
             va='center', ha='center', rotation=90, fontsize=14, fontweight='bold')
    fig.text(0.02, (pos1.y0 + pos1.y1) / 2, "After SDA", 
             va='center', ha='center', rotation=90, fontsize=14, fontweight='bold')

    # 5. Global Legend
    # "글씨 크기랑 dot 크기를 조금 더 키워주고" -> fontsize=14, markerscale=2.0
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=2, fontsize=14, frameon=False, markerscale=2.5) # Increased markerscale
    
    print(f"Saving combined plot to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_combined_tsne()
