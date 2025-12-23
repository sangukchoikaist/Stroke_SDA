
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials

def visualize_test_trial():
    # Target specific fold results
    subject = CONFIG['target_subject']
    
    # Auto-detect the folder for T006 (Fold 3) which we know is a test trial from training logs
    test_trial = 'S003_T002' 
    fold_dir = f"results/{subject}/Fold_1_{test_trial}"
    
    # Fallback if folder doesn't exist (e.g. if I picked the wrong fold index)
    if not os.path.exists(fold_dir):
        print(f"Error: Folder {fold_dir} not found. Checking results dir...")
        # Simple search
        base_results = f"results/{subject}"
        found = False
        if os.path.exists(base_results):
            for d in os.listdir(base_results):
                if test_trial in d:
                    fold_dir = os.path.join(base_results, d)
                    found = True
                    break
        if not found:
            print(f"Could not find result folder for {test_trial}")
            return

    print(f"Visualizing results from: {fold_dir}")
    
    model_path = os.path.join(fold_dir, 'final_model.pth')
    scaler_path = os.path.join(fold_dir, 'scaler.pkl')
    
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SDA_Dual_Model(CONFIG).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # 2. Load Scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")
    
    # 3. Load Test Data
    print(f"Loading Test Trial: {test_trial}")
    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
    
    if len(tgt_test_data) == 0:
        print("No data found for test trial.")
        return

    # 4. Normalize
    N, T, F = tgt_test_data.shape
    tgt_test_norm = scaler.transform(tgt_test_data.reshape(-1, F)).reshape(N, T, F)
    
    # 5. Inference
    batch_x = torch.tensor(tgt_test_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds, _ = model(batch_x, domain='target')
        preds = preds.cpu().numpy()
        
    # 6. Process Predictions
    pred_phase = np.arctan2(preds[:, 1], preds[:, 0])
    gt_phase = np.arctan2(tgt_test_labels[:, 1], tgt_test_labels[:, 0])
    
    pred_phase_cont = np.unwrap(pred_phase)
    gt_phase_cont = np.unwrap(gt_phase)
    
    # 7. Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(tgt_test_labels[:, 0], tgt_test_labels[:, 1], 'g-', label='GT', alpha=0.5)
    plt.plot(preds[:, 0], preds[:, 1], 'b--', label='Pred', alpha=0.7)
    plt.title(f'Phase Portrait ({test_trial})')
    plt.legend()
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(gt_phase_cont, 'g-', label='GT', linewidth=2)
    plt.plot(pred_phase_cont, 'b--', label='Pred', linewidth=1.5)
    plt.title(f'Phase Estimation ({test_trial})')
    plt.legend()
    
    outfile = f'plot_{subject}_test.png'
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Saved {outfile}")
    plt.close()

if __name__ == "__main__":
    visualize_test_trial()
