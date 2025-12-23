import os
import torch
import numpy as np
import joblib
import csv
from sklearn.metrics import r2_score, mean_squared_error
from sda_training import CONFIG, SDA_Dual_Model, load_target_data_by_trials

def calculate_metrics():
    results_dir = 'results'
    subjects = ['S002', 'S003', 'S004', 'S006', 'S007', 'S008', 'S013']
    modes = ['SDA', 'TO', 'SO', 'TL']
    
    # Store results
    data = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare CSV file
    if not os.path.exists('results_analysis'):
        os.makedirs('results_analysis')
    
    csv_file = 'results_analysis/aggregated_results.csv'
    fieldnames = ['Subject', 'Mode', 'Trial', 'RMSE', 'R2']
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for subj in subjects:
            subj_dir = os.path.join(results_dir, subj)
            if not os.path.exists(subj_dir):
                print(f"Skipping {subj} (not found)")
                continue
                
            folders = os.listdir(subj_dir)
            
            for f in folders:
                if 'frac1.0' not in f:
                    continue
                
                parts = f.split('_')
                
                mode = None
                for m in modes:
                    if f"_{m}_" in f:
                        mode = m
                        break
                
                if mode is None:
                    continue
                    
                test_trial = None
                for p in parts:
                    if p.startswith('T') and len(p) == 4 and p[1:].isdigit():
                        test_trial = f"{subj}_{p}" 
                        break
                
                if test_trial is None:
                    continue
                    
                print(f"Processing {subj} | {mode} | {test_trial} ...")
                
                model_path = os.path.join(subj_dir, f, 'final_model.pth')
                scaler_path = os.path.join(subj_dir, f, 'scaler.pkl')
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    continue
                    
                try:
                    CONFIG['target_subject'] = subj
                    
                    model = SDA_Dual_Model(CONFIG).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    
                    scaler = joblib.load(scaler_path)
                    
                    tgt_test_data, tgt_test_labels = load_target_data_by_trials(CONFIG, [test_trial])
                    
                    if len(tgt_test_data) == 0:
                        continue
                        
                    N, T, F_dim = tgt_test_data.shape
                    tgt_test_norm = scaler.transform(tgt_test_data.reshape(-1, F_dim)).reshape(N, T, F_dim)
                    
                    batch_x = torch.tensor(tgt_test_norm, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        preds, _ = model(batch_x, domain='target')
                        preds = preds.cpu().numpy()
                        
                    pred_phase = np.arctan2(preds[:, 1], preds[:, 0]) 
                    gt_phase = np.arctan2(tgt_test_labels[:, 1], tgt_test_labels[:, 0])
                    
                    pred_phase_cont = np.unwrap(pred_phase)
                    gt_phase_cont = np.unwrap(gt_phase)
                    
                    pred_pct = pred_phase_cont / (2 * np.pi)
                    gt_pct = gt_phase_cont / (2 * np.pi)
                    
                    rmse = np.sqrt(mean_squared_error(gt_pct, pred_pct))
                    r2 = r2_score(gt_pct, pred_pct)
                    
                    row = {
                        'Subject': subj,
                        'Mode': mode,
                        'Trial': test_trial,
                        'RMSE': rmse,
                        'R2': r2
                    }
                    data.append(row)
                    writer.writerow(row)
                    
                except Exception as e:
                    print(f"  Error processing {f}: {e}")

    # Basic summarization
    summary = {}
    for d in data:
        k = (d['Subject'], d['Mode'])
        if k not in summary: summary[k] = {'rmse': [], 'r2': []}
        summary[k]['rmse'].append(d['RMSE'])
        summary[k]['r2'].append(d['R2'])
        
    print("\nSummary (Mean):")
    print(f"{'Subject':<10} {'Mode':<10} {'RMSE':<10} {'R2':<10}")
    for k, v in summary.items():
        print(f"{k[0]:<10} {k[1]:<10} {np.mean(v['rmse']):.4f}     {np.mean(v['r2']):.4f}")

if __name__ == "__main__":
    calculate_metrics()
