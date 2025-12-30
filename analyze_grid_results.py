
import pandas as pd

def analyze_results():
    log_file = 'grid_search_log_theta_v3.csv'
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"File {log_file} not found.")
        return

    # Convert MSE to numeric, forcing errors to NaN
    df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')
    df = df.dropna(subset=['MSE'])
    
    # 1. Best per Subject
    print("=== Best Combination per Subject ===")
    subjects = df['Subject'].unique()
    for subj in subjects:
        subj_df = df[df['Subject'] == subj]
        best_run = subj_df.loc[subj_df['MSE'].idxmin()]
        print(f"\nSubject {subj}:")
        print(f"  Min MSE: {best_run['MSE']:.6f}")
        print(f"  Params: MMD={best_run['LambdaMMD']}, Src={best_run['LambdaSrc']}, LR={best_run['LR']}, Enc={best_run['EncDim']}, Dec={best_run['DecDim']}, Drop={best_run['Dropout']}")

    # 2. Best Overall (Average MSE across all subjects)
    # Group by hyperparameters
    param_cols = ['LambdaMMD', 'LambdaSrc', 'LR', 'EncDim', 'DecDim', 'Dropout']
    grouped = df.groupby(param_cols)['MSE'].agg(['mean', 'std', 'count', 'min', 'max'])
    
    # Filter only combinations that have results for ALL 6 subjects
    # This ensures we don't pick a "lucky" param that only ran on the easiest subject.
    complete_runs = grouped[grouped['count'] == 6].copy()
    
    if len(complete_runs) == 0:
        print("No parameter combination has results for all 6 subjects yet.")
        print("Showing top results based on available data (Count >= 1):")
        complete_runs = grouped
    
    # Score for Robustness: Mean + Std (Lower is better)
    complete_runs['Robustness'] = complete_runs['mean'] + complete_runs['std']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(f"\n\n=== Top 10 Configurations by Mean MSE (Lower is Better) ===")
    print(complete_runs.sort_values(by='mean').head(10)[['mean', 'std', 'min', 'max', 'Robustness']])

    print(f"\n\n=== Bottom 10 Configurations by Mean MSE (Worst Performance) ===")
    print(complete_runs.sort_values(by='mean', ascending=False).head(10)[['mean', 'std', 'min', 'max', 'Robustness']])

    print(f"\n\n=== Top 10 Configurations by Robustness (Mean + Std) (Stability Focus) ===")
    print(complete_runs.sort_values(by='Robustness').head(10)[['mean', 'std', 'min', 'max', 'Robustness']])

if __name__ == "__main__":
    analyze_results()
