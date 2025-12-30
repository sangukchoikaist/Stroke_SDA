
import pandas as pd

def analyze_results():
    log_file = 'grid_search_log_theta_v3.csv'
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"File {log_file} not found.")
        return

    # Filter
    df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')
    df = df.dropna(subset=['MSE'])
    
    # User Request: Enc <= 128, Dec <= 64
    df = df[df['EncDim'] <= 128]
    df = df[df['DecDim'] <= 64]
    
    # Group by hyperparameters
    param_cols = ['LambdaMMD', 'LambdaSrc', 'LR', 'EncDim', 'DecDim', 'Dropout']
    grouped = df.groupby(param_cols)['MSE'].agg(['mean', 'std', 'count', 'min', 'max'])
    
    # Filter only complete runs (Count = 6)
    complete_runs = grouped[grouped['count'] == 6].copy()
    
    # Score for Robustness
    complete_runs['Robustness'] = complete_runs['mean'] + complete_runs['std']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(f"\n=== Top 5 Configurations [Enc<=128, Dec<=64] by Mean MSE ===")
    print(complete_runs.sort_values(by='mean').head(5)[['mean', 'std', 'min', 'max', 'Robustness']])
    
    print(f"\n=== Top 5 Configurations [Enc<=128, Dec<=64] by Robustness ===")
    print(complete_runs.sort_values(by='Robustness').head(5)[['mean', 'std', 'min', 'max', 'Robustness']])

if __name__ == "__main__":
    analyze_results()
