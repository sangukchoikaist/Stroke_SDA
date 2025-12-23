import pandas as pd

def analyze_results():
    try:
        df = pd.read_csv('grid_search_log.csv')
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    subjects = df['Subject'].unique()
    
    print("=== Best Parameters (Average MSE across Subjects) ===")
    
    # Group by stored parameters (excluding Subject and MSE)
    # Columns: Subject,Lambda,LR,EncDim,DecDim,Dropout,MSE
    group_cols = ['Lambda', 'LR', 'EncDim', 'DecDim', 'Dropout']
    
    # Calculate Mean MSE and Count (to ensure we have results for all subjects)
    grouped = df.groupby(group_cols)['MSE'].agg(['mean', 'count', 'std']).reset_index()
    
    # Filter: Only consider params that have results for all analyzed subjects (usually 3)
    # We find the max count to know how many subjects participated
    max_subjects = grouped['count'].max()
    valid_groups = grouped[grouped['count'] == max_subjects]
    
    print(f"Filtering for combinations present in all {max_subjects} subjects...")
    
    # Sort by Mean MSE
    sorted_df = valid_groups.sort_values(by='mean')
    
    # Print Top 5
    print("\nTop 5 Parameter Sets:")
    print(sorted_df.head(5).to_string(index=False))
    
    best = sorted_df.iloc[0]
    print(f"\n>>> Best Overall Combined MSE: {best['mean']:.6f} <<<")
    print(f"  Lambda: {best['Lambda']}")
    print(f"  LR: {best['LR']}")
    print(f"  EncDim: {best['EncDim']}")
    print(f"  DecDim: {best['DecDim']}")
    print(f"  Dropout: {best['Dropout']}")
    
    # Detailed breakdown for best params
    print("\n=== Individual Subject Performance (Best Config) ===")
    mask = (
        (df['Lambda'] == best['Lambda']) & 
        (df['LR'] == best['LR']) & 
        (df['EncDim'] == best['EncDim']) & 
        (df['DecDim'] == best['DecDim']) & 
        (df['Dropout'] == best['Dropout'])
    )
    detail_df = df[mask][['Subject', 'MSE']].sort_values(by='Subject')
    print(detail_df.to_string(index=False))

    # Specific analysis for S004
    print("\n=== Best Parameters for S004 ===")
    s004_df = df[df['Subject'] == 'S004']
    if not s004_df.empty:
        best_s004 = s004_df.loc[s004_df['MSE'].idxmin()]
        print(f"Min MSE: {best_s004['MSE']:.6f}")
        print(f"Lambda: {best_s004['Lambda']}")
        print(f"LR: {best_s004['LR']}")
        print(f"EncDim: {best_s004['EncDim']}")
        print(f"DecDim: {best_s004['DecDim']}")
        print(f"Dropout: {best_s004['Dropout']}")


if __name__ == "__main__":
    analyze_results()
