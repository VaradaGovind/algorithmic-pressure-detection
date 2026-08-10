import pandas as pd
import numpy as np
import os

def generate_kpi_data(scores_path='results/cv_window_scores.csv', output_path='data/raw/synthetic_kpi.csv'):
    print(f"Loading scores from {scores_path}...")
    df = pd.read_csv(scores_path)
    
    # Emulate the logic in outcome_linkage.py to create time_bins
    if 'window_start_sec' not in df.columns:
        df['window_start_sec'] = np.arange(len(df), dtype=float)
        
    bin_size_sec = 300
    df['time_bin'] = (df['window_start_sec'] // bin_size_sec).astype(int)
    
    # Calculate true stress aggregates to build correlated KPIs
    grouped = df.groupby(['subject', 'time_bin'])['anomaly_score'].mean().reset_index()
    grouped.rename(columns={'anomaly_score': 'stress_mean'}, inplace=True)
    
    np.random.seed(42)
    kpi_records = []
    
    for _, row in grouped.iterrows():
        subject = row['subject']
        time_bin = row['time_bin']
        stress = row['stress_mean']
        
        # Base defect rate: 1.0% + random noise + correlated stress effect
        # Assuming stress is roughly bounded between 0 and 1
        defect_rate = max(0.0, 0.01 + 0.05 * stress + np.random.normal(0, 0.005))
        
        # Delay flag: logistic relationship with stress
        prob_delay = 1.0 / (1.0 + np.exp(-(-2.0 + 5.0 * stress)))
        delay_flag = int(np.random.rand() < prob_delay)
        
        # Cycle time: baseline 60s + penalty for stress
        cycle_time = max(30.0, 60.0 + 20.0 * stress + np.random.normal(0, 5.0))
        
        kpi_records.append({
            'subject': subject,
            'time_bin': time_bin,
            'defect_rate': defect_rate,
            'delay_flag': delay_flag,
            'cycle_time': cycle_time
        })
        
    kpi_df = pd.DataFrame(kpi_records)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kpi_df.to_csv(output_path, index=False)
    print(f"Successfully generated synthetic KPI data at {output_path} with {len(kpi_df)} rows.")

if __name__ == '__main__':
    generate_kpi_data()
