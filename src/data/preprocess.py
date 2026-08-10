import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

WESAD_DIR = 'data/raw/WESAD'
PROCESSED_DIR = 'data/processed'
SUBJECTS = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
LABEL_BASELINE = 1
LABEL_STRESS = 2


def resample_1d(signal, target_len):
    signal = np.asarray(signal).reshape(-1)
    if target_len <= 0:
        return np.array([], dtype=float)
    if signal.size == 0:
        return np.zeros(target_len, dtype=float)
    if signal.size == target_len:
        return signal.astype(float)

    source_x = np.linspace(0.0, 1.0, num=signal.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, signal).astype(float)


def acc_magnitude(acc_signal):
    acc_signal = np.asarray(acc_signal)
    if acc_signal.ndim == 2 and acc_signal.shape[1] >= 3:
        return np.sqrt((acc_signal[:, :3] ** 2).sum(axis=1))
    return acc_signal.reshape(-1)

def process_subject(subject):
    file_path = os.path.join(WESAD_DIR, subject, f'{subject}.pkl')
    if not os.path.exists(file_path):
        return None, None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    labels = data['label'].flatten()
    wrist = data['signal']['wrist']
    chest = data['signal'].get('chest', {})

    bvp = wrist['BVP'].flatten()
    eda = wrist['EDA'].flatten()
    temp = wrist.get('TEMP', np.zeros(1)).flatten()
    acc = acc_magnitude(wrist.get('ACC', np.zeros((1, 3))))
    resp = chest.get('Resp', np.zeros(1)).flatten()
    
    duration = len(labels) / 700.0
    target_len_64 = int(duration * 64)

    if target_len_64 <= 0:
        return None, None
    
    idx_labels = np.linspace(0, len(labels) - 1, target_len_64, dtype=int)
    labels_64hz = labels[idx_labels].astype(int)

    eda_64hz = resample_1d(eda, target_len_64)
    bvp_64hz = resample_1d(bvp, target_len_64)
    temp_64hz = resample_1d(temp, target_len_64)
    acc_64hz = resample_1d(acc, target_len_64)
    resp_64hz = resample_1d(resp, target_len_64)
    
    min_len = min(
        len(bvp_64hz),
        len(eda_64hz),
        len(temp_64hz),
        len(acc_64hz),
        len(resp_64hz),
        len(labels_64hz),
    )

    bvp_64hz = bvp_64hz[:min_len]
    eda_64hz = eda_64hz[:min_len]
    temp_64hz = temp_64hz[:min_len]
    acc_64hz = acc_64hz[:min_len]
    resp_64hz = resp_64hz[:min_len]
    labels_64hz = labels_64hz[:min_len]
    
    df = pd.DataFrame({
        'subject': subject,
        'EDA': eda_64hz,
        'BVP': bvp_64hz,
        'TEMP': temp_64hz,
        'ACC_MAG': acc_64hz,
        'RESP': resp_64hz,
        'label': labels_64hz
    })
    
    df_baseline = df[df['label'] == LABEL_BASELINE].copy()
    df_stress = df[df['label'] == LABEL_STRESS].copy()
    
    return df_baseline, df_stress

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_baseline = []
    all_stress = []
    
    print("Extracting Baseline & Stress states from WESAD Subjects...")
    for subj in tqdm(SUBJECTS):
        df_b, df_s = process_subject(subj)
        if df_b is not None:
            all_baseline.append(df_b)
            all_stress.append(df_s)
            
    if all_baseline:
        final_baseline = pd.concat(all_baseline, ignore_index=True)
        final_stress = pd.concat(all_stress, ignore_index=True)
        
        baseline_path = os.path.join(PROCESSED_DIR, 'wesad_baseline.csv')
        stress_path = os.path.join(PROCESSED_DIR, 'wesad_stress.csv')
        
        print("\nFinal Statistics:")
        print(f"- Saved {len(final_baseline)} synchronized Baseline samples -> {baseline_path}")
        print(f"- Saved {len(final_stress)} synchronized Stress samples -> {stress_path}")
        
        final_baseline.to_csv(baseline_path, index=False)
        final_stress.to_csv(stress_path, index=False)
        print("Preprocessing Complete!")

if __name__ == '__main__':
    main()