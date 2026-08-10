import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch
from tqdm import tqdm

PROCESSED_DIR = 'data/processed'
WINDOW_SIZE_SEC = 60
STEP_SIZE_SEC = 10
FS = 64
WINDOW_SAMPLES = int(WINDOW_SIZE_SEC * FS)
STEP_SAMPLES = int(STEP_SIZE_SEC * FS)

def _band_power(freq, power, low, high):
    mask = (freq >= low) & (freq < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freq[mask]))


def _signal_or_zeros(df, column, length):
    if column in df.columns:
        return df[column].to_numpy(dtype=float)
    return np.zeros(length, dtype=float)


def extract_hrv_features(bvp_window):
    features = {
        'hrv_rmssd': 0.0,
        'hrv_sdnn': 0.0,
        'hrv_pnn50': 0.0,
        'hrv_mean_hr': 0.0,
        'hrv_lf': 0.0,
        'hrv_hf': 0.0,
        'hrv_lf_hf': 0.0,
    }

    bvp_peaks, _ = find_peaks(bvp_window, distance=int(FS * 0.4))
    if len(bvp_peaks) < 3:
        return features

    ibi = np.diff(bvp_peaks) / FS
    if ibi.size < 2:
        return features

    diff_ibi = np.diff(ibi)
    features['hrv_rmssd'] = float(np.sqrt(np.mean(diff_ibi ** 2))) if diff_ibi.size else 0.0
    features['hrv_sdnn'] = float(np.std(ibi))
    features['hrv_pnn50'] = float(np.mean(np.abs(diff_ibi) > 0.05)) if diff_ibi.size else 0.0

    mean_ibi = float(np.mean(ibi))
    features['hrv_mean_hr'] = float(60.0 / mean_ibi) if mean_ibi > 1e-8 else 0.0

    beat_times = np.cumsum(ibi)
    if beat_times.size < 4 or (beat_times[-1] - beat_times[0]) <= 1.0:
        return features

    interp_fs = 4.0
    interp_t = np.arange(beat_times[0], beat_times[-1], 1.0 / interp_fs)
    if interp_t.size < 8:
        return features

    ibi_interp = np.interp(interp_t, beat_times, ibi)
    ibi_interp = ibi_interp - np.mean(ibi_interp)
    nperseg = min(256, ibi_interp.size)
    if nperseg < 8:
        return features

    freq, power = welch(ibi_interp, fs=interp_fs, nperseg=nperseg)
    lf = _band_power(freq, power, 0.04, 0.15)
    hf = _band_power(freq, power, 0.15, 0.40)

    features['hrv_lf'] = lf
    features['hrv_hf'] = hf
    features['hrv_lf_hf'] = float(lf / hf) if hf > 1e-8 else 0.0
    return features


def extract_features_from_window(eda_window, bvp_window, temp_window, acc_window, resp_window):
    features = {}

    features['eda_mean'] = float(np.mean(eda_window))
    features['eda_std'] = float(np.std(eda_window))
    features['eda_min'] = float(np.min(eda_window))
    features['eda_max'] = float(np.max(eda_window))

    eda_prominence = max(0.01, float(np.std(eda_window) * 0.1))
    eda_peaks, _ = find_peaks(eda_window, prominence=eda_prominence)
    features['eda_num_peaks'] = int(len(eda_peaks))

    features['bvp_mean'] = float(np.mean(bvp_window))
    features['bvp_std'] = float(np.std(bvp_window))
    features['bvp_min'] = float(np.min(bvp_window))
    features['bvp_max'] = float(np.max(bvp_window))

    features.update(extract_hrv_features(bvp_window))

    features['temp_mean'] = float(np.mean(temp_window))
    features['temp_std'] = float(np.std(temp_window))
    features['temp_slope'] = float(np.polyfit(np.arange(len(temp_window)), temp_window, 1)[0])

    features['acc_mean'] = float(np.mean(acc_window))
    features['acc_std'] = float(np.std(acc_window))
    features['acc_rms'] = float(np.sqrt(np.mean(acc_window ** 2)))

    features['resp_mean'] = float(np.mean(resp_window))
    features['resp_std'] = float(np.std(resp_window))

    return features

def process_file(input_file, output_file):
    print(f"Processing {input_file} for feature engineering...")
    df = pd.read_csv(input_file)
    extracted_data = []

    required_cols = {'subject', 'EDA', 'BVP'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_file}: {sorted(missing)}")
    
    for subject in tqdm(sorted(df['subject'].unique())):
        subj_df = df[df['subject'] == subject]
        subject_len = len(subj_df)
        if subject_len < WINDOW_SAMPLES:
            continue

        eda_sig = _signal_or_zeros(subj_df, 'EDA', subject_len)
        bvp_sig = _signal_or_zeros(subj_df, 'BVP', subject_len)
        temp_sig = _signal_or_zeros(subj_df, 'TEMP', subject_len)
        acc_sig = _signal_or_zeros(subj_df, 'ACC_MAG', subject_len)
        resp_sig = _signal_or_zeros(subj_df, 'RESP', subject_len)
        
        for start_idx in range(0, len(eda_sig) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            end_idx = start_idx + WINDOW_SAMPLES
            eda_win = eda_sig[start_idx:end_idx]
            bvp_win = bvp_sig[start_idx:end_idx]
            temp_win = temp_sig[start_idx:end_idx]
            acc_win = acc_sig[start_idx:end_idx]
            resp_win = resp_sig[start_idx:end_idx]
            
            feats = extract_features_from_window(eda_win, bvp_win, temp_win, acc_win, resp_win)
            feats['subject'] = subject
            feats['window_start_sec'] = float(start_idx / FS)
            extracted_data.append(feats)
            
    out_df = pd.DataFrame(extracted_data)
    out_df.to_csv(output_file, index=False)
    print(f"Saved {len(out_df)} windows to {output_file}")


def main():
    base_in = os.path.join(PROCESSED_DIR, 'wesad_baseline.csv')
    stress_in = os.path.join(PROCESSED_DIR, 'wesad_stress.csv')
    
    base_out = os.path.join(PROCESSED_DIR, 'features_baseline.csv')
    stress_out = os.path.join(PROCESSED_DIR, 'features_stress.csv')
    
    process_file(base_in, base_out)
    process_file(stress_in, stress_out)

if __name__ == '__main__':
    main()
