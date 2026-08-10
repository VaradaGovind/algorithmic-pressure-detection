import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


DEFAULT_WEARABLE_ROOT = (
    'data/raw/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/'
    'Wearable_Dataset'
)


def _band_power(freq, power, low, high):
    mask = (freq >= low) & (freq < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freq[mask]))


def _extract_hrv_features(bvp_signal, fs=64.0):
    features = {
        'hrv_rmssd': 0.0,
        'hrv_sdnn': 0.0,
        'hrv_pnn50': 0.0,
        'hrv_mean_hr': 0.0,
        'hrv_lf': 0.0,
        'hrv_hf': 0.0,
        'hrv_lf_hf': 0.0,
    }

    if len(bvp_signal) < int(fs * 10):
        return features

    peaks, _ = find_peaks(bvp_signal, distance=int(fs * 0.4))
    if len(peaks) < 3:
        return features

    ibi = np.diff(peaks) / fs
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


def _load_sensor_values(path):
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] <= 2:
        return np.array([])

    data = raw.iloc[2:].apply(pd.to_numeric, errors='coerce').dropna(how='any')
    if data.empty:
        return np.array([])
    return data.to_numpy(dtype=float)


def _load_signal(path):
    values = _load_sensor_values(path)
    if values.ndim == 1:
        return values
    if values.shape[1] == 1:
        return values[:, 0]
    return values


def _extract_session_features(session_dir):
    eda_path = os.path.join(session_dir, 'EDA.csv')
    bvp_path = os.path.join(session_dir, 'BVP.csv')
    temp_path = os.path.join(session_dir, 'TEMP.csv')
    acc_path = os.path.join(session_dir, 'ACC.csv')

    if not os.path.exists(eda_path) or not os.path.exists(bvp_path):
        return None

    eda = _load_signal(eda_path)
    bvp = _load_signal(bvp_path)
    temp = _load_signal(temp_path) if os.path.exists(temp_path) else np.array([])
    acc = _load_signal(acc_path) if os.path.exists(acc_path) else np.array([])

    if acc.ndim == 2 and acc.shape[1] == 3:
        acc = np.sqrt(np.sum(acc ** 2, axis=1))

    features = {
        'eda_mean': float(np.mean(eda)) if eda.size else 0.0,
        'eda_std': float(np.std(eda)) if eda.size else 0.0,
        'eda_min': float(np.min(eda)) if eda.size else 0.0,
        'eda_max': float(np.max(eda)) if eda.size else 0.0,
        'eda_num_peaks': float(len(find_peaks(eda, prominence=max(0.01, float(np.std(eda) * 0.1)))[0])) if eda.size else 0.0,
        'bvp_mean': float(np.mean(bvp)) if bvp.size else 0.0,
        'bvp_std': float(np.std(bvp)) if bvp.size else 0.0,
    }
    features.update(_extract_hrv_features(bvp))

    if temp.size:
        features['temp_mean'] = float(np.mean(temp))
        features['temp_std'] = float(np.std(temp))
        features['temp_slope'] = float(np.polyfit(np.arange(len(temp)), temp, 1)[0]) if len(temp) > 1 else 0.0
    else:
        features['temp_mean'] = 0.0
        features['temp_std'] = 0.0
        features['temp_slope'] = 0.0

    if acc.size:
        features['acc_mean'] = float(np.mean(acc))
        features['acc_std'] = float(np.std(acc))
        features['acc_rms'] = float(np.sqrt(np.mean(acc ** 2)))
    else:
        features['acc_mean'] = 0.0
        features['acc_std'] = 0.0
        features['acc_rms'] = 0.0

    features['subject'] = os.path.basename(session_dir)
    features['window_start_sec'] = 0.0
    return features


def build_real_external_dataset(source_dir, base_out, stress_out):
    wearable_root = os.path.join(source_dir, 'Wearable_Dataset') if os.path.isdir(os.path.join(source_dir, 'Wearable_Dataset')) else source_dir
    if not os.path.isdir(wearable_root):
        raise FileNotFoundError(f'Wearable dataset directory not found: {wearable_root}')

    rows = {'baseline': [], 'stress': []}
    for condition_name, target_bucket in [('AEROBIC', 'baseline'), ('ANAEROBIC', 'baseline'), ('STRESS', 'stress')]:
        condition_dir = os.path.join(wearable_root, condition_name)
        if not os.path.isdir(condition_dir):
            continue

        for subject_name in sorted(os.listdir(condition_dir)):
            session_dir = os.path.join(condition_dir, subject_name)
            if not os.path.isdir(session_dir):
                continue

            features = _extract_session_features(session_dir)
            if features is None:
                continue

            features['subject'] = f'EXT_{condition_name}_{subject_name}'
            rows[target_bucket].append(features)

    if not rows['baseline'] or not rows['stress']:
        raise ValueError('Could not build a real external dataset from the wearable corpus.')

    baseline_df = pd.DataFrame(rows['baseline']).fillna(0.0)
    stress_df = pd.DataFrame(rows['stress']).fillna(0.0)
    baseline_df.to_csv(base_out, index=False)
    stress_df.to_csv(stress_out, index=False)

    print(f'Real external baseline data saved to {base_out} ({len(baseline_df)} rows)')
    print(f'Real external stress data saved to {stress_out} ({len(stress_df)} rows)')


def build_synthetic_external_dataset(
    base_in='data/processed/features_baseline.csv',
    stress_in='data/processed/features_stress.csv',
    base_out='data/processed/external_features_baseline.csv',
    stress_out='data/processed/external_features_stress.csv',
):
    print('Loading internal feature datasets to generate a synthetic external proxy...')
    df_base = pd.read_csv(base_in)
    df_stress = pd.read_csv(stress_in)

    df_base_sim = df_base.sample(frac=0.6, random_state=42).copy()
    df_stress_sim = df_stress.sample(frac=0.6, random_state=42).copy()

    df_base_sim['subject'] = df_base_sim['subject'].apply(lambda x: f'SIM_EXT_{x}')
    df_stress_sim['subject'] = df_stress_sim['subject'].apply(lambda x: f'SIM_EXT_{x}')

    feature_cols = [c for c in df_base_sim.columns if c not in ['subject', 'window_start_sec']]
    np.random.seed(123)
    for col in feature_cols:
        col_std = df_base_sim[col].std()
        if pd.isna(col_std) or col_std == 0:
            continue
        shift = col_std * 0.08
        df_base_sim[col] += shift + np.random.normal(0, col_std * 0.06, size=len(df_base_sim))
        df_stress_sim[col] += shift + np.random.normal(0, col_std * 0.06, size=len(df_stress_sim))

    df_base_sim.to_csv(base_out, index=False)
    df_stress_sim.to_csv(stress_out, index=False)
    print(f'Saved simulated external baseline data: {base_out} ({len(df_base_sim)} rows)')
    print(f'Saved simulated external stress data: {stress_out} ({len(df_stress_sim)} rows)')


def simulate_external_dataset(source_dir=None):
    base_out = 'data/processed/external_features_baseline.csv'
    stress_out = 'data/processed/external_features_stress.csv'

    if source_dir is None:
        source_dir = DEFAULT_WEARABLE_ROOT

    if os.path.isdir(source_dir) or os.path.isdir(os.path.join(source_dir, 'Wearable_Dataset')):
        try:
            print('Building external dataset from the public wearable exercise/stress corpus...')
            build_real_external_dataset(source_dir, base_out, stress_out)
            return
        except Exception as exc:
            print(f'Warning: real external dataset build failed ({exc}); falling back to synthetic proxy.')

    build_synthetic_external_dataset(base_out=base_out, stress_out=stress_out)


def main():
    parser = argparse.ArgumentParser(description='Build an external cohort for cross-dataset validation.')
    parser.add_argument('--source-dir', default=None, help='Root directory for the public external wearable dataset.')
    args = parser.parse_args()
    simulate_external_dataset(args.source_dir)


if __name__ == '__main__':
    main()

