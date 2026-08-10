import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.stattools import grangercausalitytests


def load_analysis_frame(stress_agg_path, outcome, kpi_file=None, bin_size_sec=300):
    merged = pd.read_csv(stress_agg_path)
    merged['subject'] = merged['subject'].astype(str)

    if kpi_file is not None and outcome not in merged.columns:
        kpi_df = pd.read_csv(kpi_file)
        if 'subject' not in kpi_df.columns:
            raise ValueError("KPI file must include a 'subject' column.")
        kpi_df['subject'] = kpi_df['subject'].astype(str)

        if 'time_bin' not in kpi_df.columns:
            if 'window_start_sec' in kpi_df.columns:
                kpi_df['time_bin'] = (kpi_df['window_start_sec'] // bin_size_sec).astype(int)
            elif 'timestamp_sec' in kpi_df.columns:
                kpi_df['time_bin'] = (kpi_df['timestamp_sec'] // bin_size_sec).astype(int)
            else:
                raise ValueError("KPI file must include one of: 'time_bin', 'window_start_sec', or 'timestamp_sec'.")
        else:
            kpi_df['time_bin'] = kpi_df['time_bin'].astype(int)

        merged = merged.merge(kpi_df, on=['subject', 'time_bin'], how='inner')

    if merged.empty:
        raise ValueError('No overlap found for causal analysis. Check subject IDs and time bins.')

    merged = merged.sort_values(['subject', 'time_bin']).reset_index(drop=True)
    return merged


def add_lagged_features(df, outcome, treatment='stress_mean', lag=1):
    frame = df.copy()
    frame[f'{treatment}_lag{lag}'] = frame.groupby('subject')[treatment].shift(lag)
    if outcome in frame.columns:
        frame[f'{outcome}_lag{lag}'] = frame.groupby('subject')[outcome].shift(lag)
    return frame


def run_granger_by_subject(df, feature, outcome, maxlag=3):
    rows = []
    for subject, subject_df in df.groupby('subject'):
        series = subject_df.sort_values('time_bin')[[outcome, feature]].dropna()
        if series.shape[0] < (maxlag + 5):
            continue

        try:
            result = grangercausalitytests(series[[outcome, feature]].to_numpy(), maxlag=maxlag, verbose=False)
        except Exception:
            continue

        for lag, lag_result in result.items():
            f_stat, p_value, _, _ = lag_result[0]['ssr_ftest']
            rows.append(
                {
                    'subject': subject,
                    'lag': lag,
                    'f_stat': f_stat,
                    'p_value': p_value,
                }
            )

    return pd.DataFrame(rows)


def propensity_score_match(merged_df, treatment_col='high_pressure_flag', covariates=None, n_neighbors=1):
    if covariates is None:
        covariates = ['stress_mean', 'stress_std', 'stress_max']

    df = merged_df.dropna(subset=[treatment_col] + covariates).copy().reset_index(drop=False)
    if df.empty:
        return None

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    if treated.empty or control.empty:
        return None

    lr = LogisticRegression(max_iter=300)
    X = df[covariates].to_numpy()
    y = df[treatment_col].to_numpy()
    lr.fit(X, y)
    ps = lr.predict_proba(X)[:, 1]

    control_positions = control.index.to_numpy()
    control_ps = ps[control_positions]
    treated_positions = treated.index.to_numpy()
    treated_ps = ps[treated_positions]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(control_ps.reshape(-1, 1))
    _, control_matches = nbrs.kneighbors(treated_ps.reshape(-1, 1))

    matched_control_positions = np.unique(control_positions[control_matches.flatten()])
    matched = pd.concat([treated, df.loc[matched_control_positions]], axis=0).drop_duplicates()
    matched = matched.drop(columns=['index']) if 'index' in matched.columns else matched
    return matched.sort_values(['subject', 'time_bin'])


def panel_regression(merged_df, outcome, treatment='stress_mean_lag1'):
    df = merged_df.copy()
    df['subject'] = df['subject'].astype(str)
    df[outcome] = pd.to_numeric(df[outcome], errors='coerce')
    df[treatment] = pd.to_numeric(df[treatment], errors='coerce')

    candidate_terms = [treatment]
    for col in ['stress_std', 'stress_max', 'high_pressure_share']:
        if col in df.columns:
            candidate_terms.append(col)

    needed = [outcome, treatment, 'subject'] + [col for col in candidate_terms if col in df.columns]
    df = df.dropna(subset=needed)
    if df.empty:
        raise ValueError('No complete rows available for panel regression.')

    rhs = ' + '.join(candidate_terms + ['C(subject)'])
    formula = f'{outcome} ~ {rhs}'
    return smf.ols(formula=formula, data=df).fit(cov_type='HC3')


def summarize_policy_relevance(df, outcome):
    if outcome not in df.columns or 'stress_mean' not in df.columns:
        return {}

    valid = df[[outcome, 'stress_mean']].dropna()
    if len(valid) < 5:
        return {}

    return {
        'pearson_corr': float(valid[outcome].corr(valid['stress_mean'])),
        'mean_outcome': float(valid[outcome].mean()),
        'mean_stress': float(valid['stress_mean'].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='Estimate causal linkage between stress and operational outcomes.')
    parser.add_argument('--stress-agg', default='results/stress_kpi_merged.csv')
    parser.add_argument('--kpi-file', default=None)
    parser.add_argument('--outcome', required=True)
    parser.add_argument('--maxlag', type=int, default=3)
    parser.add_argument('--bin-size-sec', type=int, default=300)
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    merged = load_analysis_frame(args.stress_agg, outcome=args.outcome, kpi_file=args.kpi_file, bin_size_sec=args.bin_size_sec)
    lagged = add_lagged_features(merged, outcome=args.outcome, treatment='stress_mean', lag=1)

    granger_df = run_granger_by_subject(merged, feature='stress_mean', outcome=args.outcome, maxlag=args.maxlag)
    granger_path = os.path.join(args.output_dir, 'granger_causality_results.csv')
    granger_df.to_csv(granger_path, index=False)

    matched = propensity_score_match(merged)
    match_path = os.path.join(args.output_dir, 'propensity_matched_sample.csv')
    if matched is not None:
        matched.to_csv(match_path, index=False)

    reg = panel_regression(lagged, outcome=args.outcome, treatment='stress_mean_lag1')
    reg_path = os.path.join(args.output_dir, 'panel_regression_summary.txt')
    with open(reg_path, 'w', encoding='utf-8') as f:
        f.write(reg.summary().as_text())

    summary = summarize_policy_relevance(merged, outcome=args.outcome)
    summary_path = os.path.join(args.output_dir, 'causal_linkage_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Causal linkage summary\n')
        f.write('=====================\n')
        if summary:
            f.write(f"Pearson corr(stress_mean, {args.outcome}): {summary['pearson_corr']:.4f}\n")
            f.write(f"Mean stress_mean: {summary['mean_stress']:.4f}\n")
            f.write(f"Mean {args.outcome}: {summary['mean_outcome']:.4f}\n")
        if not granger_df.empty:
            f.write(f"Min Granger p-value: {granger_df['p_value'].min():.6f}\n")
        f.write('\nPanel regression summary:\n')
        f.write(reg.summary().as_text())

    print(f'Granger results saved to {granger_path}')
    if matched is not None:
        print(f'Propensity-matched sample saved to {match_path}')
    print(f'Panel regression saved to {reg_path}')
    print(f'Causal summary saved to {summary_path}')


if __name__ == '__main__':
    main()
