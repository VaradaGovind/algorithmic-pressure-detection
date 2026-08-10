import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

from autoencoder import ImprovedPhysiologicalAutoencoder

PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
MODEL_SAVE_PATH = 'models/autoencoder_improved.pth'
SCALER_ARTIFACT_PATH = 'models/autoencoder_artifacts_improved.npz'

RANDOM_SEED = 42
BATCH_SIZE = 32  # Reduced batch size for better gradients
AUTOENCODER_EPOCHS_CV = 80  # Increased epochs
AUTOENCODER_EPOCHS_FINAL = 100
LEARNING_RATE = 1e-3  # Standard learning rate
WEIGHT_DECAY = 5e-5  # Light L2 regularization
META_COLUMNS = {'subject', 'window_start_sec'}


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def roc_auc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def load_feature_data():
    print("Loading engineered features...")
    baseline_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_baseline.csv'))
    stress_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_stress.csv'))

    common_columns = sorted(set(baseline_df.columns).intersection(stress_df.columns))
    if 'subject' not in common_columns:
        raise ValueError("Feature files must contain a 'subject' column for LOSO cross-validation.")

    feature_columns = []
    for col in common_columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(baseline_df[col]) and pd.api.types.is_numeric_dtype(stress_df[col]):
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError("No shared numeric feature columns were found in the engineered datasets.")

    keep_columns = ['subject'] + [c for c in ['window_start_sec'] if c in common_columns] + feature_columns
    baseline_df = baseline_df[keep_columns].replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    stress_df = stress_df[keep_columns].replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)

    baseline_df['subject'] = baseline_df['subject'].astype(str)
    stress_df['subject'] = stress_df['subject'].astype(str)

    print(f"Using {len(feature_columns)} features for training/evaluation.")
    return baseline_df, stress_df, feature_columns


def train_autoencoder_fast(train_data, input_dim, epochs=AUTOENCODER_EPOCHS_CV, lr=LEARNING_RATE):
    """Fast training without validation loop for speed."""
    model = ImprovedPhysiologicalAutoencoder(input_dim, latent_dim=16, dropout_rate=0.25)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    train_tensor = torch.FloatTensor(train_data)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x in train_loader:
            x = batch_x[0]
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

    return model, train_losses


def get_reconstruction_scores(model, data_array):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data_array)
        recon = model(data_tensor)
        return model.get_reconstruction_error(data_tensor, recon).cpu().numpy()


def compute_detection_rates(train_baseline_scores, test_stress_scores):
    mean_train = float(np.mean(train_baseline_scores))
    std_train = float(np.std(train_baseline_scores))

    threshold_3std = mean_train + 3 * std_train
    threshold_2std = mean_train + 2 * std_train
    threshold_95th = float(np.percentile(train_baseline_scores, 95))

    return {
        'threshold_3std': threshold_3std,
        'threshold_2std': threshold_2std,
        'threshold_95th': threshold_95th,
        'stress_detect_rate_3std_pct': float(np.mean(test_stress_scores > threshold_3std) * 100),
        'stress_detect_rate_2std_pct': float(np.mean(test_stress_scores > threshold_2std) * 100),
        'stress_detect_rate_95th_pct': float(np.mean(test_stress_scores > threshold_95th) * 100),
    }


def evaluate_baseline_models(X_train_baseline, X_test_baseline, X_test_stress):
    model_aucs = {
        'isolation_forest_auc': np.nan,
        'one_class_svm_auc': np.nan,
        'lof_auc': np.nan,
    }

    X_eval = np.vstack([X_test_baseline, X_test_stress])
    y_true = np.concatenate([np.zeros(len(X_test_baseline)), np.ones(len(X_test_stress))])

    try:
        iso = IsolationForest(n_estimators=400, contamination='auto', random_state=RANDOM_SEED, n_jobs=-1)
        iso.fit(X_train_baseline)
        iso_scores = -iso.score_samples(X_eval)
        model_aucs['isolation_forest_auc'] = roc_auc_safe(y_true, iso_scores)
    except Exception as exc:
        print(f"Warning: Isolation Forest evaluation failed: {exc}")

    try:
        ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.08)
        ocsvm.fit(X_train_baseline)
        ocsvm_scores = -ocsvm.decision_function(X_eval).reshape(-1)
        model_aucs['one_class_svm_auc'] = roc_auc_safe(y_true, ocsvm_scores)
    except Exception as exc:
        print(f"Warning: One-Class SVM evaluation failed: {exc}")

    try:
        if len(X_train_baseline) > 6:
            n_neighbors = max(5, min(35, len(X_train_baseline) - 1))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1)
            lof.fit(X_train_baseline)
            lof_scores = -lof.score_samples(X_eval)
            model_aucs['lof_auc'] = roc_auc_safe(y_true, lof_scores)
    except Exception as exc:
        print(f"Warning: LOF evaluation failed: {exc}")

    return model_aucs


def evaluate_fold(subject, baseline_df, stress_df, feature_columns):
    train_baseline_df = baseline_df[baseline_df['subject'] != subject]
    test_baseline_df = baseline_df[baseline_df['subject'] == subject]
    test_stress_df = stress_df[stress_df['subject'] == subject]

    if train_baseline_df.empty or test_baseline_df.empty or test_stress_df.empty:
        return None

    scaler = MinMaxScaler()
    X_train_baseline = scaler.fit_transform(train_baseline_df[feature_columns])
    X_test_baseline = scaler.transform(test_baseline_df[feature_columns])
    X_test_stress = scaler.transform(test_stress_df[feature_columns])

    model, _ = train_autoencoder_fast(
        X_train_baseline,
        input_dim=len(feature_columns),
        epochs=AUTOENCODER_EPOCHS_CV,
        lr=LEARNING_RATE,
    )

    train_baseline_scores = get_reconstruction_scores(model, X_train_baseline)
    test_baseline_scores = get_reconstruction_scores(model, X_test_baseline)
    test_stress_scores = get_reconstruction_scores(model, X_test_stress)

    y_true = np.concatenate([np.zeros(len(test_baseline_scores)), np.ones(len(test_stress_scores))])
    ae_scores = np.concatenate([test_baseline_scores, test_stress_scores])
    autoencoder_auc = roc_auc_safe(y_true, ae_scores)

    fold_metrics = {
        'subject': subject,
        'n_train_baseline': len(train_baseline_df),
        'n_test_baseline': len(test_baseline_df),
        'n_test_stress': len(test_stress_df),
        'autoencoder_auc': autoencoder_auc,
        'baseline_mean_error': float(np.mean(test_baseline_scores)),
        'stress_mean_error': float(np.mean(test_stress_scores)),
    }
    fold_metrics.update(compute_detection_rates(train_baseline_scores, test_stress_scores))
    fold_metrics.update(evaluate_baseline_models(X_train_baseline, X_test_baseline, X_test_stress))

    baseline_window_scores = pd.DataFrame({
        'subject': subject,
        'label': 0,
        'anomaly_score': test_baseline_scores,
        'model': 'autoencoder_improved',
    })
    stress_window_scores = pd.DataFrame({
        'subject': subject,
        'label': 1,
        'anomaly_score': test_stress_scores,
        'model': 'autoencoder_improved',
    })

    if 'window_start_sec' in test_baseline_df.columns:
        baseline_window_scores['window_start_sec'] = test_baseline_df['window_start_sec'].to_numpy()
    if 'window_start_sec' in test_stress_df.columns:
        stress_window_scores['window_start_sec'] = test_stress_df['window_start_sec'].to_numpy()

    window_scores = pd.concat([baseline_window_scores, stress_window_scores], ignore_index=True)

    return {
        'metrics': fold_metrics,
        'window_scores': window_scores,
        'baseline_scores': test_baseline_scores,
        'stress_scores': test_stress_scores,
    }


def summarize_model_comparison(fold_metrics_df):
    metric_columns = {
        'Autoencoder (Improved)': 'autoencoder_auc',
        'Isolation Forest': 'isolation_forest_auc',
        'One-Class SVM': 'one_class_svm_auc',
        'LOF': 'lof_auc',
    }

    rows = []
    for model_name, col in metric_columns.items():
        valid_scores = fold_metrics_df[col].dropna()
        rows.append({
            'model': model_name,
            'mean_auc': float(valid_scores.mean()) if not valid_scores.empty else np.nan,
            'std_auc': float(valid_scores.std(ddof=0)) if not valid_scores.empty else np.nan,
            'fold_count': int(valid_scores.shape[0]),
        })
    return pd.DataFrame(rows)


def save_error_distribution_and_roc(baseline_scores, stress_scores):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(baseline_scores, bins=60, alpha=0.65, label='Baseline (Out-of-fold)', density=True, color='#2ecc71')
    plt.hist(stress_scores, bins=60, alpha=0.65, label='Stress (Out-of-fold)', density=True, color='#e74c3c')
    plt.title('Reconstruction Error Distribution (LOSO) - Improved', fontsize=14)
    plt.xlabel('Mean Squared Error (Anomaly Score)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    dist_path = os.path.join(RESULTS_DIR, 'error_distribution_improved.png')
    plt.savefig(dist_path)
    plt.close()

    y_true = np.concatenate([np.zeros(len(baseline_scores)), np.ones(len(stress_scores))])
    y_scores = np.concatenate([baseline_scores, stress_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'LOSO ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Anomaly Detection (LOSO) - Improved', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    roc_path = os.path.join(RESULTS_DIR, 'roc_curve_improved.png')
    plt.savefig(roc_path)
    plt.close()

    return roc_auc, dist_path, roc_path


def save_model_comparison_plot(model_comparison_df):
    plt.figure(figsize=(9, 5))
    x = np.arange(len(model_comparison_df))
    means = model_comparison_df['mean_auc'].to_numpy()
    stds = model_comparison_df['std_auc'].fillna(0).to_numpy()

    plt.bar(x, means, yerr=stds, capsize=5, color=['#2c3e50', '#16a085', '#d35400', '#8e44ad'])
    plt.xticks(x, model_comparison_df['model'], rotation=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel('AUC')
    plt.title('Model Comparison under LOSO Cross-Validation - Improved')
    plt.grid(axis='y', alpha=0.3)
    out_path = os.path.join(RESULTS_DIR, 'model_auc_comparison_improved.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def train_and_save_final_model(baseline_df, feature_columns):
    scaler = MinMaxScaler()
    X_baseline = scaler.fit_transform(baseline_df[feature_columns])

    final_model, losses = train_autoencoder_fast(
        X_baseline,
        input_dim=len(feature_columns),
        epochs=AUTOENCODER_EPOCHS_FINAL,
        lr=LEARNING_RATE,
    )

    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    np.savez(
        SCALER_ARTIFACT_PATH,
        scaler_min=scaler.min_,
        scaler_scale=scaler.scale_,
        feature_columns=np.array(feature_columns, dtype=object),
    )
    return losses


def main():
    set_seed(RANDOM_SEED)
    os.makedirs('models', exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        baseline_df, stress_df, feature_columns = load_feature_data()

        subjects = sorted(set(baseline_df['subject']).intersection(stress_df['subject']))
        if not subjects:
            raise ValueError("No common subjects found between baseline and stress feature files.")

        fold_rows = []
        window_score_rows = []
        pooled_baseline_scores = []
        pooled_stress_scores = []

        print(f"Running LOSO cross-validation across {len(subjects)} subjects (IMPROVED)...")
        for idx, subject in enumerate(subjects, start=1):
            fold_result = evaluate_fold(subject, baseline_df, stress_df, feature_columns)
            if fold_result is None:
                print(f"[{idx}/{len(subjects)}] {subject}: skipped (insufficient data).")
                continue

            metrics = fold_result['metrics']
            fold_rows.append(metrics)
            window_score_rows.append(fold_result['window_scores'])
            pooled_baseline_scores.append(fold_result['baseline_scores'])
            pooled_stress_scores.append(fold_result['stress_scores'])

            print(
                f"[{idx}/{len(subjects)}] {subject}: "
                f"AE AUC={metrics['autoencoder_auc']:.3f}, "
                f"IF={metrics['isolation_forest_auc']:.3f}, "
                f"OCSVM={metrics['one_class_svm_auc']:.3f}, "
                f"LOF={metrics['lof_auc']:.3f}"
            )

        if not fold_rows:
            raise ValueError("No valid LOSO folds were evaluated.")

        fold_metrics_df = pd.DataFrame(fold_rows)
        fold_metrics_path = os.path.join(RESULTS_DIR, 'loso_fold_metrics_improved.csv')
        fold_metrics_df.to_csv(fold_metrics_path, index=False)

        window_scores_df = pd.concat(window_score_rows, ignore_index=True)
        window_scores_path = os.path.join(RESULTS_DIR, 'cv_window_scores_improved.csv')
        window_scores_df.to_csv(window_scores_path, index=False)

        model_comparison_df = summarize_model_comparison(fold_metrics_df)
        model_comparison_path = os.path.join(RESULTS_DIR, 'model_comparison_improved.csv')
        model_comparison_df.to_csv(model_comparison_path, index=False)

        pooled_baseline_scores = np.concatenate(pooled_baseline_scores)
        pooled_stress_scores = np.concatenate(pooled_stress_scores)
        pooled_auc, dist_path, roc_path = save_error_distribution_and_roc(
            pooled_baseline_scores,
            pooled_stress_scores,
        )
        comp_plot_path = save_model_comparison_plot(model_comparison_df)

        final_losses = train_and_save_final_model(baseline_df, feature_columns)

        summary_lines = [
            f"=== IMPROVED TRAINING SUMMARY ===",
            f"Subjects evaluated: {len(fold_metrics_df)}",
            f"Features used: {len(feature_columns)}",
            f"LOSO pooled autoencoder AUC (Improved): {pooled_auc:.4f}",
            f"Mean autoencoder fold AUC (Improved): {fold_metrics_df['autoencoder_auc'].mean():.4f}",
            f"Mean Isolation Forest AUC: {fold_metrics_df['isolation_forest_auc'].mean():.4f}",
            f"Mean One-Class SVM AUC: {fold_metrics_df['one_class_svm_auc'].mean():.4f}",
            f"Mean LOF AUC: {fold_metrics_df['lof_auc'].mean():.4f}",
            f"Mean stress detect rate (95th threshold): {fold_metrics_df['stress_detect_rate_95th_pct'].mean():.2f}%",
            f"Final model save path: {MODEL_SAVE_PATH}",
            f"Scaler artifact path: {SCALER_ARTIFACT_PATH}",
            f"Final training epochs: {len(final_losses)}",
        ]
        summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary_improved.txt')
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write('\n'.join(summary_lines))

        print("\nImproved evaluation complete.")
        print(f"- Fold metrics: {fold_metrics_path}")
        print(f"- Model comparison: {model_comparison_path}")
        print(f"- CV window scores: {window_scores_path}")
        print(f"- Error distribution plot: {dist_path}")
        print(f"- ROC curve plot: {roc_path}")
        print(f"- Model comparison plot: {comp_plot_path}")
        print(f"- Summary report: {summary_path}")

    except FileNotFoundError:
        print("Processed data not found. Please ensure the preprocessing script has finished running.")
    except ValueError as exc:
        print(f"Configuration error: {exc}")

if __name__ == '__main__':
    main()
