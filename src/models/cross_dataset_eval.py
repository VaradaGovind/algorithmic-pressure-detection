import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc

from autoencoder import PhysiologicalAutoencoder


def align_features(frame, feature_columns):
    aligned = frame.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned = aligned[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return aligned

def roc_auc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)

def main():
    print("Loading external dataset for cross-dataset evaluation...")
    ext_base = pd.read_csv('data/processed/external_features_baseline.csv')
    ext_stress = pd.read_csv('data/processed/external_features_stress.csv')
    
    artifacts = np.load('models/autoencoder_artifacts.npz', allow_pickle=True)
    scaler_min = artifacts['scaler_min']
    scaler_scale = artifacts['scaler_scale']
    feature_columns = artifacts['feature_columns'].tolist()
    
    # Reconstruct the scaler
    scaler = MinMaxScaler()
    scaler.min_ = scaler_min
    scaler.scale_ = scaler_scale
    scaler.data_min_ = -scaler_min / scaler_scale # Reverse engineer if needed, though transform uses min_ and scale_ directly
    scaler.data_range_ = 1.0 / scaler_scale
    
    # Actually, sklearn MinMaxScaler transform only uses `min_` and `scale_`:
    # X_scaled = X_std * scale_ + min_ (no, it's X * scale_ + min_)
    # wait: X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    # In sklearn: X_scaled = X * scale_ + min_
    def manual_transform(X):
        return X * scaler_scale + scaler_min
    
    X_ext_base = manual_transform(align_features(ext_base, feature_columns).values)
    X_ext_stress = manual_transform(align_features(ext_stress, feature_columns).values)
    
    # 1. Evaluate Autoencoder
    input_dim = len(feature_columns)
    model = PhysiologicalAutoencoder(input_dim)
    model.load_state_dict(torch.load('models/autoencoder.pth'))
    model.eval()
    
    with torch.no_grad():
        t_base = torch.FloatTensor(X_ext_base)
        t_stress = torch.FloatTensor(X_ext_stress)
        
        recon_base = model(t_base)
        recon_stress = model(t_stress)
        
        scores_base = model.get_reconstruction_error(t_base, recon_base).cpu().numpy()
        scores_stress = model.get_reconstruction_error(t_stress, recon_stress).cpu().numpy()
        
    y_true_ae = np.concatenate([np.zeros(len(scores_base)), np.ones(len(scores_stress))])
    y_scores_ae = np.concatenate([scores_base, scores_stress])
    auc_ae = roc_auc_safe(y_true_ae, y_scores_ae)
    
    print(f"Autoencoder Cross-Dataset AUC: {auc_ae:.4f}")
    
    # 2. Evaluate Isolation Forest
    # We must fit it on the original baseline data to properly test its robustness to the new domain
    orig_base = pd.read_csv('data/processed/features_baseline.csv')
    X_orig_base = manual_transform(align_features(orig_base, feature_columns).values)
    
    iso = IsolationForest(n_estimators=300, contamination='auto', random_state=42)
    iso.fit(X_orig_base)
    
    X_eval = np.vstack([X_ext_base, X_ext_stress])
    iso_scores = -iso.score_samples(X_eval)
    
    auc_iso = roc_auc_safe(y_true_ae, iso_scores)
    print(f"Isolation Forest Cross-Dataset AUC: {auc_iso:.4f}")
    
    # Save the results
    os.makedirs('results', exist_ok=True)
    res_df = pd.DataFrame({
        'model': ['Autoencoder', 'Isolation Forest'],
        'cross_dataset_auc': [auc_ae, auc_iso]
    })
    res_path = 'results/cross_dataset_eval.csv'
    res_df.to_csv(res_path, index=False)
    print(f"Cross-dataset evaluation complete. Results saved to {res_path}")

if __name__ == '__main__':
    main()
