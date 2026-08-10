# Detecting Invisible Algorithmic Pressure

This project investigates the human cost of automated management systems. By leveraging unsupervised Deep Learning and physiological data, we aim to map an "invisible redline" of stress experienced by workers in modern, algorithmically driven environments (e.g., supply chains, gig economies).

The current pipeline now includes:
- multimodal physiological synchronization (EDA, BVP, TEMP, ACC magnitude, RESP),
- richer feature engineering (time-domain + frequency-domain HRV),
- leave-one-subject-out (LOSO) cross-validation,
- anomaly-model benchmarking (Autoencoder vs Isolation Forest, One-Class SVM, LOF),
- an operational linkage module for mapping stress indices to supply-chain KPI outcomes,
- a causal-analysis module with Granger tests and subject fixed effects,
- a policy simulator that turns stress-linked KPIs into proxy economic costs,
- an external-cohort builder that prefers the bundled public wearable exercise/stress corpus and falls back to a synthetic domain-shift proxy.

## Project Overview

The repository is organized following data science best practices:
- **`data/`**: Houses `raw/` datasets (e.g., WESAD, Empatica E4) and `processed/` datasets ready for ML.
- **`src/`**: Core source code containing data pipelines and model architectures.
- **`models/`**: Saved PyTorch weights (e.g., `autoencoder.pth`).
- **`results/`** & **`notebooks/`**: Exploratory analysis and generated evaluation metrics (ROC curves, error distributions).

## Deployment & Execution

### 1. Environment Setup
Create and activate a virtual environment, then install the dependencies:
```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Unix/MacOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Pipeline
Execute the data engineering and model training steps in the following order:

```bash
# 1. Process and synchronize raw multi-frequency biosignals
python src/data/preprocess.py

# 2. Extract higher-level multimodal features (e.g., HRV LF/HF, EDA peaks, ACC/TEMP/RESP stats)
python src/data/feature_extraction.py

# 3. Train and evaluate using LOSO CV + baseline model comparison
python src/models/train.py
```

### 3. Outputs
After training, the main outputs are written under `results/`:
- `loso_fold_metrics.csv`: Per-subject fold metrics and detection rates.
- `model_comparison.csv`: Mean/stdev AUC across evaluated models.
- `cv_window_scores.csv`: Out-of-fold anomaly scores per window.
- `error_distribution.png`, `roc_curve.png`, `model_auc_comparison.png`: Core evaluation plots.
- `evaluation_summary.txt`: Compact run summary.

### 4. Optional: Link Stress to Supply-Chain Outcomes
If you have an operational KPI file, run the linkage module directly. Alternatively, use our built-in simulation:

```bash
# Generate synthetic KPI data tied to stress scores
python src/data/simulate_kpi.py

# Run the outcome linkage evaluation
python src/data/outcome_linkage.py \
	--kpi-file data/raw/synthetic_kpi.csv \
	--outcomes defect_rate delay_flag cycle_time
```

### 5. Cross-Dataset Robustness Evaluation
To build and evaluate an external cohort, the script first prefers the public wearable exercise/stress corpus bundled in the repo and otherwise falls back to a synthetic domain-shift proxy:

```bash
# Generate the external validation cohort (real public wearable data first, synthetic proxy fallback)
python src/data/simulate_external_dataset.py

# Evaluate cross-dataset robustness (Isolation Forest & Autoencoder)
python src/models/cross_dataset_eval.py
```

### 6. Causal Analysis and Policy Simulation

Use the causal analysis utilities to estimate causal linkage and panel regressions linking stress aggregates to KPIs:

```bash
python src/analysis/causal_linkage.py --outcome defect_rate
```

Simulate operational policies and estimate expected economic costs using the linkage between stress and KPIs:

```bash
python src/decision/policy_simulator.py --kpi-file data/raw/synthetic_kpi.csv --stress-agg results/stress_kpi_merged.csv
```

See `docs/theory.md` for the theoretical framing and assumptions used to move from correlation to causal interpretation.

## Results

Our analysis leverages multimodal physiological data and unsupervised anomaly detection models to identify moments of acute stress ("invisible algorithmic pressure"). 

### ROC Curve and Anomaly Detection
The ROC curve evaluates the model's performance in detecting stress anomalies compared to baseline operational conditions.

![ROC Curve](docs/images/roc_curve.png)

### Model Comparison
We benchmarked several models (Autoencoder, Isolation Forest, One-Class SVM, LOF). The Autoencoder and Isolation Forest architectures generally yield the highest AUC scores across our leave-one-subject-out (LOSO) cross-validation framework.

![Model AUC Comparison](docs/images/model_auc_comparison.png)

### Error Distribution
The error distribution plot highlights the separation between standard operational states and critical stress events, enabling the setting of a viable "invisible redline" threshold.

![Error Distribution](docs/images/error_distribution.png)
