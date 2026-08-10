import os
import subprocess
import sys

PYTHON = sys.executable

STEPS = [
    ('Preprocess raw signals', [PYTHON, 'src/data/preprocess.py']),
    ('Extract features', [PYTHON, 'src/data/feature_extraction.py']),
    ('Train models and LOSO evaluation', [PYTHON, 'src/models/train.py']),
    ('Build external cohort', [PYTHON, 'src/data/simulate_external_dataset.py']),
    ('Cross-dataset evaluation', [PYTHON, 'src/models/cross_dataset_eval.py']),
    ('Generate synthetic KPI', [PYTHON, 'src/data/simulate_kpi.py']),
    ('Outcome linkage', [PYTHON, 'src/data/outcome_linkage.py', '--kpi-file', 'data/raw/synthetic_kpi.csv', '--outcomes', 'defect_rate', 'delay_flag', 'cycle_time']),
    ('Panel regression and causal checks', [PYTHON, 'src/analysis/causal_linkage.py', '--stress-agg', 'results/stress_kpi_merged.csv', '--kpi-file', 'data/raw/synthetic_kpi.csv', '--outcome', 'defect_rate']),
    ('Policy simulation', [PYTHON, 'src/decision/policy_simulator.py', '--kpi-file', 'data/raw/synthetic_kpi.csv']),
]


def run_step(name, cmd):
    command_text = ' '.join(cmd)
    print(f'==> {name}: {command_text}')
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f'Step failed: {name} (exit {exc.returncode}). Continuing with remaining steps.')


def main():
    for name, cmd in STEPS:
        run_step(name, cmd)


if __name__ == '__main__':
    main()

