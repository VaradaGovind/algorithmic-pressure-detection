import os
import argparse
import pandas as pd
import numpy as np


DEFAULT_COST_PARAMS = {
    'cost_per_defect': 100.0,
    'cost_per_delay': 50.0,
    'cost_per_cycle_sec': 0.5,
    'cost_per_stress_unit': 25.0,
    'rotation_overhead': 250.0,
    'workforce_overhead': 1000.0,
}


def _economic_cost(df, cost_params):
    return (
        df['defect_rate'] * cost_params['cost_per_defect']
        + df['delay_flag'] * cost_params['cost_per_delay']
        + df['cycle_time'] * cost_params['cost_per_cycle_sec']
        + df.get('stress_mean', pd.Series(0.0, index=df.index)) * cost_params['cost_per_stress_unit']
    ).sum()


def simulate_policies(kpi_df, stress_agg, cost_params=None):
    if cost_params is None:
        cost_params = DEFAULT_COST_PARAMS

    required_cols = {'defect_rate', 'delay_flag', 'cycle_time'}
    if required_cols.issubset(stress_agg.columns):
        merged = stress_agg.copy()
    else:
        merged = stress_agg.merge(kpi_df, on=['subject', 'time_bin'], how='inner')
    if merged.empty:
        raise ValueError('No overlap between stress and KPI bins for policy simulation.')

    outcomes = []

    base_cost = _economic_cost(merged, cost_params)
    outcomes.append(
        {
            'policy': 'baseline',
            'expected_cost': base_cost,
            'savings_vs_baseline': 0.0,
            'savings_pct': 0.0,
            'mean_defect_rate': float(merged['defect_rate'].mean()),
            'mean_delay_flag': float(merged['delay_flag'].mean()),
            'mean_cycle_time': float(merged['cycle_time'].mean()),
            'mean_stress_mean': float(merged.get('stress_mean', pd.Series(0.0, index=merged.index)).mean()),
        }
    )

    median_stress = merged['stress_mean'].median()
    m = merged.copy()
    high_stress = m['stress_mean'] > median_stress
    m['defect_rate'] = np.where(high_stress, m['defect_rate'] * 0.9, m['defect_rate'])
    m['delay_flag'] = np.where(high_stress, m['delay_flag'] * 0.95, m['delay_flag'])
    m['cycle_time'] = np.where(high_stress, m['cycle_time'] * 0.98, m['cycle_time'])
    buffer_cost = _economic_cost(m, cost_params) + cost_params['rotation_overhead']
    outcomes.append(
        {
            'policy': 'buffer_stock',
            'expected_cost': buffer_cost,
            'savings_vs_baseline': base_cost - buffer_cost,
            'savings_pct': float((base_cost - buffer_cost) / base_cost * 100.0) if base_cost else 0.0,
            'mean_defect_rate': float(m['defect_rate'].mean()),
            'mean_delay_flag': float(m['delay_flag'].mean()),
            'mean_cycle_time': float(m['cycle_time'].mean()),
            'mean_stress_mean': float(m['stress_mean'].mean()),
        }
    )

    flex = merged.copy()
    flex['cycle_time'] = flex['cycle_time'] * 0.92
    flex['defect_rate'] = flex['defect_rate'] * 0.96
    flex['stress_mean'] = flex.get('stress_mean', pd.Series(0.0, index=flex.index)) * 0.94
    flex_cost = _economic_cost(flex, cost_params) + cost_params['workforce_overhead']
    outcomes.append(
        {
            'policy': 'flexible_workforce',
            'expected_cost': flex_cost,
            'savings_vs_baseline': base_cost - flex_cost,
            'savings_pct': float((base_cost - flex_cost) / base_cost * 100.0) if base_cost else 0.0,
            'mean_defect_rate': float(flex['defect_rate'].mean()),
            'mean_delay_flag': float(flex['delay_flag'].mean()),
            'mean_cycle_time': float(flex['cycle_time'].mean()),
            'mean_stress_mean': float(flex['stress_mean'].mean()),
        }
    )

    rotation = merged.copy()
    rotation['stress_mean'] = rotation.get('stress_mean', pd.Series(0.0, index=rotation.index)) * 0.88
    rotation['delay_flag'] = rotation['delay_flag'] * 0.96
    rotation['cycle_time'] = rotation['cycle_time'] * 0.97
    rotation_cost = _economic_cost(rotation, cost_params) + cost_params['rotation_overhead']
    outcomes.append(
        {
            'policy': 'stress_rotation',
            'expected_cost': rotation_cost,
            'savings_vs_baseline': base_cost - rotation_cost,
            'savings_pct': float((base_cost - rotation_cost) / base_cost * 100.0) if base_cost else 0.0,
            'mean_defect_rate': float(rotation['defect_rate'].mean()),
            'mean_delay_flag': float(rotation['delay_flag'].mean()),
            'mean_cycle_time': float(rotation['cycle_time'].mean()),
            'mean_stress_mean': float(rotation['stress_mean'].mean()),
        }
    )

    return pd.DataFrame(outcomes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kpi-file', required=True)
    parser.add_argument('--stress-agg', default='results/stress_kpi_merged.csv')
    parser.add_argument('--output', default='results/policy_simulation.csv')
    parser.add_argument('--cost-per-defect', type=float, default=DEFAULT_COST_PARAMS['cost_per_defect'])
    parser.add_argument('--cost-per-delay', type=float, default=DEFAULT_COST_PARAMS['cost_per_delay'])
    parser.add_argument('--cost-per-cycle-sec', type=float, default=DEFAULT_COST_PARAMS['cost_per_cycle_sec'])
    parser.add_argument('--cost-per-stress-unit', type=float, default=DEFAULT_COST_PARAMS['cost_per_stress_unit'])
    args = parser.parse_args()

    stress = pd.read_csv(args.stress_agg)
    kpi = pd.read_csv(args.kpi_file)
    kpi = kpi.copy()
    if 'time_bin' not in kpi.columns:
        raise ValueError('KPI file must include time_bin column for simulation.')

    cost_params = {
        'cost_per_defect': args.cost_per_defect,
        'cost_per_delay': args.cost_per_delay,
        'cost_per_cycle_sec': args.cost_per_cycle_sec,
        'cost_per_stress_unit': args.cost_per_stress_unit,
        'rotation_overhead': DEFAULT_COST_PARAMS['rotation_overhead'],
        'workforce_overhead': DEFAULT_COST_PARAMS['workforce_overhead'],
    }

    sim = simulate_policies(kpi, stress, cost_params=cost_params)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sim.to_csv(args.output, index=False)
    print(f'Policy simulation saved to {args.output}')


if __name__ == '__main__':
    main()
