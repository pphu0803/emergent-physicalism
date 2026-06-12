"""
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import time
import json
from pathlib import Path

from core.state import init_agents
from core.jax_steps import jax_step_on, jax_step_demographic
from analysis.runtime_metrics import compute_metrics

BASE_CONFIG = {
    'subsistence_cost': 0.9,
    'natural_inflow_per_capita': 0.72,
    'resource_ceiling': 5000.0,
    'depreciation_rate': 0.02,
    'production_scale': 3.0,
    'tool_saturation_K': 10.0,
    'population_ceiling': 500,
    'interest_rate': 0.001,
    'debt_service_ratio': 0.05,
    'lending_surplus_rate': 0.03,
    'goods_q_decay': 0.0005,
    'tools_q_decay': 0.0005,
    'aging_enabled': True,
    'aging_base_rate': 0.0001,
    'aging_rate_multiplier': 0.0005,
    'adult_age': 936,
    'elder_age': 3120,
    'reproduction_base_prob': 0.01,
    'reproduction_cooldown': 20,
    'inheritance_ratio': 0.8,
    'mutation_strength': 0.15,
    'demographic_enabled': True,
    'coercion_enabled': False,
    'employment_enabled': True,
    'employer_ratio': 0.20,
    'lending_enabled': True,
    'num_tags': 0,
}

RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
SEED = 42
NUM_TICKS = 10_000
SAMPLE_INTERVAL = 500

KEY_METRICS = [
    'pop', 'employment_rate', 'tools_mean', 'wealth_gini',
    'tools_quality_mean', 'goods_quality_mean',
    'lending_total', 'trade_volume',
    'adverse_selection', 'quality_premium',
    'employer_tools_quality_mean', 'non_employer_tools_quality_mean',
    'avg_debt',
]


def run_single(ratio):
    cfg = {**BASE_CONFIG, 'employer_ratio': ratio}
    arrays, key = init_agents(
        200, seed=SEED, initial_endowment=10.0, num_tags=0,
        initial_age_range=(cfg['adult_age'], cfg['elder_age']),
    )
    prev = None
    series = []
    start = time.time()

    for tick in range(NUM_TICKS):
        prev = arrays
        arrays, key, stats = jax_step_on(arrays, key, cfg, tick)
        arrays, key, births, aging_deaths = jax_step_demographic(arrays, key, cfg)

        if (tick + 1) % SAMPLE_INTERVAL == 0:
            m = compute_metrics(arrays, stats, tick + 1, prev, cfg)
            series.append(m)

    elapsed = time.time() - start
    final = series[-1]
    return {
        'ratio': ratio,
        'elapsed': elapsed,
        'time_series': [
            {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}
            for d in series
        ],
        'final': {k: final.get(k, 0) for k in KEY_METRICS},
    }


def main():
    import numpy as np

    all_results = []
    for ratio in RATIOS:
        print(f"\n{'='*60}")
        print(f"  employer_ratio = {ratio}")
        print(f"{'='*60}")

        result = run_single(ratio)
        all_results.append(result)
        f = result['final']

        emp_tq = f.get('employer_tools_quality_mean', 0)
        nonemp_tq = f.get('non_employer_tools_quality_mean', 0.001)
        matthew = emp_tq / nonemp_tq if nonemp_tq > 0.001 else 0

        print(
            f"  pop={f['pop']} emp={f['employment_rate']:.3f} "
            f"tools={f['tools_mean']:.1f} gini={f['wealth_gini']:.3f} "
            f"tq={f['tools_quality_mean']:.3f} gq={f['goods_quality_mean']:.3f} "
            f"lend={f['lending_total']:.0f} trade={f['trade_volume']:.0f} "
            f"matthew={matthew:.2f}x adv={f['adverse_selection']:.3f} "
            f"| {result['elapsed']:.0f}s"
        )

    # Summary
    print(f"\n{'='*80}")
    print("  Employer threshold sensitivity summary")
    print(f"{'='*80}")

    header = f"{'ratio':>8s}"
    for k in KEY_METRICS:
        header += f" {k[:10]:>11s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        f = r['final']
        row = f"{r['ratio']:8.2f}"
        for k in KEY_METRICS:
            row += f" {f[k]:11.3f}"
        print(row)

    # Compute ranges
    print(f"\n  --- Phenomenon robustness check ---")
    for k in ['employment_rate', 'tools_quality_mean', 'goods_quality_mean',
              'adverse_selection', 'quality_premium']:
        vals = [r['final'][k] for r in all_results]
        mean = np.mean(vals)
        std = np.std(vals)
        cv = std / mean * 100 if mean != 0 else 0
        print(f"  {k}: mean={mean:.3f} std={std:.3f} CV={cv:.1f}%")

    # Save
    output_dir = Path(__file__).parent.parent / 'findings' / 'sweep_data' / 'p43_employer_threshold'
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        with open(output_dir / f"ratio_{r['ratio']}.json", 'w') as fout:
            json.dump(r, fout, indent=2, default=str)

    print(f"\nResults saved: {output_dir}")
    print(f"Total time: {sum(r['elapsed'] for r in all_results):.0f}s")


if __name__ == '__main__':
    main()
