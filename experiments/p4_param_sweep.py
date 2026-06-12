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
    'lending_enabled': True,
    'num_tags': 0,
}

SWEEP_PARAMS = {
    'depreciation_rate':       [0.005, 0.01, 0.02, 0.04, 0.08],
    'natural_inflow_per_capita': [0.3, 0.5, 0.72, 1.0, 1.5],
    'tool_saturation_K':       [3.0, 5.0, 10.0, 20.0, 50.0],
    'production_scale':        [1.0, 2.0, 3.0, 5.0, 8.0],
    'subsistence_cost':        [0.5, 0.72, 0.9, 1.2, 1.8],
}

SEED = 42
NUM_AGENTS = 200
NUM_TICKS = 5_000   # Shorter for sweep speed
SAMPLE_INTERVAL = 500

KEY_METRICS = [
    'pop', 'employment_rate', 'tools_mean', 'wealth_gini',
    'tools_quality_mean', 'goods_quality_mean', 'res_mean',
    'lending_total', 'avg_debt', 'adverse_selection', 'quality_premium',
    'trade_volume', 'skill_mean', 'service_share',
]


def run_single(param_name, param_value):
    cfg = {**BASE_CONFIG, param_name: param_value}
    arrays, key = init_agents(
        NUM_AGENTS, seed=SEED, initial_endowment=10.0, num_tags=0,
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
        'param': param_name,
        'value': param_value,
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
    total = sum(len(v) for v in SWEEP_PARAMS.values())
    done = 0

    for param_name, values in SWEEP_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"  Sweep: {param_name}")
        print(f"{'='*60}")

        param_results = []
        for val in values:
            done += 1
            result = run_single(param_name, val)
            all_results.append(result)
            param_results.append(result)
            f = result['final']

            print(
                f"  [{done}/{total}] {param_name}={val:<8g} | "
                f"pop={f['pop']} emp={f['employment_rate']:.3f} "
                f"tools={f['tools_mean']:.1f} wg={f['wealth_gini']:.3f} "
                f"tq={f['tools_quality_mean']:.3f} gq={f['goods_quality_mean']:.3f} "
                f"lend={f['lending_total']:.0f} trade={f['trade_volume']:.0f} "
                f"| {result['elapsed']:.0f}s"
            )

        # Per-param summary
        print(f"\n  --- {param_name} summary ---")
        header = f"  {'value':>10s}"
        for k in KEY_METRICS[:8]:
            header += f" {k[:8]:>9s}"
        print(header)

        for r in param_results:
            row = f"  {r['value']:10g}"
            for k in KEY_METRICS[:8]:
                row += f" {r['final'][k]:9.3f}"
            print(row)

    # Save
    output_dir = Path(__file__).parent.parent / 'findings' / 'sweep_data' / 'p41_param_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        fname = output_dir / f"{r['param']}_{r['value']}.json"
        with open(fname, 'w') as f:
            json.dump(r, f, indent=2, default=str)

    # Save matrix
    matrix = {}
    for param_name, values in SWEEP_PARAMS.items():
        matrix[param_name] = {
            'values': values,
            'results': {
                str(v): next(
                    {k: r['final'][k] for k in KEY_METRICS}
                    for r in all_results
                    if r['param'] == param_name and r['value'] == v
                )
                for v in values
            },
        }

    with open(output_dir / 'sweep_matrix.json', 'w') as f:
        json.dump(matrix, f, indent=2)

    print(f"\nResults saved: {output_dir}")
    print(f"Total time: {sum(r['elapsed'] for r in all_results):.0f}s")


if __name__ == '__main__':
    main()
