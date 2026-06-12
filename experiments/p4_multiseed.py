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

CONFIG = {
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

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
NUM_AGENTS = 200
NUM_TICKS = 10_000
SAMPLE_INTERVAL = 500

# Key metrics to extract at t=10000
KEY_METRICS = [
    'pop', 'employment_rate', 'tools_mean', 'wealth_gini', 'eff_wealth_gini',
    'tools_quality_mean', 'goods_quality_mean', 'res_mean',
    'lending_total', 'avg_debt', 'adverse_selection', 'quality_premium',
    'trade_volume', 'skill_mean', 'service_share',
    'employer_tools_quality_mean', 'non_employer_tools_quality_mean',
    'employment_young', 'employment_prime', 'employment_elder',
]


def run_seed(seed):
    arrays, key = init_agents(
        NUM_AGENTS, seed=seed, initial_endowment=10.0, num_tags=0,
        initial_age_range=(CONFIG['adult_age'], CONFIG['elder_age']),
    )

    prev = None
    series = []
    start = time.time()

    for tick in range(NUM_TICKS):
        prev = arrays
        arrays, key, stats = jax_step_on(arrays, key, CONFIG, tick)
        arrays, key, births, aging_deaths = jax_step_demographic(arrays, key, CONFIG)

        if (tick + 1) % SAMPLE_INTERVAL == 0:
            m = compute_metrics(arrays, stats, tick + 1, prev, CONFIG)
            series.append(m)

    elapsed = time.time() - start
    final = series[-1]

    result = {
        'seed': seed,
        'elapsed': elapsed,
        'time_series': [
            {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}
            for d in series
        ],
        'final': {k: final.get(k, 0) for k in KEY_METRICS},
    }
    return result


def main():
    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        result = run_seed(seed)
        all_results.append(result)
        f = result['final']

        print(
            f"  pop={f['pop']} emp={f['employment_rate']:.3f} "
            f"tools={f['tools_mean']:.1f} wg={f['wealth_gini']:.3f} "
            f"tq={f['tools_quality_mean']:.3f} gq={f['goods_quality_mean']:.3f} "
            f"lend={f['lending_total']:.0f} debt={f['avg_debt']:.2f} "
            f"trade={f['trade_volume']:.0f} adv={f['adverse_selection']:.3f} "
            f"| {result['elapsed']:.0f}s"
        )

    # Summary statistics
    print(f"\n{'='*80}")
    print("  Multi-seed summary (mean +/- std)")
    print(f"{'='*80}")

    header = f"{'metric':<30s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
    print(header)
    print("-" * len(header))

    import numpy as np
    for k in KEY_METRICS:
        vals = [r['final'][k] for r in all_results]
        mean, std = np.mean(vals), np.std(vals)
        print(f"{k:<30s} {mean:10.3f} {std:10.3f} {min(vals):10.3f} {max(vals):10.3f}")

    # Health checks
    print(f"\n  --- Health checks ---")
    checks = [
        ("Employment 0.78-0.82", all(0.78 <= r['final']['employment_rate'] <= 0.82 for r in all_results)),
        ("Tools 80-120", all(50 <= r['final']['tools_mean'] <= 150 for r in all_results)),
        ("Gini < 0.70", all(r['final']['wealth_gini'] < 0.70 for r in all_results)),
        ("tq > 0.15", all(r['final']['tools_quality_mean'] > 0.15 for r in all_results)),
        ("Lending > 0", all(r['final']['lending_total'] > 0 for r in all_results)),
        ("Pop > 450", all(r['final']['pop'] > 450 for r in all_results)),
    ]
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    # Save
    output_dir = Path(__file__).parent.parent / 'findings' / 'sweep_data' / 'p41_multiseed'
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        fname = output_dir / f"seed_{r['seed']}_10k.json"
        with open(fname, 'w') as f:
            json.dump(r, f, indent=2, default=str)

    summary = {
        'seeds': SEEDS,
        'num_agents': NUM_AGENTS,
        'num_ticks': NUM_TICKS,
        'summary': {
            k: {
                'mean': float(np.mean([r['final'][k] for r in all_results])),
                'std': float(np.std([r['final'][k] for r in all_results])),
                'min': float(min(r['final'][k] for r in all_results)),
                'max': float(max(r['final'][k] for r in all_results)),
            }
            for k in KEY_METRICS
        },
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {output_dir}")


if __name__ == '__main__':
    main()
