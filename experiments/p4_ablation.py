"""

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
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
    # Ablation flags
    'quality_investment_enabled': True,
    'quality_in_pricing_enabled': True,
}

EXPERIMENTS = {
    'baseline': {},
    'A1_no_lending': {'lending_enabled': False},
    'A2_no_employment': {'employment_enabled': False},
    'A3_no_quality_investment': {'quality_investment_enabled': False},
    'A4_no_quality_pricing': {'quality_in_pricing_enabled': False},
}

SEED = 42
NUM_AGENTS = 200
NUM_TICKS = 10_000
SAMPLE_INTERVAL = 500

KEY_METRICS = [
    'pop', 'employment_rate', 'tools_mean', 'wealth_gini', 'eff_wealth_gini',
    'tools_quality_mean', 'goods_quality_mean', 'res_mean',
    'lending_total', 'avg_debt', 'adverse_selection', 'quality_premium',
    'trade_volume', 'skill_mean', 'service_share', 'quality_labor_share_mean',
    'employer_tools_quality_mean', 'non_employer_tools_quality_mean',
    'employer_tools_mean', 'non_employer_tools_mean',
]


def run_experiment(name, overrides):
    cfg = {**BASE_CONFIG, **overrides}
    arrays, key = init_agents(
        NUM_AGENTS, seed=SEED, initial_endowment=10.0, num_tags=0,
        initial_age_range=(cfg['adult_age'], cfg['elder_age']),
    )

    # A3: no quality investment — patch quality_share to 0
    # A4: no quality in pricing — patch goods_quality to 1.0
    no_quality_invest = not cfg.get('quality_investment_enabled', True)
    no_quality_pricing = not cfg.get('quality_in_pricing_enabled', True)

    prev = None
    series = []
    start = time.time()

    for tick in range(NUM_TICKS):
        prev = arrays
        arrays, key, stats = jax_step_on(arrays, key, cfg, tick)
        arrays, key, births, aging_deaths = jax_step_demographic(arrays, key, cfg)

        # Apply ablation patches after step
        if no_quality_invest:
            arrays = arrays._replace(tools_quality=prev.tools_quality if prev is not None else arrays.tools_quality)

        if no_quality_pricing:
            arrays = arrays._replace(goods_quality=jnp.ones_like(arrays.goods_quality))

        if (tick + 1) % SAMPLE_INTERVAL == 0:
            m = compute_metrics(arrays, stats, tick + 1, prev, cfg)
            series.append(m)

    elapsed = time.time() - start
    final = series[-1]

    return {
        'name': name,
        'overrides': overrides,
        'elapsed': elapsed,
        'time_series': [
            {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}
            for d in series
        ],
        'final': {k: final.get(k, 0) for k in KEY_METRICS},
    }


def main():
    import jax.numpy as jnp

    all_results = []

    for name, overrides in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        result = run_experiment(name, overrides)
        all_results.append(result)
        f = result['final']

        print(
            f"  pop={f['pop']} emp={f['employment_rate']:.3f} "
            f"tools={f['tools_mean']:.1f} wg={f['wealth_gini']:.3f} "
            f"tq={f['tools_quality_mean']:.3f} gq={f['goods_quality_mean']:.3f} "
            f"lend={f['lending_total']:.0f} trade={f['trade_volume']:.0f} "
            f"adv={f['adverse_selection']:.3f} qp={f['quality_premium']:.3f} "
            f"qlabor={f['quality_labor_share_mean']:.3f} "
            f"emp_tools={f['employer_tools_mean']:.1f} nonemp_tools={f['non_employer_tools_mean']:.1f} "
            f"| {result['elapsed']:.0f}s"
        )

    # Comparison table
    print(f"\n{'='*80}")
    print("  Ablation comparison (t=10000)")
    print(f"{'='*80}")

    import numpy as np
    header = f"{'metric':<30s}" + "".join(f"{name[:12]:>13s}" for name in EXPERIMENTS.keys())
    print(header)
    print("-" * len(header))

    for k in KEY_METRICS:
        row = f"{k:<30s}"
        for r in all_results:
            row += f"{r['final'][k]:13.3f}"
        print(row)

    # Key deltas vs baseline
    bl = all_results[0]['final']
    print(f"\n  --- vs Baseline deltas ---")
    for r in all_results[1:]:
        deltas = {k: r['final'][k] - bl[k] for k in KEY_METRICS}
        interesting = {k: v for k, v in deltas.items() if abs(v) > 0.01}
        print(f"\n  {r['name']}:")
        for k, v in sorted(interesting.items(), key=lambda x: -abs(x[1])):
            sign = "+" if v > 0 else ""
            print(f"    {k}: {sign}{v:.3f}")

    # Save
    output_dir = Path(__file__).parent.parent / 'findings' / 'sweep_data' / 'p41_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        fname = output_dir / f"{r['name']}_10k.json"
        with open(fname, 'w') as f:
            json.dump(r, f, indent=2, default=str)

    print(f"\nResults saved: {output_dir}")


if __name__ == '__main__':
    main()
