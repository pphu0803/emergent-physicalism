"""
Phase 4 压力测试 — 8 种极端场景

设计原则：
  - 单参数扫描已覆盖极端单参数值（p4_param_sweep）
  - 本实验聚焦「组合冲击」和「时序冲击」
  - 每个场景都有理论动机，不是纯粹凑极端值

场景设计：
  S1: 大萧条 — 低禀赋 + 低生产力 (inflow=0.3, production_scale=1.5)
      动机: 两个低于相变阈值的参数叠加，模拟经济衰退期

  S2: 黄金时代 — 高禀赋 + 高生产力 + 低折旧 (inflow=1.5, production_scale=5, depr=0.005)
      动机: 三个利好叠加，观察不平等是否失控、品质是否过热

  S3: 高周转高风险 — 高折旧 + 大资本空间 (depr=0.08, K=50)
      动机: 资本快速折旧但空间大，观察社会流动性特征

  S4: 工匠困境 — 小资本空间 + 低生产力 (K=3, production_scale=1.5)
      动机: 资本效率低 + 空间小，系统是否陷入低资本陷阱

  S5: 生存紧缩 — 高消耗 + 低禀赋 (subsistence=1.5, inflow=0.5)
      动机: inflow/subsistence=0.33，远低于品质激活阈值0.56

  S6: 时序冲击：资源断崖 — t=2000时inflow从0.72突降到0.3，持续1000 ticks后恢复
      动机: 模拟自然灾害/战争导致资源骤降，观察恢复能力

  S7: 时序冲击：折旧风暴 — t=2000时depreciation_rate从0.02升到0.10，持续500 ticks后恢复
      动机: 模拟技术毁灭/资本突然过时

  S8: 人口翻倍 — 400 agents初始化，观察系统是否在更大规模下保持均衡
      动机: 验证系统规模可扩展性
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
}

# S1-S5: Static combo scenarios (full duration)
STATIC_SCENARIOS = {
    'S1_great_depression': {
        'description': 'Great Depression: low endowment+low productivity',
        'overrides': {
            'natural_inflow_per_capita': 0.3,
            'production_scale': 1.5,
        },
        'num_agents': 200,
    },
    'S2_golden_age': {
        'description': 'Golden Age: high endowment+high productivity+low depreciation',
        'overrides': {
            'natural_inflow_per_capita': 1.5,
            'production_scale': 5.0,
            'depreciation_rate': 0.005,
        },
        'num_agents': 200,
    },
    'S3_high_turnover': {
        'description': 'High turnover: high depreciation+large capital space',
        'overrides': {
            'depreciation_rate': 0.08,
            'tool_saturation_K': 50.0,
        },
        'num_agents': 200,
    },
    'S4_artisan_trap': {
        'description': 'Artisan trap: small capital space+low productivity',
        'overrides': {
            'tool_saturation_K': 3.0,
            'production_scale': 1.5,
        },
        'num_agents': 200,
    },
    'S5_survival_crunch': {
        'description': 'Survival crunch: high cost+low endowment (inflow/subs=0.33)',
        'overrides': {
            'subsistence_cost': 1.5,
            'natural_inflow_per_capita': 0.5,
        },
        'num_agents': 200,
    },
}

# S6-S7: Temporal shock scenarios
TEMPORAL_SCENARIOS = {
    'S6_resource_cliff': {
        'description': 'Resource cliff: t=5000 inflow->0.05, 2000 ticks then recover',
        'shock_param': 'natural_inflow_per_capita',
        'normal_value': 0.72,
        'shock_value': 0.05,
        'shock_start': 5000,
        'shock_duration': 2000,
    },
    'S7_depreciation_storm': {
        'description': 'Depreciation storm: t=5000 depr->0.10, 500 ticks then recover',
        'shock_param': 'depreciation_rate',
        'normal_value': 0.02,
        'shock_value': 0.10,
        'shock_start': 5000,
        'shock_duration': 500,
    },
}

SEED = 42
NUM_TICKS = 10_000
SAMPLE_INTERVAL = 200  # finer sampling around shock windows

KEY_METRICS = [
    'pop', 'employment_rate', 'tools_mean', 'wealth_gini',
    'tools_quality_mean', 'goods_quality_mean', 'res_mean',
    'lending_total', 'avg_debt', 'adverse_selection', 'quality_premium',
    'trade_volume', 'skill_mean',
]


def run_static_scenario(name, scenario):
    """S1-S5: 组合冲击场景"""
    cfg = {**BASE_CONFIG, **scenario['overrides']}
    n_agents = scenario.get('num_agents', 200)

    arrays, key = init_agents(
        n_agents, seed=SEED, initial_endowment=10.0, num_tags=0,
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
        'name': name,
        'description': scenario['description'],
        'type': 'static',
        'overrides': scenario['overrides'],
        'num_agents': n_agents,
        'elapsed': elapsed,
        'time_series': [
            {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}
            for d in series
        ],
        'final': {k: final.get(k, 0) for k in KEY_METRICS},
    }


def run_temporal_scenario(name, scenario):
    """S6-S7: 时序冲击场景"""
    cfg = {**BASE_CONFIG}
    shock_start = scenario['shock_start']
    shock_end = scenario['shock_start'] + scenario['shock_duration']
    shock_param = scenario['shock_param']
    normal_value = scenario['normal_value']
    shock_value = scenario['shock_value']

    arrays, key = init_agents(
        200, seed=SEED, initial_endowment=10.0, num_tags=0,
        initial_age_range=(cfg['adult_age'], cfg['elder_age']),
    )

    prev = None
    series = []
    start = time.time()

    for tick in range(NUM_TICKS):
        # Apply shock
        if shock_start <= tick < shock_end:
            cfg[shock_param] = shock_value
        else:
            cfg[shock_param] = normal_value

        prev = arrays
        arrays, key, stats = jax_step_on(arrays, key, cfg, tick)
        arrays, key, births, aging_deaths = jax_step_demographic(arrays, key, cfg)

        # Finer sampling around shock window
        if (tick + 1) % SAMPLE_INTERVAL == 0:
            m = compute_metrics(arrays, stats, tick + 1, prev, cfg)
            m['shock_active'] = shock_start <= tick < shock_end
            series.append(m)

    elapsed = time.time() - start
    final = series[-1]

    # Compute pre-shock, during-shock, post-shock snapshots
    # Note: tick stored in series is tick+1 (the sample label), shock checks use raw tick
    pre_shock = [s for s in series if not s.get('shock_active', False)]
    pre_shock = pre_shock[-1] if pre_shock else None
    during_shock = [s for s in series if s.get('shock_active', False)]
    during_shock = during_shock[-1] if during_shock else None
    post_shock = [s for s in series if s.get('tick', 0) >= shock_end + SAMPLE_INTERVAL]
    post_shock = post_shock[0] if post_shock else None

    recovery = {}
    if pre_shock and post_shock:
        for k in KEY_METRICS:
            if k in pre_shock and k in post_shock:
                pre_val = pre_shock[k]
                post_val = post_shock[k]
                if abs(pre_val) > 0.01:
                    recovery[f'{k}_post_recovery_pct'] = (post_val / pre_val) * 100

    # Final recovery (t=10000 vs pre-shock)
    final_recovery = {}
    if pre_shock:
        for k in KEY_METRICS:
            if k in pre_shock and k in final:
                pre_val = pre_shock[k]
                fin_val = final[k]
                if abs(pre_val) > 0.01:
                    final_recovery[f'{k}_final_recovery_pct'] = (fin_val / pre_val) * 100

    return {
        'name': name,
        'description': scenario['description'],
        'type': 'temporal',
        'shock_param': shock_param,
        'shock_start': shock_start,
        'shock_end': shock_end,
        'normal_value': normal_value,
        'shock_value': shock_value,
        'pre_shock': {k: pre_shock.get(k, 0) for k in KEY_METRICS} if pre_shock else {},
        'during_shock': {k: during_shock.get(k, 0) for k in KEY_METRICS} if during_shock else {},
        'post_shock': {k: post_shock.get(k, 0) for k in KEY_METRICS} if post_shock else {},
        'post_recovery_pct': recovery,
        'final_recovery_pct': final_recovery,
        'elapsed': elapsed,
        'time_series': [
            {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}
            for d in series
        ],
        'final': {k: final.get(k, 0) for k in KEY_METRICS},
    }


def run_scale_scenario():
    """S8: 400 agents"""
    cfg = {**BASE_CONFIG}
    n_agents = 400

    arrays, key = init_agents(
        n_agents, seed=SEED, initial_endowment=10.0, num_tags=0,
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
        'name': 'S8_double_population',
        'description': '人口翻倍: 400 agents',
        'type': 'scale',
        'num_agents': n_agents,
        'elapsed': elapsed,
        'time_series': [
            {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}
            for d in series
        ],
        'final': {k: final.get(k, 0) for k in KEY_METRICS},
    }


def main():
    all_results = []

    # --- S1-S5: Static combo scenarios ---
    for name, scenario in STATIC_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  {name}: {scenario['description']}")
        print(f"{'='*60}")

        result = run_static_scenario(name, scenario)
        all_results.append(result)
        f = result['final']

        print(
            f"  pop={f['pop']} emp={f['employment_rate']:.3f} "
            f"tools={f['tools_mean']:.1f} wg={f['wealth_gini']:.3f} "
            f"tq={f['tools_quality_mean']:.3f} gq={f['goods_quality_mean']:.3f} "
            f"lend={f['lending_total']:.0f} trade={f['trade_volume']:.0f} "
            f"res={f['res_mean']:.0f} "
            f"| {result['elapsed']:.0f}s"
        )

    # --- S6-S7: Temporal shock scenarios ---
    for name, scenario in TEMPORAL_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  {name}: {scenario['description']}")
        print(f"{'='*60}")

        result = run_temporal_scenario(name, scenario)
        all_results.append(result)

        # Print impact summary
        pre = result.get('pre_shock', {})
        during = result.get('during_shock', {})
        post = result.get('post_shock', {})
        final = result['final']

        print(f"  Pre-shock:  tools={pre.get('tools_mean',0):.1f} gini={pre.get('wealth_gini',0):.3f} res={pre.get('res_mean',0):.0f} lend={pre.get('lending_total',0):.0f}")
        print(f"  During:     tools={during.get('tools_mean',0):.1f} gini={during.get('wealth_gini',0):.3f} res={during.get('res_mean',0):.0f} lend={during.get('lending_total',0):.0f}")
        print(f"  Post-shock: tools={post.get('tools_mean',0):.1f} gini={post.get('wealth_gini',0):.3f} res={post.get('res_mean',0):.0f} lend={post.get('lending_total',0):.0f}")
        print(f"  Recovery%:  tools={result['post_recovery_pct'].get('tools_mean_post_recovery_pct',0):.1f}% "
              f"gini={result['post_recovery_pct'].get('wealth_gini_post_recovery_pct',0):.1f}% "
              f"res={result['post_recovery_pct'].get('res_mean_post_recovery_pct',0):.1f}%")
        print(f"  Final%:    tools={result['final_recovery_pct'].get('tools_mean_final_recovery_pct',0):.1f}% "
              f"gini={result['final_recovery_pct'].get('wealth_gini_final_recovery_pct',0):.1f}% "
              f"res={result['final_recovery_pct'].get('res_mean_final_recovery_pct',0):.1f}%")
        print(f"  | {result['elapsed']:.0f}s")

    # --- S8: Scale test ---
    print(f"\n{'='*60}")
    print(f"  S8_double_population: 400 agents")
    print(f"{'='*60}")

    result = run_scale_scenario()
    all_results.append(result)
    f = result['final']

    print(
        f"  pop={f['pop']} emp={f['employment_rate']:.3f} "
        f"tools={f['tools_mean']:.1f} wg={f['wealth_gini']:.3f} "
        f"tq={f['tools_quality_mean']:.3f} gq={f['goods_quality_mean']:.3f} "
        f"lend={f['lending_total']:.0f} trade={f['trade_volume']:.0f} "
        f"res={f['res_mean']:.0f} "
        f"| {result['elapsed']:.0f}s"
    )

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("  Stress test summary (t=10000)")
    print(f"{'='*80}")

    header = f"{'scenario':<28s}"
    for k in ['pop', 'emp', 'tools', 'gini', 'tq', 'gq', 'lend', 'trade', 'res']:
        header += f" {k:>8s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        f = r['final']
        row = f"{r['name']:<28s}"
        row += f" {f['pop']:8.0f}"
        row += f" {f['employment_rate']:8.3f}"
        row += f" {f['tools_mean']:8.1f}"
        row += f" {f['wealth_gini']:8.3f}"
        row += f" {f['tools_quality_mean']:8.3f}"
        row += f" {f['goods_quality_mean']:8.3f}"
        row += f" {f['lending_total']:8.0f}"
        row += f" {f['trade_volume']:8.0f}"
        row += f" {f['res_mean']:8.0f}"
        print(row)

    # Save
    output_dir = Path(__file__).parent.parent / 'findings' / 'sweep_data' / 'p41_stress_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        fname = output_dir / f"{r['name']}.json"
        with open(fname, 'w') as f:
            json.dump(r, f, indent=2, default=str)

    print(f"\nResults saved: {output_dir}")
    print(f"Total time: {sum(r['elapsed'] for r in all_results):.0f}s")


if __name__ == '__main__':
    main()
