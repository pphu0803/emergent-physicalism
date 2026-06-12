"""
Phase 4 运行时指标 — 经济内部演变分析

设计原则：
- 砍掉 Phase 4 不用的（暴力、奢侈品、标签相关）
- 增加解释"为什么"的指标（按年龄组拆分、投资边际、生产结构）
- 所有指标纯向量化，无 Python 循环

指标分层：
  L0 结构：人口、年龄分布、劳动能力
  L1 生产：产出、投资、品质、自给自足转换
  L2 分配：不平等、按年龄组的资源分布
  L3 交换：贸易量/深度、雇佣、借贷
  L4 演化：技能漂移、工具收敛
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
from core.state import AgentArrays


# ============================================================
# 辅助函数
# ============================================================

def _gini(values: jax.Array, mask: jax.Array) -> float:
    vals = jnp.where(mask, values, 0.0)
    n = jnp.sum(mask).astype(jnp.int32)
    vals = jnp.sort(vals)
    cumw = jnp.cumsum(vals)
    total = cumw[-1]
    return jnp.where(
        (n < 2) | (total <= 0),
        0.0,
        jnp.clip((n + 1 - 2 * jnp.sum(cumw) / total) / n, 0.0, 1.0),
    )


def _top_k_share(values: jax.Array, mask: jax.Array, k_pct: float) -> float:
    alive_vals = jnp.where(mask, values, 0.0)
    total = jnp.sum(alive_vals)
    n = jnp.sum(mask).astype(jnp.int32)
    k_count = jnp.maximum(1, (n * k_pct).astype(jnp.int32))
    sorted_desc = jnp.sort(alive_vals)[::-1]
    return jnp.where(total > 0, jnp.sum(sorted_desc[:k_count]) / total, 0.0)


def _masked_mean(values: jax.Array, mask: jax.Array) -> float:
    n = jnp.maximum(1, jnp.sum(mask))
    return float(jnp.sum(jnp.where(mask, values, 0.0)) / n)


# ============================================================
# L0: 结构指标 — 人口、年龄、劳动能力
# ============================================================

def compute_structure_metrics(arrays: AgentArrays, config: dict) -> Dict[str, Any]:
    alive = arrays.alive
    n = jnp.maximum(1, jnp.sum(alive))
    ages = arrays.age.astype(jnp.float32)

    adult_age = float(config.get('adult_age', 936))
    elder_age = float(config.get('elder_age', 3120))

    young = alive & (ages < adult_age)
    prime = alive & (ages >= adult_age) & (ages < elder_age)
    elder = alive & (ages >= elder_age)

    from core.jax_steps import jax_age_capacity
    wc = jax_age_capacity(arrays, config)

    return {
        # 人口
        'pop': int(jnp.sum(alive)),
        'pop_young': int(jnp.sum(young)),
        'pop_prime': int(jnp.sum(prime)),
        'pop_elder': int(jnp.sum(elder)),
        # 劳动能力
        'wc_mean': _masked_mean(wc, alive),
        'wc_young_mean': _masked_mean(wc, young),
        'wc_prime_mean': _masked_mean(wc, prime),
        'wc_elder_mean': _masked_mean(wc, elder),
        # 年龄
        'age_mean': _masked_mean(ages, alive),
    }


# ============================================================
# L1: 生产指标 — 产出、投资、品质
# ============================================================

def compute_production_metrics(arrays: AgentArrays, config: dict) -> Dict[str, Any]:
    alive = arrays.alive
    n = jnp.maximum(1, jnp.sum(alive))

    K = float(config.get('tool_saturation_K', 10.0))
    production_scale = float(config.get('production_scale', 3.0))
    subsistence = float(config.get('subsistence_cost', 0.9))

    from core.jax_steps import jax_age_capacity, _effective_tool_bonus
    wc = jax_age_capacity(arrays, config)
    eff_tool_b = _effective_tool_bonus(arrays.tools, config)

    # 生产潜力 = ew × wc × effective_tool_bonus × scale
    output_potential = arrays.effective_skill * wc * eff_tool_b * production_scale

    # 工具资本化度 = effective_bonus / 个人级bonus(t=K, q=1)
    tool_capitalization = eff_tool_b / (1.0 + jnp.log(2.0))

    # 投资边际收益 = d(1+ln(1+t/K))/d(tools) = 1 / (K + t)
    marginal_return = 1.0 / (K + arrays.tools)

    # 自给自足转换量估算
    eff_goods = arrays.goods * arrays.goods_quality
    goods_convert = eff_goods * 0.3

    # 品质投资劳动占比（几何均值：品质稀缺度 × 提升空间）
    can_improve = arrays.tools_quality < arrays.skill
    quality_share = jnp.where(
        can_improve,
        jnp.sqrt(
            arrays.tools * jnp.maximum(arrays.skill - arrays.tools_quality, 0.0)
            / (arrays.skill * (arrays.tools + arrays.tools_quality) + 1e-8)
        ),
        0.0,
    )

    return {
        # 产出
        'output_potential_mean': _masked_mean(output_potential, alive),
        'output_potential_total': float(jnp.sum(jnp.where(alive, output_potential, 0.0))),
        # 工具
        'tools_mean': _masked_mean(arrays.tools, alive),
        'tools_gini': float(_gini(jnp.where(alive, arrays.tools, 0.0), alive)),
        'tool_bonus_mean': _masked_mean(eff_tool_b, alive),
        'tool_saturation_mean': _masked_mean(tool_capitalization, alive),
        'marginal_return_mean': _masked_mean(marginal_return, alive),
        # 品质
        'goods_quality_mean': _masked_mean(arrays.goods_quality, alive),
        'tools_quality_mean': _masked_mean(arrays.tools_quality, alive),
        # 品质投资劳动占比
        'quality_labor_share_mean': _masked_mean(quality_share, alive),
        # 商品与消费
        'goods_mean': _masked_mean(arrays.goods, alive),
        'goods_gini': float(_gini(jnp.where(alive, arrays.goods, 0.0), alive)),
        'goods_convert_mean': _masked_mean(goods_convert, alive),
        # 购买意愿
        'buy_urgency_mean': _masked_mean(1.0 / (1.0 + eff_goods / subsistence), alive),
    }


# ============================================================
# L2: 分配指标 — 不平等 + 按年龄组拆分
# ============================================================

def compute_distribution_metrics(arrays: AgentArrays, config: dict) -> Dict[str, Any]:
    alive = arrays.alive
    n = jnp.maximum(1, jnp.sum(alive))
    ages = arrays.age.astype(jnp.float32)

    adult_age = float(config.get('adult_age', 936))
    elder_age = float(config.get('elder_age', 3120))

    young = alive & (ages < adult_age)
    prime = alive & (ages >= adult_age) & (ages < elder_age)
    elder = alive & (ages >= elder_age)

    res = jnp.where(alive, arrays.resources, 0.0)
    # effective wealth = resources + tools (amount only for gini comparison)
    wealth = jnp.where(alive, jnp.maximum(0, arrays.resources + arrays.tools), 0.0)
    # effective wealth including quality
    eff_wealth = jnp.where(
        alive,
        jnp.maximum(0, arrays.resources
                     + arrays.tools * arrays.tools_quality
                     + arrays.goods * arrays.goods_quality),
        0.0,
    )

    # 不平等总指标
    result = {
        'res_mean': _masked_mean(arrays.resources, alive),
        'res_gini': float(_gini(res, alive)),
        'wealth_gini': float(_gini(wealth, alive)),
        'eff_wealth_gini': float(_gini(eff_wealth, alive)),
        'wealth_top10': float(_top_k_share(wealth, alive, 0.10)),
        'wealth_bot50': float(1.0 - _top_k_share(wealth, alive, 0.50)),
    }

    # 按年龄组拆分
    for group_name, group_mask in [('young', young), ('prime', prime), ('elder', elder)]:
        result[f'res_{group_name}_mean'] = _masked_mean(arrays.resources, group_mask)
        result[f'tools_{group_name}_mean'] = _masked_mean(arrays.tools, group_mask)
        result[f'wealth_{group_name}_mean'] = _masked_mean(
            jnp.where(alive, arrays.resources + arrays.tools, 0.0), group_mask)

    # 代际转移比率
    prime_res = _masked_mean(arrays.resources, prime)
    elder_res = _masked_mean(arrays.resources, elder)
    result['intergen_transfer_ratio'] = float(
        elder_res / max(prime_res, 1e-8))

    return result


# ============================================================
# L3: 交换指标 — 贸易、雇佣、借贷
# ============================================================

def compute_exchange_metrics(arrays: AgentArrays, stats: dict, config: dict) -> Dict[str, Any]:
    alive = arrays.alive
    n = jnp.maximum(1, jnp.sum(alive))
    ages = arrays.age.astype(jnp.float32)

    adult_age = float(config.get('adult_age', 936))
    elder_age = float(config.get('elder_age', 3120))

    trades = stats.get('total_trades', 0)
    employers = arrays.is_employer & alive
    employees = (arrays.employer_id >= 0) & alive & ~arrays.is_employer

    young = alive & (ages < adult_age)
    prime = alive & (ages >= adult_age) & (ages < elder_age)
    elder = alive & (ages >= elder_age)

    is_employee = (arrays.employer_id >= 0) & alive & ~arrays.is_employer
    young_employees = is_employee & young
    prime_employees = is_employee & prime
    elder_employees = is_employee & elder

    employer_tools = _masked_mean(arrays.tools, employers)
    employer_res = _masked_mean(arrays.resources, employers)
    non_employer_tools = _masked_mean(arrays.tools, alive & ~arrays.is_employer)

    # 雇主品质
    employer_tools_quality = _masked_mean(arrays.tools_quality, employers)
    non_employer_tools_quality = _masked_mean(arrays.tools_quality, alive & ~arrays.is_employer)

    debt = jnp.where(alive, arrays.debt, 0.0)

    return {
        # 贸易
        'trade_volume': trades,
        'trade_per_agent': float(trades) / float(n),
        # 雇佣
        'employment_rate': float(jnp.sum(employees)) / float(n),
        'employer_count': int(jnp.sum(employers)),
        'employer_tools_mean': employer_tools,
        'employer_res_mean': employer_res,
        'non_employer_tools_mean': non_employer_tools,
        # 品质差距
        'employer_tools_quality_mean': employer_tools_quality,
        'non_employer_tools_quality_mean': non_employer_tools_quality,
        # 按年龄组就业率
        'employment_young': float(jnp.sum(young_employees)) / max(1, float(jnp.sum(young))),
        'employment_prime': float(jnp.sum(prime_employees)) / max(1, float(jnp.sum(prime))),
        'employment_elder': float(jnp.sum(elder_employees)) / max(1, float(jnp.sum(elder))),
        # 借贷
        'lending_total': float(stats.get('total_lending', 0)),
        'avg_debt': _masked_mean(arrays.debt, alive),
        'debt_gini': float(_gini(debt, alive)),
        # 柠檬市场指标
        'adverse_selection': float(stats.get('adverse_selection', 0)),
        'quality_premium': float(stats.get('quality_premium', 0)),
    }


# ============================================================
# L4: 演化指标 — 技能漂移、工具收敛
# ============================================================

def compute_evolution_metrics(arrays: AgentArrays, config: dict) -> Dict[str, Any]:
    alive = arrays.alive
    n = jnp.maximum(1, jnp.sum(alive))

    skill = jnp.where(alive, arrays.skill, 0.0)
    ew = jnp.where(alive, arrays.effective_skill, 0.0)

    skill_mean = jnp.sum(skill) / n
    skill_var = jnp.sum(jnp.where(alive, (arrays.skill - skill_mean) ** 2, 0.0)) / n
    skill_std = jnp.sqrt(jnp.maximum(skill_var, 0.0))

    exp_bonus_rate = jnp.where(
        alive & (arrays.skill > 0.01),
        arrays.effective_skill / jnp.maximum(arrays.skill, 0.01) - 1.0,
        0.0,
    )

    price = jnp.where(alive, arrays.price_belief, 0.0)
    price_mean = jnp.sum(price) / n
    price_var = jnp.sum(jnp.where(alive, (arrays.price_belief - price_mean) ** 2, 0.0)) / n

    return {
        'skill_mean': float(skill_mean),
        'skill_std': float(skill_std),
        'skill_gini': float(_gini(skill, alive)),
        'ew_mean': _masked_mean(arrays.effective_skill, alive),
        'exp_bonus_rate_mean': _masked_mean(exp_bonus_rate, alive),
        'price_mean': float(price_mean),
        'price_std': float(jnp.sqrt(jnp.maximum(price_var, 0.0))),
    }


# ============================================================
# 聚合入口
# ============================================================

def compute_metrics(
    arrays: AgentArrays,
    stats: dict,
    tick: int,
    prev_arrays: Optional[AgentArrays] = None,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    config = config or {}
    m = {'tick': tick}

    m.update(compute_structure_metrics(arrays, config))
    m.update(compute_production_metrics(arrays, config))
    m.update(compute_distribution_metrics(arrays, config))
    m.update(compute_exchange_metrics(arrays, stats, config))
    m.update(compute_evolution_metrics(arrays, config))

    # 服务经济指标
    if prev_arrays is not None:
        # GDP_services = tools × Δ(tools_quality) — 品质改善
        delta_tq = jnp.maximum(0, arrays.tools_quality - prev_arrays.tools_quality)
        gdp_services = float(jnp.sum(jnp.where(arrays.alive, arrays.tools * delta_tq, 0.0)))
        gdp_goods = float(jnp.sum(jnp.where(arrays.alive, arrays.goods * arrays.goods_quality, 0.0)))
        m['gdp_goods'] = gdp_goods
        m['gdp_services'] = gdp_services
        m['service_share'] = gdp_services / max(gdp_goods + gdp_services, 1e-8)
    else:
        m['gdp_goods'] = 0.0
        m['gdp_services'] = 0.0
        m['service_share'] = 0.0

    # 人口动态
    if prev_arrays is not None:
        new_dead = prev_arrays.alive & ~arrays.alive
        new_born = arrays.alive & ~prev_arrays.alive
        m['births'] = int(jnp.sum(new_born))
        m['deaths'] = int(jnp.sum(new_dead))
    else:
        m['births'] = 0
        m['deaths'] = 0

    return m
