"""
JAX Pure Function Stepping -- O(N) + O(N^2) subsystems

Core design:
- jax_step_core: pure JAX function, @jax.jit compiled, returns (AgentArrays, PRNGKey)
- jax_step_on: outer wrapper, calls core then computes Python statistics
- All subsystems can be independently @jax.jit compiled
"""

import jax
import jax.numpy as jnp
import functools

from .state import AgentArrays, SECTOR_NECESSITIES


# ============================================================
# MetaLaw: Survival Baseline (ML-5)
# ============================================================

def jax_enforce_survival(arrays: AgentArrays, config: dict) -> AgentArrays:
    subsistence = config.get('subsistence_cost', 0.9)

    consumed_amount = jnp.minimum(subsistence, arrays.resources)
    remaining = arrays.resources - consumed_amount
    alive = arrays.alive & (remaining >= 0)

    return arrays._replace(
        resources=jnp.where(alive, remaining, 0.0),
        alive=alive,
    )


# ============================================================
# MetaLaw: Material Conservation (ML-1) -- Natural Resource Inflow
# ============================================================

def jax_enforce_conservation(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    inflow_per_capita = config.get('natural_inflow_per_capita', 0.72)
    resource_ceiling = config.get('resource_ceiling', 5000.0)

    alive_count = jnp.sum(arrays.alive).astype(jnp.float32)
    inflow = inflow_per_capita * alive_count

    total_current = jnp.sum(jnp.where(arrays.alive, arrays.resources, 0.0))
    inflow = jnp.minimum(inflow, jnp.maximum(0.0, resource_ceiling - total_current))

    per_agent = jnp.where(alive_count > 0, inflow / alive_count, 0.0)
    new_resources = jnp.where(arrays.alive, arrays.resources + per_agent, arrays.resources)

    new_key = jax.random.fold_in(key, 1)
    return arrays._replace(resources=new_resources), new_key


# ============================================================
# MetaLaw: Labor Capacity (ML-2a)
# ============================================================

def jax_enforce_labor(arrays: AgentArrays, config: dict) -> AgentArrays:
    labor = 1.0 - arrays.reproduction_penalty
    return arrays._replace(
        labor_remaining=labor,
        reproduction_penalty=jnp.zeros_like(arrays.reproduction_penalty),
    )


# ============================================================
# Age Lifecycle -- Capacity Coefficient
# ============================================================

def jax_age_capacity(arrays: AgentArrays, config: dict) -> jax.Array:
    """Age-stage capacity coefficient -- continuous S-curve

    Smooth rise from youth to prime age, smooth decline from prime to elderly.
    Range [0.3, 1.0], no new parameters introduced.
    """
    adult_age = config.get('adult_age', 936)
    elder_age = config.get('elder_age', 3120)
    sigma = adult_age / 5.0

    age = arrays.age.astype(jnp.float32)
    rising = jax.nn.sigmoid((age - adult_age) / sigma)
    falling = 1.0 - jax.nn.sigmoid((age - elder_age) / sigma)
    return rising * falling * 0.7 + 0.3


# ============================================================
# Tool Bonus -- Diminishing Returns
# ============================================================

def _effective_tool_bonus(tools, config):
    """Tool bonus -- diminishing returns

    bonus = 1 + ln(1 + tools / K)

    Quality does not affect production quantity, only output quality (output_quality).
    Real-world analogy: precision machines don't produce more parts, only better parts.
    """
    K = config.get('tool_saturation_K', 10.0)
    return 1.0 + jnp.log1p(tools / K)


def _merge_quality(old_amount, old_quality, new_amount, new_quality):
    """Quality merge: effective value conservation weighted average"""
    total = old_amount + new_amount
    return jnp.where(
        total > 1e-8,
        (old_amount * old_quality + new_amount * new_quality) / total,
        new_quality,
    )


# ============================================================
# Sector Production
# ============================================================

def jax_sector_production(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    """Sector production -- labor split: goods production vs quality investment

    Allocation by marginal return ratio:
    quality_share = tools * (skill - tq) / (skill * (tools + tq))
    Factor 1: tools/(tools+tq) -- high quality return when tools are abundant and quality is low
    Factor 2: (skill-tq)/skill -- investment makes sense when quality improvement space is large
    Both naturally in [0,1], product naturally in [0,1], no magic numbers
    """
    production_scale = config.get('production_scale', 3.0)
    q_rate = config.get('quality_improvement_rate', 0.1)
    n = arrays.alive.shape[0]

    noise = jax.random.normal(key, (n,)) * 0.05
    work_capacity = jax_age_capacity(arrays, config)
    labor = arrays.labor_remaining * work_capacity

    # 经验加成：边际递减，上界 +30%
    exp_bonus = 1.0 + 0.3 * (1.0 - jnp.exp(-arrays.production_experience * 0.01))
    eff_skill = arrays.skill * exp_bonus
    eff_skill = jnp.where(arrays.coercion_debuff_ticks > 0, eff_skill * 0.7, eff_skill)

    eff_tool_b = _effective_tool_bonus(arrays.tools, config)

    # --- Labor split: marginal return ratio x improvement space (geometric mean, avoids quadratic gap) ---
    # Factor 1: tools/(tools+tq) -- quality scarcity
    # Factor 2: (skill-tq)/skill -- improvement space
    # Geometric mean: sqrt(f1 x f2) -- combined effect of two [0,1] factors
    can_improve = arrays.tools_quality < arrays.skill
    quality_share = jnp.where(
        can_improve,
        jnp.sqrt(
            arrays.tools * jnp.maximum(arrays.skill - arrays.tools_quality, 0.0)
            / (arrays.skill * (arrays.tools + arrays.tools_quality) + 1e-8)
        ),
        0.0,
    )
    production_labor = labor * (1.0 - quality_share)
    quality_labor = labor * quality_share

    # --- Goods production ---
    output = eff_skill * production_labor * eff_tool_b * production_scale * (1.0 + noise)
    output = jnp.maximum(0.0, output)

    # --- Quality investment: trade labor time for quality improvement ---
    # sqrt(gap): avoids quadratic gap effect (quality_share already has gap, linear gain would be O(gap^2) making maintenance impossible)
    quality_gain = quality_labor * eff_skill * q_rate * jnp.sqrt(jnp.maximum(arrays.skill - arrays.tools_quality, 0.0))
    quality_gain = jnp.maximum(0.0, quality_gain)
    new_tq = arrays.tools_quality + quality_gain

    # Output quality = sqrt(skill * tq) (geometric mean: bottleneck effect)
    output_quality = jnp.sqrt(eff_skill * jnp.maximum(new_tq, 1e-8))

    # Experience accumulation
    marginal_learning = 0.02 / (1.0 + arrays.production_experience * 0.1)
    new_production_exp = arrays.production_experience + labor * (1.0 + marginal_learning)
    new_sector_exp = arrays.sector_experience.at[:, 0].add(
        labor * (1.0 + marginal_learning)
    )

    # goods + goods_quality merge
    new_goods = jnp.where(arrays.alive, arrays.goods + output, arrays.goods)
    new_goods_quality = jnp.where(
        arrays.alive,
        _merge_quality(arrays.goods, arrays.goods_quality, output, output_quality),
        arrays.goods_quality,
    )
    new_labor = jnp.where(arrays.alive, 0.0, arrays.labor_remaining)

    new_arrays = arrays._replace(
        goods=new_goods, goods_quality=new_goods_quality,
        tools_quality=new_tq,
        labor_remaining=new_labor,
        production_experience=new_production_exp, sector_experience=new_sector_exp,
        effective_skill=eff_skill,
    )

    new_key = jax.random.fold_in(key, 0)
    return new_arrays, new_key


# ============================================================
# Self-Sufficiency -- production output directly fills survival gap
# ============================================================

def jax_self_sufficiency(arrays: AgentArrays, config: dict) -> AgentArrays:
    """自给自足：消耗 goods 维持生存

    1. 应急：resources 不足时，将 goods 转换为 resources 弥补缺口
    2. 常规：消耗 30% 的 goods（生物代谢需求），转化为 resources
    生存转换按数量（卡路里），不按品质（口味）—— 品质影响市场价值，不影响生存
    """
    subsistence = config.get('subsistence_cost', 0.9)

    # 应急：弥补 resource 缺口（按数量，不按品质）
    gap = jnp.maximum(0.0, subsistence - arrays.resources)
    emergency = jnp.minimum(arrays.goods, gap)

    # 常规消费：30% 的 goods 转为 resources（生物代谢）
    normal = arrays.goods * 0.3

    convert = jnp.maximum(emergency, normal)
    convert = jnp.where(arrays.alive, convert, 0.0)

    return arrays._replace(
        resources=arrays.resources + convert,
        goods=arrays.goods - convert,
    )


# ============================================================
# Tool Investment + Depreciation
# ============================================================

def jax_investment(arrays: AgentArrays, config: dict) -> AgentArrays:
    subsistence = config.get('subsistence_cost', 0.9)
    K = config.get('tool_saturation_K', 10.0)

    surplus = jnp.maximum(0.0, arrays.resources - subsistence * 2)
    marginal_return = 1.0 / (K + arrays.tools)
    invest = surplus * 0.1 * marginal_return * (arrays.skill / 0.2)
    invest = jnp.where(arrays.alive & (surplus > 0), invest, 0.0)
    invest = jnp.clip(invest, 0.0, arrays.resources)

    new_tools = arrays.tools + invest

    return arrays._replace(
        tools=new_tools,
        resources=arrays.resources - invest,
    )


def jax_depreciation(arrays: AgentArrays, config: dict) -> AgentArrays:
    rate = config.get('depreciation_rate', 0.02)
    tools_q_decay = config.get('tools_q_decay', 0.0005)
    goods_q_decay = config.get('goods_q_decay', 0.005)

    # Quantity depreciation
    new_tools = jnp.maximum(0.0, arrays.tools * (1 - rate))

    # Quality depreciation -- simple constant decay
    new_tools_quality = arrays.tools_quality * (1 - tools_q_decay)
    new_goods_quality = arrays.goods_quality * (1 - goods_q_decay)

    return arrays._replace(
        tools=new_tools, tools_quality=new_tools_quality,
        goods_quality=new_goods_quality,
    )


# ============================================================
# Demographic Dynamics (ML-9) -- Reproduction (fully vectorized, no fori_loop)
# ============================================================

def jax_demographic(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    """ML-9: Demographic dynamics -- reproduction + aging + age lifecycle

    First principles:
    - Reproduction: resource-abundant prime-age individuals produce offspring with probability (biological fact)
    - Aging: the longer an individual survives, the higher its death probability (embodiment of irreversibility ML-3)
    - Age stages: minors cannot reproduce, elderly cannot reproduce (lifecycle constraint)
    """
    pop_ceiling = config.get('population_ceiling', 500)
    subsistence = config.get('subsistence_cost', 0.9)
    repro_threshold = subsistence * 2.0
    base_prob = config.get('reproduction_base_prob', 0.01)
    cooldown_ticks = config.get('reproduction_cooldown', 20)
    cost_ratio = 0.2
    inheritance = config.get('inheritance_ratio', 0.8)
    mutation = config.get('mutation_strength', 0.15)
    aging_enabled = config.get('aging_enabled', True)
    aging_base = config.get('aging_base_rate', 0.001)
    aging_mult = config.get('aging_rate_multiplier', 0.001)
    adult_age = config.get('adult_age', 936)
    elder_age = config.get('elder_age', 3120)

    n = arrays.alive.shape[0]
    keys = jax.random.split(key, 3)

    # --- 1. Age increment + aging death ---
    new_age = jnp.where(arrays.alive, arrays.age + 1, arrays.age)
    # Gompertz aging: p_death = a0 * exp(b * age), cap at 0.1
    # Nearly no death when young (safety window), exponential rise when elderly (natural aging)
    # a0=0.0001, b=0.0005: S(936)=0.89, S(3120)=0.47, median lifespan ~3000
    aging_death_prob = jnp.where(
        aging_enabled,
        jnp.clip(aging_base * jnp.exp(aging_mult * new_age.astype(jnp.float32)), 0.0, 0.1),
        0.0,
    )
    aging_roll = jax.random.uniform(keys[0], (n,))
    aging_death = arrays.alive & (aging_roll < aging_death_prob)

    # --- 2. Reproduction ---
    alive_count = jnp.sum(arrays.alive)
    capacity_pressure = jnp.clip(
        1.0 - (alive_count - pop_ceiling * 0.8) / (pop_ceiling * 0.2), 0.0, 1.0,
    )
    # Age constraint: work_capacity > 0.5 means sufficient labor capacity to reproduce
    work_cap = jax_age_capacity(arrays._replace(age=new_age), config)
    is_fertile = work_cap > 0.5
    can_reproduce = (
        arrays.alive & is_fertile & (arrays.reproduction_cooldown <= 0)
        & (arrays.resources > repro_threshold) & (capacity_pressure > 0)
    )
    surplus_ratio = jnp.clip(arrays.resources / repro_threshold - 1.0, 0.0, 5.0)
    p_reproduce = base_prob * surplus_ratio * capacity_pressure
    repro_roll = jax.random.uniform(keys[1], (n,))
    reproducing = can_reproduce & (repro_roll < p_reproduce)

    # --- 3. Death mask (original death + aging death) ---
    dead_mask = ~arrays.alive | aging_death

    # --- 4. Parent update (transfer computed from original resources, fixing double-deduction bug) ---
    child_transfer = jnp.where(reproducing, arrays.resources * cost_ratio, 0.0)

    new_arrays = arrays._replace(
        age=new_age,
        alive=jnp.where(aging_death, False, arrays.alive),
        resources=jnp.where(reproducing, arrays.resources - child_transfer, arrays.resources),
        reproduction_cooldown=jnp.where(reproducing, cooldown_ticks, arrays.reproduction_cooldown),
        reproduction_penalty=jnp.where(reproducing, 0.4, 0.0),
        labor_remaining=jnp.where(reproducing, arrays.labor_remaining * 0.6, arrays.labor_remaining),
    )

    # --- 5. Offspring fill dead slots (unique slot assignment) ---
    # Use argsort to find sorted indices of all dead slots, ensuring each birth occupies a unique slot
    MAX_BIRTHS = n // 10
    # argsort sorts False(0) first True(1) last
    # Last n_dead elements of dead_sorted are the dead slot indices
    dead_sorted = jnp.argsort(dead_mask.astype(jnp.int32))
    # Last n_repro elements of repro_sorted are the reproducing parent indices
    repro_sorted = jnp.argsort(reproducing.astype(jnp.int32))

    birth_range = jnp.arange(MAX_BIRTHS, dtype=jnp.int32)
    # Take from the end: dead_sorted[n-1], dead_sorted[n-2], ... are dead slots
    # Use jnp.maximum(0, n - birth_range - 1) as index, skip 0-index boundary
    didx = dead_sorted[jnp.maximum(0, n - birth_range - 1)]
    pidx = repro_sorted[jnp.maximum(0, n - birth_range - 1)]

    # n_dead 和 n_repro 是 traced，但 birth_range 是静态的
    # valid 条件：slot 确实是 dead 且 parent 确实在 reproducing
    n_dead = jnp.sum(dead_mask).astype(jnp.int32)
    n_repro = jnp.sum(reproducing).astype(jnp.int32)
    has_birth = (
        (birth_range < n_dead) & (birth_range < n_repro) & (birth_range < MAX_BIRTHS)
        & dead_mask[didx] & reproducing[pidx]
    )

    # Build per-slot markers
    newborn_at = jnp.zeros(n, dtype=jnp.bool_)
    newborn_at = newborn_at.at[didx].set(jnp.where(has_birth, True, newborn_at[didx]))

    # parent -> child traits
    k_skill, k_risk = jax.random.split(keys[2], 2)
    child_skill = jnp.clip(
        arrays.skill * inheritance
        + jax.random.beta(k_skill, 2, 5, shape=(n,)) * (1 - inheritance)
        + jax.random.normal(jax.random.fold_in(k_skill, 0), shape=(n,)) * mutation,
        0.01, 0.99,
    )
    child_risk = jnp.clip(
        arrays.risk_tolerance * inheritance
        + jax.random.beta(k_risk, 2, 2, shape=(n,)) * (1 - inheritance)
        + jax.random.normal(jax.random.fold_in(k_risk, 0), shape=(n,)) * mutation * 0.5,
        0.01, 0.99,
    )

    # scatter parent data to unique dead slot positions
    child_res_at_slot = jnp.zeros(n, dtype=jnp.float32)
    child_skill_at_slot = jnp.zeros(n, dtype=jnp.float32)
    child_risk_at_slot = jnp.zeros(n, dtype=jnp.float32)
    child_tag_at_slot = jnp.zeros(n, dtype=jnp.int32)
    child_parent_at_slot = jnp.full(n, -1, dtype=jnp.int32)
    for i in range(MAX_BIRTHS):
        valid = has_birth[i]
        child_res_at_slot = child_res_at_slot.at[didx[i]].set(
            jnp.where(valid, child_transfer[pidx[i]], child_res_at_slot[didx[i]]))
        child_skill_at_slot = child_skill_at_slot.at[didx[i]].set(
            jnp.where(valid, child_skill[pidx[i]], child_skill_at_slot[didx[i]]))
        child_risk_at_slot = child_risk_at_slot.at[didx[i]].set(
            jnp.where(valid, child_risk[pidx[i]], child_risk_at_slot[didx[i]]))
        child_tag_at_slot = child_tag_at_slot.at[didx[i]].set(
            jnp.where(valid, arrays.tag[pidx[i]], child_tag_at_slot[didx[i]]))
        child_parent_at_slot = child_parent_at_slot.at[didx[i]].set(
            jnp.where(valid, pidx[i], child_parent_at_slot[didx[i]]))

    # Write newborn data
    new_arrays = new_arrays._replace(
        alive=jnp.where(newborn_at, True, new_arrays.alive),
        age=jnp.where(newborn_at, 0, new_arrays.age),
        resources=jnp.where(newborn_at, child_res_at_slot, new_arrays.resources),
        skill=jnp.where(newborn_at, child_skill_at_slot, new_arrays.skill),
        effective_skill=jnp.where(newborn_at, child_skill_at_slot, new_arrays.effective_skill),
        risk_tolerance=jnp.where(newborn_at, child_risk_at_slot, new_arrays.risk_tolerance),
        tag=jnp.where(newborn_at, child_tag_at_slot, new_arrays.tag),
        labor_remaining=jnp.where(newborn_at, 1.0, new_arrays.labor_remaining),
        sector=jnp.where(newborn_at, SECTOR_NECESSITIES, new_arrays.sector),
        sector_experience=jnp.where(newborn_at[:, None], 0.0, new_arrays.sector_experience),
        production_experience=jnp.where(newborn_at, 0.0, new_arrays.production_experience),
        goods=jnp.where(newborn_at, 0.0, new_arrays.goods),
        goods_quality=jnp.where(newborn_at, 1.0, new_arrays.goods_quality),
        tools=jnp.where(newborn_at, 0.0, new_arrays.tools),
        tools_quality=jnp.where(newborn_at, 0.0, new_arrays.tools_quality),
        employer_id=jnp.where(newborn_at, -1, new_arrays.employer_id),
        is_employer=jnp.where(newborn_at, False, new_arrays.is_employer),
        debt=jnp.where(newborn_at, 0.0, new_arrays.debt),
        credit_score=jnp.where(newborn_at, 0.5, new_arrays.credit_score),
        crisis_ticks=jnp.where(newborn_at, 0, new_arrays.crisis_ticks),
        coercion_debuff_ticks=jnp.where(newborn_at, 0, new_arrays.coercion_debuff_ticks),
        reproduction_cooldown=jnp.where(newborn_at, cooldown_ticks, new_arrays.reproduction_cooldown),
        reproduction_penalty=jnp.where(newborn_at, 0.0, new_arrays.reproduction_penalty),
        price_belief=jnp.where(newborn_at, 1.0, new_arrays.price_belief),
        parent_id=jnp.where(newborn_at, child_parent_at_slot, new_arrays.parent_id),
    )

    new_arrays = new_arrays._replace(
        reproduction_cooldown=jnp.maximum(0, new_arrays.reproduction_cooldown - 1),
    )

    num_births_actual = jnp.sum(has_birth).astype(jnp.int32)
    num_aging_deaths = jnp.sum(aging_death).astype(jnp.int32)

    new_key = jax.random.fold_in(key, 3)
    return new_arrays, new_key, num_births_actual, num_aging_deaths


# ============================================================
# Employer Status
# ============================================================

def jax_employer_status(arrays: AgentArrays, cfg: dict) -> AgentArrays:
    n = arrays.alive.shape[0]
    alive_count = jnp.sum(arrays.alive).astype(jnp.float32)
    employer_ratio = cfg.get('employer_ratio', _CFG_DEFAULTS['employer_ratio'])
    n_employers = jnp.maximum(1, (alive_count * employer_ratio).astype(jnp.int32))

    # Employers = capital holders (by tools quantity; quality determines productivity but not identity)
    alive_tools = jnp.where(arrays.alive, arrays.tools, -jnp.inf)
    sorted_tools = jnp.sort(alive_tools)
    threshold_idx = jnp.maximum(0, n - n_employers)
    dynamic_threshold = sorted_tools[threshold_idx]

    is_employer = arrays.alive & (arrays.tools >= dynamic_threshold)
    return arrays._replace(
        is_employer=is_employer,
        employer_id=jnp.where(is_employer, arrays.employer_id, -1),
    )


# ============================================================
# Coercion Channel (ML-11) -- pure computation, no int()
# ============================================================

def jax_coercion_core(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    enabled = config.get('coercion_enabled', False)
    subsistence = config.get('subsistence_cost', 0.9)
    n = arrays.alive.shape[0]

    # Tool contribution: logarithmic decay, no hard cap
    tool_contrib = jnp.log1p(arrays.tools / config.get('tool_saturation_K', 10.0)) * 2.0
    my_power = (arrays.resources * 0.3 + tool_contrib
                + arrays.effective_skill * 5.0)

    social_defense = jnp.sum(jnp.maximum(0, arrays.reciprocity_ledger) * 0.05, axis=0)
    their_power = my_power + social_defense

    success_prob = jax.nn.sigmoid((my_power[:, None] - their_power[None, :]) * 0.3)

    eye = jnp.eye(n, dtype=jnp.bool_)
    alive_2d = arrays.alive[:, None] & arrays.alive[None, :]
    success_prob = jnp.where(eye | ~alive_2d, 0.0, success_prob)

    # Attack condition: resources > 2x survival line (not starving), and power > target
    can_attack = arrays.alive & (arrays.resources > subsistence * 2) & enabled
    best_target = jnp.argmax(success_prob, axis=1)
    best_prob = success_prob[jnp.arange(n), best_target]
    target_power = their_power[best_target]

    attacking = can_attack & (best_prob > 0.5) & (my_power > target_power)

    safe_target = jnp.where(attacking, best_target, 0).astype(jnp.int32)
    target_valid = attacking

    # Transfer a fixed 10% of target's resources
    requested_transfer = jnp.where(
        target_valid, arrays.resources[safe_target] * 0.1, 0.0,
    )

    # Clamp total transfer to each target to not exceed 10%
    raw_lost = jnp.zeros(n, dtype=jnp.float32).at[safe_target].add(requested_transfer)
    max_loss = arrays.resources * 0.1
    actual_lost = jnp.minimum(raw_lost, jnp.maximum(0, max_loss))
    safe_raw = jnp.maximum(raw_lost, 1e-8)
    loss_ratio = jnp.where(raw_lost > 0, actual_lost[safe_target] / safe_raw[safe_target], 0.0)
    actual_transfer = requested_transfer * loss_ratio

    gained = actual_transfer
    lost = jnp.zeros(n, dtype=jnp.float32).at[safe_target].add(
        jnp.where(target_valid, actual_transfer, 0.0)
    )

    new_resources = arrays.resources + gained - lost

    was_attacked = jnp.zeros(n, dtype=jnp.bool_).at[safe_target].set(target_valid)
    new_debuff = jnp.where(was_attacked & arrays.alive, 2, arrays.coercion_debuff_ticks)
    new_debuff = jnp.where(arrays.alive, jnp.maximum(0, new_debuff - 1), 0)

    new_ledger = arrays.reciprocity_ledger
    attacker_rows = jnp.where(target_valid, jnp.arange(n), 0)
    new_ledger = new_ledger.at[attacker_rows, safe_target].add(-actual_transfer)
    new_ledger = new_ledger.at[safe_target, attacker_rows].add(actual_transfer)

    new_key = jax.random.fold_in(key, 101)
    cross_tag_coerce = jnp.sum(
        (arrays.tag != arrays.tag[safe_target]) & target_valid
    )
    return (
        arrays._replace(
            resources=new_resources, coercion_debuff_ticks=new_debuff,
            reciprocity_ledger=new_ledger,
        ),
        new_key,
        jnp.sum(attacking),
        jnp.sum(target_valid),
        cross_tag_coerce,
    )


# ============================================================
# Goods Market -- single commodity, fully vectorized
# ============================================================

def jax_goods_market_core(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    max_k = 10  # fixed constant
    current_tick = config.get('current_tick', 0)
    subsistence = config.get('subsistence_cost', 0.9)
    n = arrays.alive.shape[0]

    alive_count = jnp.maximum(1, jnp.sum(arrays.alive))
    k_dyn = jnp.clip(alive_count // 10, 5, max_k).astype(jnp.int32)

    k1 = jax.random.fold_in(key, 200)
    partners = jax.random.randint(k1, (n, max_k), 0, n)
    k_range = jnp.arange(max_k)
    partners = jnp.where(k_range[None, :] < k_dyn, partners, jnp.arange(n)[:, None])

    k2, k3 = jax.random.split(jax.random.fold_in(key, 201))
    explore_rolls = jax.random.uniform(k2, (n, max_k))
    sell_noise = jax.random.uniform(k3, (n, max_k))

    partner_price_belief = arrays.price_belief[partners]

    partner_goods = arrays.goods[partners]
    partner_alive = arrays.alive[partners]
    same_tag = (arrays.tag[:, None] == arrays.tag[partners]).astype(jnp.float32)
    ledger_bal = arrays.reciprocity_ledger[jnp.arange(n)[:, None], partners]
    self_mask = (partners == jnp.arange(n)[:, None])

    # Family trust bonus
    is_family = ((arrays.parent_id[:, None] == partners) |
                 (arrays.parent_id[partners] == jnp.arange(n)[:, None]))
    family_bonus = is_family.astype(jnp.float32) * 0.8

    trust = jax.nn.sigmoid(ledger_bal * 0.1 + same_tag * 0.5 + family_bonus - 0.5)

    # --- Quality in pricing ---
    # Market average quality (lemon market's "average quality" expectation)
    avg_market_gq = jnp.sum(jnp.where(arrays.alive, arrays.goods_quality, 0.0)) / alive_count

    # Seller ask: quality premium -- high quality goods sell for more
    sell_signal = (arrays.resources - subsistence * 2) / (subsistence * 5)
    sell_ask = (arrays.price_belief[:, None]
                * jnp.maximum(arrays.goods_quality[:, None], 1e-8)
                * (1.0 - 0.1 * jnp.clip(sell_signal[:, None], -1, 1)))

    # Buyer bid: information asymmetry -- trust determines how much real quality is known
    # trust=1 -> knows real quality; trust=0 -> uses market average estimate
    partner_gq = arrays.goods_quality[partners]
    estimated_gq = trust * partner_gq + (1.0 - trust) * avg_market_gq
    buy_urgency = 1.0 / (1.0 + arrays.goods * arrays.goods_quality / subsistence)
    buy_bid = (partner_price_belief
               * jnp.maximum(estimated_gq, 1e-8)
               * (1.0 + 0.2 * buy_urgency[:, None]))

    # Price matching: seller ask <= buyer bid -> trade window exists
    price_match = (sell_ask <= buy_bid)

    # Sell quantity: more when resources are abundant
    base_sell = jax.nn.sigmoid(sell_signal + arrays.risk_tolerance * 0.3)
    sell_ratio = (jnp.clip(base_sell, 0.05, 0.95) / max_k)[:, None]
    sell_ratio = jnp.where(explore_rolls < 0.05, sell_noise * 0.3 / max_k, sell_ratio)
    sell_qty = arrays.goods[:, None] * sell_ratio

    # Buy quantity: continuous demand driven, buy_urgency controls purchase volume
    buy_qty = jnp.where(
        (trust > 0.4) & partner_alive & (partner_goods > 0) & price_match,
        jnp.minimum(partner_goods * buy_urgency[:, None] * trust * 0.3,
                     arrays.resources[:, None] * buy_urgency[:, None] * trust * 0.05),
        0.0,
    )

    trade_qty = jnp.minimum(sell_qty, buy_qty)
    trade_qty = jnp.where(
        (trade_qty > 0) & partner_alive & (arrays.goods[:, None] > 0) & (partner_goods > 0),
        trade_qty, 0.0,
    )

    # Trade price = midpoint
    trade_price = jnp.where(trade_qty > 0, (sell_ask + buy_bid) / 2, 0.0)
    trade_value = trade_qty * trade_price
    trade_qty = jnp.where(self_mask, 0.0, trade_qty)
    trade_value = jnp.where(self_mask, 0.0, trade_value)

    total_sold_qty = jnp.sum(trade_qty, axis=1)
    total_sell_value = jnp.sum(trade_value, axis=1)

    flat_partners = partners.reshape(-1)
    flat_qty = trade_qty.reshape(-1)
    flat_value = trade_value.reshape(-1)
    flat_partner_alive = partner_alive.reshape(-1)
    flat_valid = (flat_qty > 0) & flat_partner_alive
    total_bought = jnp.zeros(n).at[flat_partners].add(jnp.where(flat_valid, flat_qty, 0.0))
    total_buy_cost = jnp.zeros(n).at[flat_partners].add(jnp.where(flat_valid, flat_value, 0.0))

    # goods_quality merge: bought goods carry partner's quality
    partner_goods_quality_flat = arrays.goods_quality[flat_partners]
    total_bought_quality_val = jnp.zeros(n).at[flat_partners].add(
        jnp.where(flat_valid, flat_qty * partner_goods_quality_flat, 0.0))

    new_resources = arrays.resources + total_sell_value - total_buy_cost
    new_goods = arrays.goods - total_sold_qty + total_bought

    # merge quality: effective value conservation
    new_goods_quality = jnp.where(
        new_goods > 1e-8,
        (arrays.goods * arrays.goods_quality - total_sold_qty * arrays.goods_quality
         + total_bought_quality_val) / new_goods,
        arrays.goods_quality,
    )

    new_resources = jnp.where(arrays.alive, new_resources, arrays.resources)
    new_goods = jnp.where(arrays.alive, new_goods, arrays.goods)
    new_goods_quality = jnp.where(arrays.alive, new_goods_quality, arrays.goods_quality)

    # Adaptive price adjustment: supply-demand driven
    any_trade_per_partner = trade_qty > 0
    any_trade = jnp.any(any_trade_per_partner, axis=1)
    total_deal_qty = jnp.maximum(0.001, jnp.sum(trade_qty, axis=1))
    total_deal_value = jnp.sum(trade_value, axis=1)
    avg_deal_price = total_deal_value / total_deal_qty

    want_buy = (buy_urgency > 0.3) & ~any_trade & jnp.any(
        partner_alive & (partner_goods > 0), axis=1)
    want_sell = (sell_signal > 0) & ~any_trade & (arrays.goods > 0)

    new_price = arrays.price_belief
    new_price = jnp.where(want_buy, new_price * 1.02, new_price)
    new_price = jnp.where(want_sell, new_price * 0.98, new_price)
    new_price = jnp.where(any_trade,
                          0.8 * arrays.price_belief + 0.2 * avg_deal_price,
                          new_price)
    new_price = jnp.clip(new_price, 0.1, 10.0)
    new_price = jnp.where(arrays.alive, new_price, arrays.price_belief)

    rows_flat = jnp.repeat(jnp.arange(n), max_k)
    new_ledger = arrays.reciprocity_ledger
    new_ledger = new_ledger.at[rows_flat, flat_partners].add(
        jnp.where(flat_valid, flat_value, 0.0)
    )
    new_ledger = new_ledger.at[flat_partners, rows_flat].add(
        jnp.where(flat_valid, flat_value, 0.0)
    )

    tick_val = jnp.int32(current_tick)
    traded_flat = flat_qty > 0
    not_self = flat_partners != rows_flat
    valid_tick = traded_flat & not_self & flat_partner_alive
    new_last_tick = arrays.last_interaction_tick.at[rows_flat, flat_partners].set(
        jnp.where(valid_tick, tick_val, arrays.last_interaction_tick[rows_flat, flat_partners])
    )

    new_key = jax.random.fold_in(key, 202)

    ingroup_count = jnp.sum((same_tag > 0.5) & (trade_qty > 0))
    total_count = jnp.sum(trade_qty > 0)

    # Adverse selection metric: share of low-quality (gq < avg) goods in total trade
    low_quality_trade = jnp.sum(trade_qty * (partner_gq < avg_market_gq).astype(jnp.float32))
    total_trade_qty = jnp.maximum(jnp.sum(trade_qty), 1e-8)
    adverse_selection = low_quality_trade / total_trade_qty

    # Quality premium metric: high-quality trade price / low-quality trade price
    high_q_trade_value = jnp.sum(trade_value * (partner_gq >= avg_market_gq).astype(jnp.float32))
    high_q_trade_qty = jnp.maximum(jnp.sum(trade_qty * (partner_gq >= avg_market_gq).astype(jnp.float32)), 1e-8)
    low_q_trade_value = jnp.sum(trade_value * (partner_gq < avg_market_gq).astype(jnp.float32))
    low_q_trade_qty = jnp.maximum(jnp.sum(trade_qty * (partner_gq < avg_market_gq).astype(jnp.float32)), 1e-8)
    quality_premium = (high_q_trade_value / high_q_trade_qty) / jnp.maximum(low_q_trade_value / low_q_trade_qty, 1e-8)

    # --- ML-10: Lending -- social mobility driven ---
    # Borrowers: non-employers who want to become employers (demand = employer threshold - current tools)
    # Lenders: employers with surplus (supply = resources above survival line)
    lending_enabled = config.get('lending_enabled', True)

    if lending_enabled:
        # Compute employer threshold (same logic as jax_employer_status)
        alive_count = jnp.sum(arrays.alive).astype(jnp.float32)
        employer_ratio = config.get('employer_ratio', 0.20)
        n_employers = jnp.maximum(1, (alive_count * employer_ratio).astype(jnp.int32))
        alive_tools = jnp.where(arrays.alive, arrays.tools, -jnp.inf)
        sorted_tools = jnp.sort(alive_tools)
        threshold_idx = jnp.maximum(0, n - n_employers)
        dynamic_threshold = sorted_tools[threshold_idx]

        # Borrowers: non-employers, debt doesn't exceed tools (collateral), room for growth
        borrower_ok = arrays.alive & ~arrays.is_employer & \
                      (arrays.debt < arrays.tools) & \
                      (arrays.tools < dynamic_threshold)
        # Lenders: employers with surplus
        lender_ok = arrays.alive & arrays.is_employer & (new_resources > subsistence * 3)

        partner_borrower = borrower_ok[partners]
        can_lend = lender_ok[:, None] & partner_borrower & (trust > 0.4) & ~self_mask

        # Lending amount = min(lender surplus * lending_rate, borrower gap to threshold)
        lending_rate = config.get('lending_surplus_rate', 0.03)
        borrower_gap = jnp.maximum(0, dynamic_threshold - arrays.tools)
        lend_amount = jnp.minimum(
            new_resources[:, None] * lending_rate,
            borrower_gap[partners],
        )
        lend_amount = jnp.where(can_lend, lend_amount, 0.0)
        total_lent_out = jnp.sum(lend_amount, axis=1)

        flat_lend = lend_amount.reshape(-1)
        flat_lend_valid = flat_lend > 0
        total_borrowed = jnp.zeros(n).at[flat_partners].add(
            jnp.where(flat_lend_valid, flat_lend, 0.0)
        )

        new_resources = new_resources - total_lent_out + total_borrowed
        new_debt = arrays.debt + total_borrowed

        new_ledger = new_ledger.at[rows_flat, flat_partners].add(
            jnp.where(flat_lend_valid, -flat_lend, 0.0)
        )
        new_ledger = new_ledger.at[flat_partners, rows_flat].add(
            jnp.where(flat_lend_valid, flat_lend, 0.0)
        )

        total_lending = jnp.sum(lend_amount)
    else:
        new_debt = arrays.debt
        total_lending = jnp.float32(0.0)

    return (
        arrays._replace(
            resources=new_resources, goods=new_goods, goods_quality=new_goods_quality,
            price_belief=new_price,
            reciprocity_ledger=new_ledger, last_interaction_tick=new_last_tick,
            debt=new_debt,
        ),
        new_key,
        jnp.sum(trade_qty > 0),
        ingroup_count,
        total_count,
        total_lending,
        adverse_selection,
        quality_premium,
    )


# ============================================================
# ML-10: 债务偿还
# ============================================================

def jax_debt_repayment(arrays: AgentArrays, config: dict) -> tuple:
    """Per-tick interest accumulation + automatic principal repayment + collateral default writeoff

    Design: lending is resource redistribution (borrower->lender), does not destroy resources.
    - Interest = depreciation rate * leverage ratio (low leverage low interest, high leverage high interest)
    - Principal repayment = min(debt*0.1, res*dsr) (automatic amortization)
    - Total repayment distributed proportionally to employers (lender returns)
    - Default: debt > tools * 3 -> write off (insufficient collateral)
    """

    # Interest = depreciation rate * leverage ratio
    leverage = arrays.debt / (arrays.debt + arrays.tools + 1e-8)
    interest_per_agent = config.get('depreciation_rate', 0.02) * leverage
    new_debt = arrays.debt * (1 + interest_per_agent)

    # Automatic principal repayment
    repay = jnp.minimum(new_debt * 0.1, arrays.resources * config.get('debt_service_ratio', 0.05))
    repay = jnp.clip(repay, 0.0, new_debt)
    repay = jnp.where(arrays.alive & (new_debt > 0), repay, 0.0)

    # Default writeoff: debt > tools * 3
    max_debt = arrays.tools * 3.0
    defaulted = arrays.alive & (new_debt > max_debt) & (arrays.debt > 0)
    new_debt = jnp.where(defaulted, 0.0, new_debt - repay)
    repay = jnp.where(defaulted, 0.0, repay)

    # Repayment resources first deducted from borrowers, then distributed to employers (lender returns)
    new_resources = arrays.resources - repay

    total_repaid = jnp.sum(repay)
    employers = arrays.is_employer & arrays.alive
    n_employers = jnp.maximum(1, jnp.sum(employers))
    per_employer = total_repaid / n_employers
    new_resources = new_resources + jnp.where(employers, per_employer, 0.0)

    total_default = jnp.sum(jnp.where(defaulted, arrays.debt, 0.0))

    return (
        arrays._replace(resources=new_resources, debt=new_debt),
        total_repaid,
        total_default,
    )


# ============================================================
# Labor Market
# ============================================================

def jax_labor_market_core(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    max_k = 10  # fixed constant
    employment_enabled = config.get('employment_enabled', True)
    subsistence_cost = config.get('subsistence_cost', 0.9)
    n = arrays.alive.shape[0]

    # Dynamic k: scales with alive population
    alive_count = jnp.maximum(1, jnp.sum(arrays.alive))
    k_dyn = jnp.clip(alive_count // 10, 5, max_k).astype(jnp.int32)

    workers = arrays.alive & ~arrays.is_employer & (arrays.employer_id < 0)
    employers = arrays.alive & arrays.is_employer
    has_workers = jnp.sum(workers) > 0
    has_employers = jnp.sum(employers) > 0
    can_market = has_workers & has_employers & employment_enabled

    k1 = jax.random.fold_in(key, 300)
    employer_cands = jax.random.randint(k1, (n, max_k), 0, n)
    # Positions beyond dynamic k set to self (filtered out by cand_not_self)
    k_range = jnp.arange(max_k)
    employer_cands = jnp.where(k_range[None, :] < k_dyn, employer_cands, jnp.arange(n)[:, None])

    k2 = jax.random.fold_in(key, 301)
    wages = jax.random.uniform(k2, (n, max_k), minval=subsistence_cost * 0.5, maxval=subsistence_cost * 3)

    employer_res = arrays.resources[employer_cands]
    cand_alive = arrays.alive[employer_cands]
    cand_not_self = employer_cands != jnp.arange(n)[:, None]
    can_afford = employer_res > subsistence_cost * 3

    # Physical capacity filter: work_capacity determines employability (continuous)
    work_cap = jax_age_capacity(arrays, config)
    k_cap = jax.random.fold_in(key, 303)
    cap_roll = jax.random.uniform(k_cap, (n,))
    can_work = cap_roll < work_cap

    # ML-2b employment voluntariness: compare wage income vs self-production income
    # Self-production income estimate (consistent with jax_sector_production but without noise)
    production_scale = config.get('production_scale', 3.0)
    self_prod_value = (arrays.effective_skill * arrays.labor_remaining
                       * _effective_tool_bonus(arrays.tools, config) * production_scale * 0.3)
    desperation = jnp.maximum(0, 1.0 - arrays.resources / (subsistence_cost * 5))
    effective_wage = wages * (1 + desperation[:, None] * 0.5)
    accept_wage = effective_wage > self_prod_value[:, None]

    match = workers[:, None] & cand_alive & cand_not_self & can_afford & accept_wage & can_work[:, None]
    first_idx = jnp.argmax(match.astype(jnp.int32), axis=1)
    has_match = jnp.any(match, axis=1)

    matched_wage = wages[jnp.arange(n), first_idx]
    matched_emp = employer_cands[jnp.arange(n), first_idx]

    do_match = workers & has_match & can_market

    new_resources = jnp.where(do_match, arrays.resources + matched_wage, arrays.resources)
    new_labor = jnp.where(do_match, arrays.labor_remaining * 0.2, arrays.labor_remaining)

    new_key = jax.random.fold_in(key, 302)

    return (
        arrays._replace(
            resources=new_resources,
            employer_id=jnp.where(do_match, matched_emp.astype(jnp.int32), arrays.employer_id),
            labor_remaining=new_labor,
        ),
        new_key,
        jnp.sum(do_match),
    )


# ============================================================
# Employer Production
# ============================================================

def jax_employer_production(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    n = arrays.alive.shape[0]
    is_employee = arrays.alive & (arrays.employer_id >= 0) & ~arrays.is_employer
    emp_labor = arrays.labor_remaining * 0.8 * jax_age_capacity(arrays, config)

    safe_emp_id = jnp.clip(arrays.employer_id, 0, n - 1)
    employer_tools = jnp.where(arrays.employer_id >= 0, arrays.tools[safe_emp_id], 0.0)
    employer_tools_quality = jnp.where(arrays.employer_id >= 0, arrays.tools_quality[safe_emp_id], 0.0)

    production_scale = config.get('production_scale', 3.0)
    q_rate = config.get('quality_improvement_rate', 0.1)
    noise = jax.random.normal(key, (n,)) * 0.1
    emp_eff_tool_b = _effective_tool_bonus(employer_tools, config)

    # --- Employee labor split: geometric mean (same as above) ---
    emp_can_improve = employer_tools_quality < arrays.effective_skill
    emp_quality_share = jnp.where(
        emp_can_improve,
        jnp.sqrt(
            employer_tools * jnp.maximum(arrays.effective_skill - employer_tools_quality, 0.0)
            / (arrays.effective_skill * (employer_tools + employer_tools_quality) + 1e-8)
        ),
        0.0,
    )
    emp_production_labor = emp_labor * (1.0 - emp_quality_share)
    emp_quality_labor = emp_labor * emp_quality_share

    # Goods production
    output = jnp.maximum(0, arrays.effective_skill * emp_eff_tool_b * production_scale * emp_production_labor * (1 + noise))

    # Quality investment: employee labor improves employer tool quality (sqrt(gap) avoids quadratic gap effect)
    quality_gain = emp_quality_labor * arrays.effective_skill * q_rate * jnp.sqrt(jnp.maximum(arrays.effective_skill - employer_tools_quality, 0.0))
    quality_gain = jnp.maximum(0.0, quality_gain)

    # Aggregate quality improvement per employer
    quality_gain_per_employer = jnp.zeros(n, dtype=jnp.float32).at[safe_emp_id].add(
        jnp.where(is_employee, quality_gain, 0.0),
    )

    new_emp_tq = employer_tools_quality + quality_gain
    output_quality = jnp.sqrt(arrays.effective_skill * jnp.maximum(new_emp_tq, 1e-8))

    output_per_employer = jnp.zeros(n, dtype=jnp.float32).at[safe_emp_id].add(
        jnp.where(is_employee, output, 0.0),
    )
    output_quality_per_employer = jnp.zeros(n, dtype=jnp.float32).at[safe_emp_id].add(
        jnp.where(is_employee, output * output_quality, 0.0),
    )

    # Employer receives goods + goods_quality merge
    is_emp = arrays.is_employer
    new_goods = arrays.goods + jnp.where(is_emp, output_per_employer, 0.0)
    emp_goods_quality_val = arrays.goods * arrays.goods_quality + jnp.where(is_emp, output_quality_per_employer, 0.0)
    new_goods_quality = jnp.where(
        new_goods > 1e-8,
        emp_goods_quality_val / new_goods,
        arrays.goods_quality,
    )

    # Employer tools_quality update (single channel: quality investment)
    new_tools_quality = jnp.where(
        is_emp,
        jnp.minimum(arrays.tools_quality + quality_gain_per_employer, arrays.skill),
        arrays.tools_quality,
    )

    new_key = jax.random.fold_in(key, 400)
    return (
        arrays._replace(
            goods=new_goods, goods_quality=new_goods_quality,
            tools_quality=new_tools_quality,
            labor_remaining=jnp.where(is_employee, arrays.labor_remaining - emp_labor, arrays.labor_remaining),
        ),
        new_key,
    )


# ============================================================
# JIT Core -- Pure JAX Computation Layer
# ============================================================

# config defaults
_CFG_DEFAULTS = {
    # Objective physical parameters
    'subsistence_cost': 0.9,
    'natural_inflow_per_capita': 0.72,
    'resource_ceiling': 5000.0,
    'depreciation_rate': 0.02,
    'production_scale': 3.0,
    'tool_saturation_K': 10.0,
    'quality_improvement_rate': 0.1,
    'population_ceiling': 500,
    'interest_rate': 0.001,
    'debt_service_ratio': 0.05,
    'goods_q_decay': 0.005,
    'tools_q_decay': 0.0005,

    # Human physiological parameters
    'aging_enabled': True,
    'aging_base_rate': 0.0001,
    'aging_rate_multiplier': 0.0005,
    'adult_age': 936,
    'elder_age': 3120,
    'reproduction_base_prob': 0.01,
    'reproduction_cooldown': 20,
    'inheritance_ratio': 0.8,
    'mutation_strength': 0.15,

    # Institutional switches
    'demographic_enabled': False,
    'coercion_enabled': False,
    'employment_enabled': True,
    'employer_ratio': 0.20,
    'lending_enabled': True,
    'lending_surplus_rate': 0.03,
    'num_tags': 0,
}

# config key list (stable ordering)
_CFG_ORDER = tuple(_CFG_DEFAULTS.keys())


def _cfg(config: dict, key: str):
    """Get value from dict, use default if missing"""
    return config.get(key, _CFG_DEFAULTS[key])


@functools.partial(jax.jit, static_argnums=tuple(range(3, 3 + len(_CFG_ORDER))))
def jax_step_core(arrays, key, current_tick, *static_params) -> tuple:
    """Pure JAX single step (@jax.jit compiled)

    arrays, key, current_tick: JAX dynamic inputs
    static_params: compile-time constants, order matches _CFG_ORDER
    """
    cfg = dict(zip(_CFG_ORDER, static_params))
    cfg['current_tick'] = current_tick

    # 1. Production
    arrays = jax_enforce_labor(arrays, cfg)
    arrays, key = jax_sector_production(arrays, key, cfg)

    # 2. Natural inflow -> self-sufficiency (emergency fallback) -> survival deduction
    arrays, key = jax_enforce_conservation(arrays, key, cfg)
    arrays = jax_self_sufficiency(arrays, cfg)     # Only supplements when inflow is insufficient
    arrays = jax_enforce_survival(arrays, cfg)

    # 3. Depreciation (physical fact, before economic decisions)
    arrays = jax_depreciation(arrays, cfg)

    # 3. Coercion O(N^2)
    arrays, key, c_attempt, c_success, c_cross = jax_coercion_core(arrays, key, cfg)

    # 4. Employer status
    arrays = jax_employer_status(arrays, cfg)

    # 5. Labor market O(NK) -- only when employment is enabled
    arrays, key, l_hired = jax_labor_market_core(arrays, key, cfg)

    # 6. Employer production -- only when employment is enabled
    arrays, key = jax_employer_production(arrays, key, cfg)

    # 7. Goods market O(NK) + lending
    arrays, key, t_total, t_ingroup, t_count, t_lending, adverse_sel, q_premium = jax_goods_market_core(arrays, key, cfg)

    # 8. Investment (after lending, so loaned resources can be invested)
    arrays = jax_investment(arrays, cfg)

    # 9. Debt repayment + interest + default writeoff
    arrays, t_repaid, t_default = jax_debt_repayment(arrays, cfg)

    # Demographic dynamics (ML-9) moved to after institutional rules (called in jax_simulation.run)

    return arrays, key, (c_attempt, c_success, c_cross), (t_total, t_ingroup, t_count), l_hired, (t_lending, t_repaid, t_default), (adverse_sel, q_premium)


# ============================================================
# Demographic Dynamics -- called independently (after institutional rules)
# ============================================================

_DEMO_CFG_ORDER = (
    'subsistence_cost',
    'population_ceiling', 'reproduction_base_prob',
    'reproduction_cooldown',
    'inheritance_ratio', 'mutation_strength',
    'aging_enabled', 'aging_base_rate', 'aging_rate_multiplier',
    'adult_age', 'elder_age',
)


@functools.partial(jax.jit, static_argnums=tuple(range(2, 2 + len(_DEMO_CFG_ORDER))))
def jax_step_demographic_core(arrays, key, *static_params):
    """ML-9 demographic dynamics -- pure JAX, independent JIT compilation"""
    cfg = dict(zip(_DEMO_CFG_ORDER, static_params))
    arrays, key, num_births, num_aging_deaths = jax_demographic(arrays, key, cfg)
    return arrays, key, num_births, num_aging_deaths


def jax_step_demographic(arrays: AgentArrays, key: jax.Array, config: dict) -> tuple:
    """ML-9 demographic dynamics entry point -- called after institutional rules"""
    static_params = tuple(_cfg(config, k) for k in _DEMO_CFG_ORDER)
    arrays, key, num_births, num_aging_deaths = jax_step_demographic_core(arrays, key, *static_params)
    return arrays, key, int(num_births), int(num_aging_deaths)


# ============================================================
# Outer Wrapper -- Python Statistics
# ============================================================

def jax_step_on(arrays: AgentArrays, key: jax.Array, config: dict, tick: int = 0) -> tuple:
    """Outer wrapper: call JIT core, handle Python-side statistics"""
    static_params = tuple(_cfg(config, k) for k in _CFG_ORDER)
    arrays, key, coercion_stats, trade_stats, labor_hired, lending_stats, market_stats = jax_step_core(arrays, key, jnp.int32(tick), *static_params)

    c_attempt, c_success, c_cross = coercion_stats
    t_total, t_ingroup, t_count = trade_stats
    t_lending, t_repaid, t_default = lending_stats
    adverse_sel, q_premium = market_stats

    c_cross_int = int(c_cross)
    c_success_int = int(c_success)
    t_ingroup_val = float(t_ingroup) / max(1, float(t_count))
    t_total_val = int(t_total) // 2

    alive = arrays.alive
    alive_count = jnp.maximum(1, jnp.sum(alive))

    stats = {
        'coercion_attempts': int(c_attempt),
        'coercion_successes': c_success_int,
        'cross_tag_coercion': c_cross_int,
        'cross_tag_coercion_ratio': c_cross_int / max(1, c_success_int),
        'total_trades': t_total_val,
        'ingroup_trade_ratio': t_ingroup_val,
        'labor_hired': int(labor_hired),
        'total_lending': float(t_lending),
        'total_repayment': float(t_repaid),
        'total_default': float(t_default),
        'avg_debt': float(jnp.sum(jnp.where(alive, arrays.debt, 0.0)) / alive_count),
        'agents_in_debt': int(jnp.sum((arrays.debt > 0) & alive)),
        'adverse_selection': float(adverse_sel),
        'quality_premium': float(q_premium),
    }

    return arrays, key, stats
