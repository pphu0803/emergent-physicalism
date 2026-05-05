"""
Structured Agent State Arrays (JAX pytree friendly)

Uses plain Python NamedTuple + JAX arrays, registered as JAX pytree via register_pytree_node.
Reference: Abmax (2025) fixed-size array + placeholder pattern.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Any, Tuple, Optional
import functools


# Sector encoding
SECTOR_NECESSITIES = 0
SECTOR_CAPITAL_GOODS = 1
SECTOR_LUXURIES = 2
SECTOR_NAMES = ['necessities', 'capital_goods', 'luxuries']
SECTOR_MAP = {name: i for i, name in enumerate(SECTOR_NAMES)}

# Behavior mode encoding
MODE_SURVIVAL = 0
MODE_SECURITY = 1
MODE_PROFIT = 2

# Default max agent count (reserve space for births)
DEFAULT_MAX_AGENTS = 1024


class AgentArrays(NamedTuple):
    """Structured arrays for all agent state (JAX pytree friendly)"""
    alive: jax.Array
    resources: jax.Array
    tools: jax.Array
    tools_quality: jax.Array
    skill: jax.Array
    effective_skill: jax.Array
    risk_tolerance: jax.Array
    labor_remaining: jax.Array
    tag: jax.Array
    sector: jax.Array
    sector_experience: jax.Array  # (MAX, 3)
    production_experience: jax.Array
    price_belief: jax.Array
    behavior_mode: jax.Array
    is_employer: jax.Array
    employer_id: jax.Array
    goods: jax.Array
    goods_quality: jax.Array
    reciprocity_ledger: jax.Array  # (MAX, MAX)
    last_interaction_tick: jax.Array  # (MAX, MAX)
    reproduction_cooldown: jax.Array
    reproduction_penalty: jax.Array
    coercion_debuff_ticks: jax.Array
    crisis_ticks: jax.Array
    age: jax.Array
    debt: jax.Array
    credit_score: jax.Array
    agent_id: jax.Array
    parent_id: jax.Array  # (int32) -1=no parent, >=0=parent index


# Register as JAX pytree (enables ._replace() and jit compilation)
jax.tree_util.register_pytree_node(
    AgentArrays,
    lambda xs: (tuple(xs), None),  # children=all fields, aux=None
    lambda aux, children: AgentArrays(*children),
)


def zeros(max_agents: int = DEFAULT_MAX_AGENTS) -> AgentArrays:
    """Create all-zero/default AgentArrays"""
    return AgentArrays(
        alive=jnp.zeros(max_agents, dtype=jnp.bool_),
        resources=jnp.zeros(max_agents, dtype=jnp.float32),
        tools=jnp.zeros(max_agents, dtype=jnp.float32),
        tools_quality=jnp.zeros(max_agents, dtype=jnp.float32),
        skill=jnp.zeros(max_agents, dtype=jnp.float32),
        effective_skill=jnp.zeros(max_agents, dtype=jnp.float32),
        risk_tolerance=jnp.full(max_agents, 0.5, dtype=jnp.float32),
        labor_remaining=jnp.ones(max_agents, dtype=jnp.float32),
        tag=jnp.zeros(max_agents, dtype=jnp.int32),
        sector=jnp.full(max_agents, SECTOR_NECESSITIES, dtype=jnp.int32),
        sector_experience=jnp.zeros((max_agents, 3), dtype=jnp.float32),
        production_experience=jnp.zeros(max_agents, dtype=jnp.float32),
        price_belief=jnp.ones(max_agents, dtype=jnp.float32),
        behavior_mode=jnp.zeros(max_agents, dtype=jnp.int32),
        is_employer=jnp.zeros(max_agents, dtype=jnp.bool_),
        employer_id=jnp.full(max_agents, -1, dtype=jnp.int32),
        goods=jnp.zeros(max_agents, dtype=jnp.float32),
        goods_quality=jnp.ones(max_agents, dtype=jnp.float32),
        reciprocity_ledger=jnp.zeros((max_agents, max_agents), dtype=jnp.float32),
        last_interaction_tick=jnp.full((max_agents, max_agents), -1, dtype=jnp.int32),
        reproduction_cooldown=jnp.zeros(max_agents, dtype=jnp.int32),
        reproduction_penalty=jnp.zeros(max_agents, dtype=jnp.float32),
        coercion_debuff_ticks=jnp.zeros(max_agents, dtype=jnp.int32),
        crisis_ticks=jnp.zeros(max_agents, dtype=jnp.int32),
        age=jnp.zeros(max_agents, dtype=jnp.int32),
        debt=jnp.zeros(max_agents, dtype=jnp.float32),
        credit_score=jnp.full(max_agents, 0.5, dtype=jnp.float32),
        agent_id=jnp.arange(max_agents, dtype=jnp.int32),
        parent_id=jnp.full(max_agents, -1, dtype=jnp.int32),
    )


def init_agents(
    num_agents: int,
    seed: int,
    initial_endowment: float = 10.0,
    num_tags: int = 0,
    max_agents: int = DEFAULT_MAX_AGENTS,
    initial_age_range: tuple = (0, 0),
) -> Tuple[AgentArrays, jax.Array]:
    """
    Initialize agent arrays

    Args:
        num_tags: Number of identity tags. 0=no tags (all tag=0), >0=uniformly distributed in [0, num_tags)
        initial_age_range: Initial age range (min, max), default (0, 0) means all agents age 0

    Returns:
        (AgentArrays, PRNGKey)
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    arrays = zeros(max_agents)

    alive = jnp.arange(max_agents) < num_agents

    # Initial endowment: lognormal distribution, clipped to [2, 50]
    endowments = jnp.exp(jax.random.normal(k1, (max_agents,)) * 0.3) * initial_endowment
    endowments = jnp.clip(endowments, 2.0, 50.0)

    skill = jax.random.beta(k2, 2.0, 5.0, shape=(max_agents,))
    risk_tolerance = jax.random.beta(k3, 2.0, 2.0, shape=(max_agents,))

    # Tag assignment: when num_tags=0, no tags assigned (all tag=0)
    if num_tags > 0:
        tags = jax.random.randint(k4, (max_agents,), 0, num_tags)
    else:
        tags = jnp.zeros(max_agents, dtype=jnp.int32)

    # Initial age distribution
    age_min, age_max = initial_age_range
    if age_max > age_min:
        ages = jax.random.randint(k6, (max_agents,), age_min, age_max + 1)
    else:
        ages = jnp.full(max_agents, age_min, dtype=jnp.int32)

    arrays = arrays._replace(
        alive=alive,
        resources=endowments,
        tools_quality=skill,
        skill=skill,
        effective_skill=skill,
        risk_tolerance=risk_tolerance,
        tag=tags,
        age=ages,
    )

    return arrays, k5


def compute_state_from_arrays(
    arrays: AgentArrays,
    tick: int,
    meta_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute EconomyState dict from AgentArrays (Python, not JIT)"""
    alive = arrays.alive
    n = int(jnp.sum(alive))
    if n < 2:
        return {'tick': tick, 'population': n, 'gini_coefficient': 0.0}

    wealths = jnp.where(alive, jnp.maximum(0, arrays.resources + arrays.tools), 0.0)
    total = jnp.sum(wealths)

    if total <= 0:
        return {'tick': tick, 'population': n, 'gini_coefficient': 0.0}

    sorted_w = jnp.sort(wealths)
    cumw = jnp.cumsum(sorted_w)
    gini = (n + 1 - 2 * jnp.sum(cumw) / cumw[-1]) / n
    gini = float(jnp.clip(gini, 0.0, 1.0))

    sorted_desc = jnp.sort(wealths)[::-1]
    top10_share = float(sorted_desc[:max(1, n // 10)].sum() / total)
    bottom50_share = float(sorted_desc[max(1, n // 2):].sum() / total)

    return {
        'tick': tick,
        'population': n,
        'gini_coefficient': gini,
        'top10_share': top10_share,
        'bottom50_share': bottom50_share,
        'mean_wealth': float(jnp.mean(wealths)),
        'median_wealth': float(jnp.median(wealths)),
        'num_employers': int(jnp.sum(arrays.is_employer & alive)),
        'avg_tools': float(jnp.sum(jnp.where(alive, arrays.tools, 0.0)) / n),
        'avg_effective_skill': float(jnp.sum(jnp.where(alive, arrays.effective_skill, 0.0)) / n),
        'avg_reciprocity_size': float(jnp.mean(jnp.sum(arrays.reciprocity_ledger != 0, axis=1))),
    }
