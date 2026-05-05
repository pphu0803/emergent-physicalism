"""
JAX Accelerated Simulator

Same external interface as Simulation, internally uses JAX pure function stepping.
Used as a drop-in replacement for Simulation for large-scale experiments.

Execution order (per tick):
1. jax_step_on: meta-laws + economic activity (excluding demographic dynamics)
2. apply_rules_jax: institutional rules
3. jax_step_demographic: demographic dynamics (ML-9, after institutional rules)
"""

import time
from typing import Dict, Any, Optional, Callable, List

import jax
import jax.numpy as jnp

from .state import AgentArrays, init_agents, compute_state_from_arrays, DEFAULT_MAX_AGENTS
from .jax_steps import jax_step_on, jax_step_demographic
from rules.jax_rules import apply_rules_jax, reset_command_state
from analysis.runtime_metrics import compute_metrics


class JaxSimulation:
    """JAX accelerated simulator -- same external interface as Simulation"""

    def __init__(
        self,
        num_agents: int = 100,
        initial_endowment: float = 10.0,
        meta_law_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        rules: Optional[list] = None,
        max_agents: int = DEFAULT_MAX_AGENTS,
        enable_metrics: bool = False,
        initial_age_range: tuple = (0, 0),
    ):
        self._meta_config = meta_law_config or {}
        self.rules = rules or []

        # Derive ML-11/ML-2b switches from institutional rules
        rule_names = {r.name() for r in self.rules}
        derived_ml = dict(self._meta_config)

        # ML-11: Institutional rules take priority -- FreeCoercion enables, NoCoercion disables
        # Only fall back to meta_law_config when no relevant rule exists
        if 'coercion_free' in rule_names:
            derived_ml['coercion_enabled'] = True
        elif 'coercion_none' in rule_names:
            derived_ml['coercion_enabled'] = False
        elif 'coercion_enabled' not in derived_ml:
            derived_ml['coercion_enabled'] = False

        # ML-2b: Institutional rules take priority -- NoEmployment disables employment
        if 'employment_none' in rule_names:
            derived_ml['employment_enabled'] = False
        elif 'employment_free' in rule_names:
            derived_ml['employment_enabled'] = True
        elif 'employment_enabled' not in derived_ml:
            derived_ml['employment_enabled'] = True

        # ML-10: num_tags defaults to 0 (no tags assigned when trust channel is inactive)
        if 'num_tags' not in derived_ml:
            derived_ml['num_tags'] = 0

        self.config = {
            'num_agents': num_agents,
            'initial_endowment': initial_endowment,
            'meta_law_config': derived_ml,
            'seed': seed,
            'rules': [r.to_dict() for r in self.rules] if self.rules else [],
            'max_agents': max_agents,
            'initial_age_range': initial_age_range,
        }
        self._arrays: Optional[AgentArrays] = None
        self._key = None
        self.state_history: List[Dict[str, Any]] = []
        self.death_log: List[Dict[str, Any]] = []
        self.enable_metrics = enable_metrics
        self.metrics_history: List[Dict[str, Any]] = []

    def initialize(self) -> 'JaxSimulation':
        """Initialize JAX state"""
        ml = self.config['meta_law_config']
        num_tags = ml.get('num_tags', 0)

        self._arrays, self._key = init_agents(
            self.config['num_agents'],
            seed=self.config['seed'],
            initial_endowment=self.config['initial_endowment'],
            num_tags=num_tags,
            max_agents=self.config['max_agents'],
            initial_age_range=self.config.get('initial_age_range', (0, 0)),
        )
        self.state_history = []
        self.death_log = []
        self.metrics_history = []
        reset_command_state()
        return self

    def run(
        self,
        num_ticks: int = 500,
        callback: Optional[Callable] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the simulation

        Execution order (per tick):
        1. jax_step_on: meta-laws + economic activity (excluding demographic dynamics)
        2. apply_rules_jax: institutional rules
        3. jax_step_demographic: ML-9 demographic dynamics (if demographic_enabled)
        """
        if self._arrays is None:
            self.initialize()

        start_time = time.time()
        ml = self.config['meta_law_config']
        demographic_enabled = ml.get('demographic_enabled', False)

        for tick in range(num_ticks):
            # Record pre-tick state (for death detection and metrics computation)
            prev_alive = self._arrays.alive
            prev_arrays_snapshot = self._arrays if self.enable_metrics else None

            # 1. Meta-laws + economic activity (excluding demographic dynamics)
            self._arrays, self._key, stats = jax_step_on(
                self._arrays, self._key, ml, tick
            )

            # Record starvation deaths after step_on (before demographic, to avoid birth overwrite)
            starvation_deaths = int(jnp.sum(prev_alive & ~self._arrays.alive))

            # 2. Institutional rules (JAX adapted)
            if self.rules:
                self._arrays = apply_rules_jax(
                    self._arrays, self.rules, tick, stats,
                    current_market_price=ml.get('current_market_price', 1.0),
                    subsistence_cost=ml.get('subsistence_cost', 0.9),
                )

            # 3. ML-9 demographic dynamics (after institutional rules, ensuring institutional allocation affects reproduction)
            demo_births = 0
            demo_aging_deaths = 0
            if demographic_enabled:
                self._arrays, self._key, demo_births, demo_aging_deaths = jax_step_demographic(
                    self._arrays, self._key, ml
                )

            # 4. Compute state
            state = compute_state_from_arrays(self._arrays, tick, ml)

            # 5. Additional statistics
            state['coercion_attempts'] = stats.get('coercion_attempts', 0)
            state['coercion_successes'] = stats.get('coercion_successes', 0)
            state['cross_tag_coercion_ratio'] = (
                stats.get('cross_tag_coercion', 0)
                / max(1, stats.get('coercion_successes', 0))
            )
            state['ingroup_trade_ratio'] = stats.get('ingroup_trade_ratio', 0)
            state['total_trades'] = stats.get('total_trades', 0)
            state['coercion_victim_debuff_count'] = int(
                jnp.sum(self._arrays.coercion_debuff_ticks > 0)
            )
            state['total_production'] = float(
                jnp.sum(jnp.where(self._arrays.alive, self._arrays.goods, 0.0))
            )
            state['total_lending'] = stats.get('total_lending', 0)
            state['total_repayment'] = stats.get('total_repayment', 0)
            state['avg_debt'] = stats.get('avg_debt', 0)
            state['agents_in_debt'] = stats.get('agents_in_debt', 0)
            state['demo_births'] = demo_births
            state['demo_aging_deaths'] = demo_aging_deaths

            # 6. Death records
            # Actual deaths = starvation deaths (step_on) + aging deaths (demographic)
            # Cannot use prev_alive & ~new_alive because newborns fill same-tick death slots causing undercounting
            n_dead = starvation_deaths + demo_aging_deaths
            new_dead = prev_alive & ~self._arrays.alive
            if n_dead > 0:
                dead_indices = jnp.where(new_dead)[0]
                for idx in dead_indices:
                    i = int(idx)
                    self.death_log.append({
                        'tick': tick,
                        'agent_id': int(self._arrays.agent_id[i]),
                        'resources': float(self._arrays.resources[i]),
                        'skill': float(self._arrays.skill[i]),
                    })
            state['deaths_this_tick'] = n_dead

            self.state_history.append(state)

            # 7. Runtime metrics (optional)
            if self.enable_metrics:
                metrics = compute_metrics(
                    self._arrays, stats, tick,
                    prev_arrays=prev_arrays_snapshot,
                    config=ml,
                )
                self.metrics_history.append(metrics)

            if callback:
                callback(state)

            if verbose and (tick + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Tick {tick+1:4d}/{num_ticks} | "
                    f"Pop: {state['population']:3d} | "
                    f"Gini: {state.get('gini_coefficient', 0):.3f} | "
                    f"[{elapsed:.1f}s]"
                )

        if verbose:
            elapsed = time.time() - start_time
            print(f"\n  JaxSimulation complete: {num_ticks} ticks in {elapsed:.1f}s")

        return self.state_history

    def get_results(self) -> Dict[str, Any]:
        """Get structured data from simulation results -- format compatible with Simulation.get_results()"""
        if not self.state_history:
            return {}

        states = self.state_history
        final = states[-1]

        time_series = {
            'tick': [s['tick'] for s in states],
            'population': [s['population'] for s in states],
            'gini': [s.get('gini_coefficient', 0) for s in states],
            'top10_share': [s.get('top10_share', 0) for s in states],
            'bottom50_share': [s.get('bottom50_share', 0) for s in states],
            'mean_wealth': [s.get('mean_wealth', 0) for s in states],
            'median_wealth': [s.get('median_wealth', 0) for s in states],
            'num_employers': [s.get('num_employers', 0) for s in states],
            'avg_tools': [s.get('avg_tools', 0) for s in states],
            'avg_effective_skill': [s.get('avg_effective_skill', 0) for s in states],
            'avg_reciprocity_size': [s.get('avg_reciprocity_size', 0) for s in states],
            'coercion_attempts': [s.get('coercion_attempts', 0) for s in states],
            'coercion_successes': [s.get('coercion_successes', 0) for s in states],
            'ingroup_trade_ratio': [s.get('ingroup_trade_ratio', 0) for s in states],
            'cross_tag_coercion_ratio': [s.get('cross_tag_coercion_ratio', 0) for s in states],
            'coercion_victim_debuff_count': [s.get('coercion_victim_debuff_count', 0) for s in states],
            'deaths': [s.get('deaths_this_tick', 0) for s in states],
            'total_lending': [s.get('total_lending', 0) for s in states],
            'total_repayment': [s.get('total_repayment', 0) for s in states],
            'avg_debt': [s.get('avg_debt', 0) for s in states],
            'agents_in_debt': [s.get('agents_in_debt', 0) for s in states],
            'demo_births': [s.get('demo_births', 0) for s in states],
            'demo_aging_deaths': [s.get('demo_aging_deaths', 0) for s in states],
        }

        return {
            'config': self.config,
            'num_ticks': len(states),
            'final_state': {
                'population': final.get('population', 0),
                'gini': final.get('gini_coefficient', 0),
                'top10_share': final.get('top10_share', 0),
                'bottom50_share': final.get('bottom50_share', 0),
                'mean_wealth': final.get('mean_wealth', 0),
                'num_employers': final.get('num_employers', 0),
                'avg_tools': final.get('avg_tools', 0),
                'avg_reciprocity_size': final.get('avg_reciprocity_size', 0),
            },
            'time_series': time_series,
            'death_log': self.death_log,
            'metrics_history': self.metrics_history,
        }
