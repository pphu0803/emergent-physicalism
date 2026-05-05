"""
Simulation Runner

Manages the simulation lifecycle: initialization -> run -> result collection.
Supports parallel execution of multiple simulations for statistical robustness.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from copy import deepcopy
import json
import time

from .economy import Economy, EconomyState


class Simulation:
    """Simulation runner"""

    def __init__(
        self,
        num_agents: int = 100,
        initial_endowment: float = 10.0,
        meta_law_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        rules: Optional[list] = None,
    ):
        self.config = {
            'num_agents': num_agents,
            'initial_endowment': initial_endowment,
            'meta_law_config': meta_law_config or {},
            'seed': seed,
            'rules': [r.to_dict() for r in rules] if rules else [],
        }
        self.economy: Optional[Economy] = None
        self.rules = rules or []

    def initialize(self) -> 'Simulation':
        """Initialize the economic system"""
        self.economy = Economy(
            num_agents=self.config['num_agents'],
            initial_endowment=self.config['initial_endowment'],
            meta_law_config=self.config['meta_law_config'],
            seed=self.config['seed'],
        )
        for rule in self.rules:
            self.economy.add_rule(rule)
        return self

    def run(
        self,
        num_ticks: int = 500,
        callback: Optional[Callable[[EconomyState], None]] = None,
        verbose: bool = True,
    ) -> List[EconomyState]:
        """
        Run the simulation

        Args:
            num_ticks: Number of simulation timesteps
            callback: Per-step callback function
            verbose: Whether to print progress

        Returns:
            List of state history
        """
        if self.economy is None:
            self.initialize()

        start_time = time.time()

        for tick in range(num_ticks):
            state = self.economy.step()

            if callback:
                callback(state)

            if verbose and (tick + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Tick {tick+1:4d}/{num_ticks} | "
                    f"Pop: {state.population:3d} | "
                    f"Gini: {state.gini_coefficient:.3f} | "
                    f"Employers: {state.num_employers:3d} | "
                    f"Avg Tools: {state.avg_tools:.2f} | "
                    f"[{elapsed:.1f}s]"
                )

        if verbose:
            elapsed = time.time() - start_time
            print(f"\n  Simulation complete: {num_ticks} ticks in {elapsed:.1f}s")

        return self.economy.state_history

    def get_results(self) -> Dict[str, Any]:
        """Get structured data from simulation results"""
        if not self.economy or not self.economy.state_history:
            return {}

        states = self.economy.state_history
        return {
            'config': self.config,
            'num_ticks': len(states),
            'final_state': {
                'population': states[-1].population,
                'gini': states[-1].gini_coefficient,
                'top1_share': states[-1].top1_share,
                'top10_share': states[-1].top10_share,
                'bottom50_share': states[-1].bottom50_share,
                'mean_wealth': states[-1].mean_wealth,
                'num_employers': states[-1].num_employers,
                'hhi': states[-1].hhi,
                'total_deaths': states[-1].total_deaths,
                'total_births': states[-1].total_births,
                'profit_rate_std': states[-1].profit_rate_std,
                'sector_profit_rates': states[-1].sector_profit_rates,
                'sector_producers': states[-1].sector_producers,
            },
            'time_series': {
                'tick': [s.tick for s in states],
                'population': [s.population for s in states],
                'gini': [s.gini_coefficient for s in states],
                'top1_share': [s.top1_share for s in states],
                'top10_share': [s.top10_share for s in states],
                'bottom50_share': [s.bottom50_share for s in states],
                'mean_wealth': [s.mean_wealth for s in states],
                'median_wealth': [s.median_wealth for s in states],
                'num_employers': [s.num_employers for s in states],
                'avg_tools': [s.avg_tools for s in states],
                'avg_experience': [s.avg_experience for s in states],
                'avg_effective_skill': [s.avg_effective_skill for s in states],
                'profit_rate_std': [s.profit_rate_std for s in states],
                'total_production': [s.total_production for s in states],
                'hhi': [s.hhi for s in states],
                # ML-9: Demographic indicators
                'births': [s.births_this_tick for s in states],
                'deaths': [s.deaths_this_tick for s in states],
                # ML-10: Credit/debt indicators
                'total_credit': [s.total_credit for s in states],
                'avg_debt': [s.avg_debt for s in states],
                'avg_credit_score': [s.avg_credit_score for s in states],
                'default_rate': [s.default_rate for s in states],
                'credit_gini': [s.credit_gini for s in states],
                # ML-10/11: Trust and coercion indicators
                'coercion_attempts': [s.coercion_attempts for s in states],
                'coercion_successes': [s.coercion_successes for s in states],
                'avg_reciprocity_size': [s.avg_reciprocity_size for s in states],
                'ingroup_trade_ratio': [s.ingroup_trade_ratio for s in states],
                'cross_tag_coercion_ratio': [s.cross_tag_coercion_ratio for s in states],
                'coercion_victim_debuff_count': [s.coercion_victim_debuff_count for s in states],
            },
            'death_log': self.economy.death_log,
            'birth_log': self.economy.birth_log,
        }

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file"""
        results = self.get_results()
        # Serialize data in death_log and birth_log
        if 'death_log' in results:
            for entry in results['death_log']:
                if 'skill' in entry:
                    entry['skill'] = float(entry['skill'])
        if 'birth_log' in results:
            for entry in results['birth_log']:
                if 'child_skill' in entry:
                    entry['child_skill'] = float(entry['child_skill'])
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to {filepath}")


def run_parallel(
    num_runs: int = 10,
    num_ticks: int = 500,
    base_seed: int = 42,
    **sim_kwargs,
) -> List[Dict[str, Any]]:
    """
    Run multiple simulations in parallel (different random seeds)
    Used to obtain statistically robust conclusions
    """
    all_results = []
    for i in range(num_runs):
        seed = base_seed + i
        sim_kwargs['seed'] = seed
        sim = Simulation(**sim_kwargs).initialize()
        print(f"\n=== Run {i+1}/{num_runs} (seed={seed}) ===")
        sim.run(num_ticks=num_ticks, verbose=True)
        all_results.append(sim.get_results())

    return all_results
