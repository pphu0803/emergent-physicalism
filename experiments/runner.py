"""
Experiment Runner

Manages the execution, result collection, and comparison analysis of multiple experiments.
Supports multi-process parallel acceleration for independent simulation runs.
"""

import os
import json
import time
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

from core.simulation import Simulation
from analysis.metrics import EmergenceAnalyzer
from analysis.visualization import plot_results, plot_comparison
from .config import ExperimentConfig


def _run_single_sim(args: Tuple[int, int, float, dict, list]) -> Dict[str, Any]:
    """Top-level function: run a single simulation (pickle-friendly, for multiprocessing)

    Args:
        args: (seed, num_agents, initial_endowment, meta_law_config, rules_dicts)

    Returns:
        Simulation result dictionary
    """
    seed, num_agents, initial_endowment, meta_law_config, rules_dicts = args

    # Rebuild rule instances
    from rules import (
        PrivateProperty, CollectiveProperty, MixedProperty,
        FreeExchange, RegulatedExchange, NoExchange,
        FreeEmployment, MinWageEmployment, NoEmployment,
        NoTax, ProportionalTax, ProgressiveTax,
        NoAntitrust, ThresholdAntitrust, ProactiveAntitrust,
        CommandEconomy,
        FreeCoercion, NoCoercion,
    )

    RULE_REGISTRY = {
        'property_private': PrivateProperty,
        'property_collective': CollectiveProperty,
        'property_mixed': lambda **kw: MixedProperty(**{k: v for k, v in kw.items()}),
        'exchange_free': FreeExchange,
        'exchange_regulated': lambda **kw: RegulatedExchange(**{k: v for k, v in kw.items()}),
        'exchange_none': NoExchange,
        'employment_free': FreeEmployment,
        'employment_minwage': lambda **kw: MinWageEmployment(**{k: v for k, v in kw.items()}),
        'employment_none': NoEmployment,
        'tax_none': NoTax,
        'tax_proportional': lambda **kw: ProportionalTax(**{k: v for k, v in kw.items()}),
        'tax_progressive': lambda **kw: ProgressiveTax(**{k: v for k, v in kw.items()}),
        'antitrust_none': NoAntitrust,
        'antitrust_threshold': lambda **kw: ThresholdAntitrust(**{k: v for k, v in kw.items()}),
        'antitrust_proactive': lambda **kw: ProactiveAntitrust(**{k: v for k, v in kw.items()}),
        'command_economy': lambda **kw: CommandEconomy(**{k: v for k, v in kw.items()}),
        'coercion_free': FreeCoercion,
        'coercion_none': lambda **kw: NoCoercion(**{k: v for k, v in kw.items()}),
    }

    rules = []
    for rd in rules_dicts:
        rule_name = rd['name']
        rule_kwargs = rd.get('kwargs', {})
        if rule_name in RULE_REGISTRY:
            rules.append(RULE_REGISTRY[rule_name](**rule_kwargs))
        else:
            rules.append(RULE_REGISTRY.get(rule_name, NoTax)())

    num_ticks = meta_law_config.pop('_num_ticks', 500)

    sim = Simulation(
        num_agents=num_agents,
        initial_endowment=initial_endowment,
        meta_law_config=meta_law_config,
        seed=seed,
        rules=rules,
    ).initialize()
    sim.run(num_ticks=num_ticks, verbose=False)
    return sim.get_results()


class ExperimentRunner:

    def __init__(self, output_dir: str = 'experiments'):
        self.output_dir = output_dir
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _serialize_rules(rules: list) -> List[Dict[str, Any]]:
        """Serialize rule instances to pickle-compatible dicts (only keep constructor parameters)"""
        # These attributes are runtime state, not constructor parameters
        RUNTIME_ATTRS = {
            'total_collected', 'interventions', 'collected_tax',
            'total_interventions', 'name', 'description', 'penalty',
        }
        result = []
        for r in rules:
            rd = {'name': r.name(), 'description': r.description(), 'kwargs': {}}
            for attr in vars(r):
                if attr.startswith('_'):
                    continue
                if attr in RUNTIME_ATTRS:
                    continue
                rd['kwargs'][attr] = getattr(r, attr)
            result.append(rd)
        return result

    def run_single(self, config: ExperimentConfig, verbose: bool = True) -> List[Dict[str, Any]]:
        print(f"\n{'='*60}")
        print(f"  Experiment: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  Config: {config.num_agents} agents, {config.num_ticks} ticks, {config.num_runs} runs")
        print(f"  Rules: {[r.name() for r in config.rules]}")
        print(f"{'='*60}")

        all_results = []
        rules = config.build_rules()

        for run_idx in range(config.num_runs):
            seed = config.seed + run_idx
            print(f"\n--- Run {run_idx + 1}/{config.num_runs} (seed={seed}) ---")

            sim = Simulation(
                num_agents=config.num_agents,
                initial_endowment=config.initial_endowment,
                meta_law_config=config.meta_law_config or {},
                seed=seed,
                rules=deepcopy(rules),
            ).initialize()

            sim.run(num_ticks=config.num_ticks, verbose=verbose)
            result = sim.get_results()
            all_results.append(result)

        self.results[config.name] = all_results
        return all_results

    def run_single_parallel(
        self,
        config: ExperimentConfig,
        max_workers: int = 16,
    ) -> List[Dict[str, Any]]:
        print(f"\n{'='*60}")
        print(f"  Experiment: {config.name} (PARALLEL, {max_workers} workers)")
        print(f"  Config: {config.num_agents} agents, {config.num_ticks} ticks, {config.num_runs} runs")
        print(f"  Rules: {[r.name() for r in config.rules]}")
        print(f"{'='*60}")

        ml_config = dict(config.meta_law_config or {})
        ml_config['_num_ticks'] = config.num_ticks

        rules_dicts = self._serialize_rules(config.rules)

        tasks = [
            (config.seed + i, config.num_agents, config.initial_endowment, ml_config, rules_dicts)
            for i in range(config.num_runs)
        ]

        start_time = time.time()

        with mp.Pool(processes=max_workers) as pool:
            all_results = list(pool.map(_run_single_sim, tasks))

        elapsed = time.time() - start_time
        print(f"\n  {config.num_runs} runs completed in {elapsed:.1f}s "
              f"({elapsed/config.num_runs:.1f}s/run)")

        self.results[config.name] = all_results
        return all_results

    def run_comparison(
        self,
        configs: List[ExperimentConfig],
        verbose: bool = True,
        parallel: bool = True,
        max_workers: int = 16,
    ) -> Dict[str, List[Dict[str, Any]]]:
        start_time = time.time()

        for config in configs:
            if parallel:
                self.run_single_parallel(config, max_workers=max_workers)
            else:
                self.run_single(config, verbose=verbose)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  All experiments complete in {elapsed:.1f}s")
        print(f"{'='*60}")

        return self.results

    def analyze(self, config_name: str) -> None:
        if config_name not in self.results:
            print(f"  No results for '{config_name}'")
            return

        result = self.results[config_name][-1]
        analysis = EmergenceAnalyzer.analyze_all(result)
        report = EmergenceAnalyzer.format_report(analysis)
        print(report)

        exp_dir = os.path.join(self.output_dir, config_name.replace(' ', '_').replace(':', ''))
        os.makedirs(exp_dir, exist_ok=True)
        report_path = os.path.join(exp_dir, 'emergence_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Report saved to {report_path}")

    def analyze_all(self) -> None:
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE EMERGENCE ANALYSIS")
        print("=" * 70)

        for config_name in self.results:
            self.analyze(config_name)

    def visualize(self, config_name: str) -> None:
        if config_name not in self.results:
            print(f"  No results for '{config_name}'")
            return

        result = self.results[config_name][-1]
        exp_dir = os.path.join(self.output_dir, config_name.replace(' ', '_').replace(':', ''))
        os.makedirs(exp_dir, exist_ok=True)

        plot_path = os.path.join(exp_dir, 'results.png')
        plot_results(result, save_path=plot_path, title=config_name)

    def visualize_comparison(self, save_name: str = 'comparison') -> None:
        if len(self.results) < 2:
            print("  Need at least 2 experiment results for comparison")
            return

        labels = list(self.results.keys())
        results_list = [self.results[name][-1] for name in labels]

        save_path = os.path.join(self.output_dir, f'{save_name}.png')
        plot_comparison(
            results_list,
            labels,
            save_path=save_path,
            title='Institutional Comparison',
        )

    def save_all(self) -> None:
        for config_name, results_list in self.results.items():
            exp_dir = os.path.join(self.output_dir, config_name.replace(' ', '_').replace(':', ''))
            os.makedirs(exp_dir, exist_ok=True)

            for i, result in enumerate(results_list):
                result_path = os.path.join(exp_dir, f'run_{i+1}.json')
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            print(f"  Saved {len(results_list)} results for '{config_name}'")
