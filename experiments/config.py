"""
Experiment Configuration

Predefined institutional rule combinations for comparison experiments.
Corresponds to presets in theory/institutional_rules/README.md.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from rules import (
    PrivateProperty, CollectiveProperty, MixedProperty,
    FreeExchange, RegulatedExchange, NoExchange,
    FreeEmployment, MinWageEmployment, NoEmployment,
    NoTax, ProportionalTax, ProgressiveTax,
    NoAntitrust, ThresholdAntitrust, ProactiveAntitrust,
    CommandEconomy,
)


@dataclass
class ExperimentConfig:
    """单个实验的配置"""
    name: str
    description: str
    num_agents: int = 200
    num_ticks: int = 500
    initial_endowment: float = 10.0
    seed: int = 42
    num_runs: int = 5  # Number of repetitions (different random seeds)
    rules: list = field(default_factory=list)
    meta_law_config: Optional[Dict[str, Any]] = None

    def build_rules(self) -> list:
        """构造规则实例列表"""
        return [rule for rule in self.rules]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'num_agents': self.num_agents,
            'num_ticks': self.num_ticks,
            'initial_endowment': self.initial_endowment,
            'seed': self.seed,
            'num_runs': self.num_runs,
            'rules': [r.name() for r in self.rules],
        }


# ============================================================
# Preset experiment configurations
# ============================================================

PRESETS: Dict[str, ExperimentConfig] = {}

# --- Combination A: Laissez-Faire Capitalism ---
PRESETS['laissez_faire'] = ExperimentConfig(
    name='A: Laissez-Faire Capitalism',
    description='Laissez-faire capitalism: no intervention, let markets evolve freely',
    num_agents=200,
    num_ticks=500,
    rules=[
        PrivateProperty(),
        FreeExchange(),
        FreeEmployment(),
        NoTax(),
        NoAntitrust(),
    ],
)

# --- Combination B: Social Democracy ---
PRESETS['social_democracy'] = ExperimentConfig(
    name='B: Social Democracy',
    description='Social democracy: free market + progressive tax + minimum wage + antitrust',
    num_agents=200,
    num_ticks=500,
    rules=[
        PrivateProperty(),
        RegulatedExchange(transaction_tax_rate=0.03),
        MinWageEmployment(minimum_wage=1.5),
        ProgressiveTax(redistribution='targeted'),
        ThresholdAntitrust(hhi_threshold=0.25),
    ],
)

# --- Combination C: Command Economy Simulation (central allocation) ---
PRESETS['command_economy'] = ExperimentConfig(
    name='C: Command Economy',
    description='Command economy: central collection and three-tier distribution by survival need + labor contribution + equal share',
    num_agents=200,
    num_ticks=500,
    rules=[
        CommandEconomy(survival_ratio=0.6, contribution_ratio=0.3, equal_ratio=0.1),
    ],
)

# --- Combination D: Mixed Economy ---
PRESETS['mixed_economy'] = ExperimentConfig(
    name='D: Mixed Economy',
    description='Mixed economy: private property + regulated exchange + proportional tax + proactive antitrust',
    num_agents=200,
    num_ticks=500,
    rules=[
        MixedProperty(tool_tax_rate=0.15, redistribution_rate=0.6),
        RegulatedExchange(transaction_tax_rate=0.02),
        MinWageEmployment(minimum_wage=1.2),
        ProportionalTax(rate=0.03, redistribution=True),
        ProactiveAntitrust(max_share=0.3),
    ],
)

# --- Crisis dynamics experiment ---
PRESETS['crisis_dynamics'] = ExperimentConfig(
    name='Crisis Dynamics (No Intervention)',
    description='Crisis dynamics experiment: long run, observe endogenous crisis emergence',
    num_agents=300,
    num_ticks=1000,
    rules=[
        PrivateProperty(),
        FreeExchange(),
        FreeEmployment(),
        NoTax(),
        NoAntitrust(),
    ],
)
