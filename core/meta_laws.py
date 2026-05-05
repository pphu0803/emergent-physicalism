"""
Meta-Law Enforcer

Enforces physical/philosophical meta-laws at each simulation timestep.
These laws are inviolable -- they constitute the "laws of physics" of the simulated world.

Corresponds to ML-1, ML-2a, ML-2b, ML-3 ~ ML-9 in theory/meta_laws.md
"""

import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any

from .agent import MicroIndividual

if TYPE_CHECKING:
    from .economy import Economy


class MetaLawEnforcer:
    """Meta-Law Enforcer -- the physics of the simulated world"""

    def __init__(
        self,
        resource_ceiling: float = 5000.0,
        natural_inflow_per_capita: float = 0.72,
        inflow_noise_std: float = 4.0,
        subsistence_cost: float = 0.9,
        crisis_threshold_ticks: int = 5,
        labor_capacity: float = 1.0,
        # ML-3 relaxation: time irreversibility loosening
        undo_prob: float = 0.0,
        # ML-7 relaxation: randomness control
        noise_std_override: float = -1.0,  # -1=use default, 0=no noise
        explore_rate_override: float = -1.0,  # -1=use default
        # ML-8 relaxation: learning rate (0=no learning)
        learning_rate: float = 0.02,
        # Depreciation rate (configurable externally)
        depreciation_rate: float = 0.02,
        # ML-9: Demographic dynamics parameters
        demographic_enabled: bool = False,
        population_ceiling: int = 500,
        reproduction_threshold: float = 2.0,
        reproduction_base_prob: float = 0.01,
        reproduction_cooldown: int = 40,
        reproduction_cost_ratio: float = 0.25,
        inheritance_ratio: float = 0.8,
        mutation_strength: float = 0.15,
        aging_enabled: bool = True,
        aging_base_rate: float = 0.0001,       # Infant mortality rate
        aging_rate_multiplier: float = 0.00004, # Life expectancy ~2200 ticks ~ 42 years
        # ML-10: Trust channel
        trust_enabled: bool = False,
        num_tags: int = 3,
        trust_alpha: float = 0.1,  # Reciprocity balance weight
        trust_beta: float = 0.3,  # Same-tag weight
        trust_gamma: float = 0.005,  # Time decay weight
        ingroup_friction_reduction: float = 0.2,  # 20% transaction friction reduction for same-tag
        # ML-11: Coercion channel
        coercion_enabled: bool = False,
        coercion_cost: float = 0.3,
        defense_bonus: float = 1.2,
        coercion_transfer_rate: float = 0.15,
        # ML-10: Unified trust lending
        lending_enabled: bool = True,
        interest_rate: float = 0.01,
    ):
        # ML-1: Material conservation
        self.resource_ceiling = resource_ceiling
        # Natural inflow per capita: default 0.72 = subsistence_cost * 0.8
        # Leave 20% gap to drive production competition (embodiment of ML-4 scarcity)
        # In per-capita terms, total inflow = per_capita * alive_count, adjusts with population
        self.natural_inflow_per_capita = natural_inflow_per_capita
        # Backward compatibility (if external code passes natural_inflow_rate, convert to per capita)
        self.natural_inflow_rate = natural_inflow_per_capita  # Compatibility field
        self.inflow_noise_std = inflow_noise_std

        # ML-5: Survival baseline
        self.subsistence_cost = subsistence_cost
        self.crisis_threshold_ticks = crisis_threshold_ticks

        # ML-2: Labor conservation
        self.labor_capacity = labor_capacity

        # ML-9: Demographic dynamics
        self.demographic_enabled = demographic_enabled
        self.population_ceiling = population_ceiling
        self.reproduction_threshold = reproduction_threshold
        self.reproduction_base_prob = reproduction_base_prob
        self.reproduction_cooldown = reproduction_cooldown
        self.reproduction_cost_ratio = reproduction_cost_ratio
        self.inheritance_ratio = inheritance_ratio
        self.mutation_strength = mutation_strength
        self.aging_enabled = aging_enabled
        self.aging_base_rate = aging_base_rate
        self.aging_rate_multiplier = aging_rate_multiplier

        # ML-3 relaxation: time irreversibility loosening
        self.undo_prob = undo_prob
        self._last_trade_log: List[Dict[str, Any]] = []

        # ML-7 relaxation: randomness control
        self.noise_std_override = noise_std_override
        self.explore_rate_override = explore_rate_override

        # ML-8 relaxation: learning rate
        self.learning_rate = learning_rate

        # Depreciation rate
        self.depreciation_rate = depreciation_rate

        # ML-10: Trust channel
        self.trust_enabled = trust_enabled
        self.num_tags = num_tags
        self.trust_alpha = trust_alpha
        self.trust_beta = trust_beta
        self.trust_gamma = trust_gamma
        self.ingroup_friction_reduction = ingroup_friction_reduction

        # ML-11: Coercion channel
        self.coercion_enabled = coercion_enabled
        self.coercion_cost = coercion_cost
        self.defense_bonus = defense_bonus
        self.coercion_transfer_rate = coercion_transfer_rate

        # ML-10: Unified trust lending
        self.lending_enabled = lending_enabled
        self.interest_rate = interest_rate

    def enforce_conservation(self, economy: 'Economy') -> None:
        """ML-1: Material conservation -- natural resource inflow (per capita * alive population)"""
        alive_agents = [a for a in economy.agents if a.alive]
        if not alive_agents:
            return

        # Total inflow = per capita inflow * alive population (Malthusian dynamics)
        base_inflow = self.natural_inflow_per_capita * len(alive_agents)
        noise = np.random.normal(0, self.inflow_noise_std)
        inflow = max(0, base_inflow * (1 + noise))
        inflow = min(inflow, self.resource_ceiling - economy.total_resources())

        # Distribute randomly to alive agents (simulates natural resource spatial distribution)
        if inflow > 0:
            per_agent = inflow / len(alive_agents)
            for agent in alive_agents:
                agent.resources += per_agent

    def enforce_labor_conservation(self, economy: 'Economy') -> None:
        """ML-2a: Bounded labor capacity -- reset each agent's labor (non-storable, unused portion wasted)"""
        for agent in economy.agents:
            if agent.alive:
                # Reproduction penalty: reproducing last tick reduces this tick's labor capacity
                penalty = getattr(agent, 'reproduction_penalty', 0.0)
                agent.labor_remaining = self.labor_capacity * (1.0 - penalty)
                # Penalty only lasts one tick
                agent.reproduction_penalty = 0.0

    def enforce_time_irreversibility(self, economy: 'Economy') -> None:
        """ML-3: Time irreversibility -- clean up dead agents, advance time"""
        # ML-3 relaxation: undo the most recent trade with probability undo_prob
        if self.undo_prob > 0 and self._last_trade_log:
            if np.random.random() < self.undo_prob:
                self._undo_last_trade(economy)

        economy.tick += 1
        # Death is a terminal state
        economy.agents = [a for a in economy.agents if a.alive]

    def _undo_last_trade(self, economy: 'Economy') -> None:
        """Undo the most recent trade (for ML-3 relaxation)"""
        if not self._last_trade_log:
            return
        trade = self._last_trade_log[-1]
        seller = next((a for a in economy.agents if a.id == trade['seller_id']), None)
        buyer = next((a for a in economy.agents if a.id == trade['buyer_id']), None)
        if seller and buyer and seller.alive and buyer.alive:
            qty = trade['qty']
            value = trade['value']
            # Restore goods
            seller_goods = getattr(seller, 'goods', 0)
            buyer_goods = getattr(buyer, 'goods', 0)
            if buyer_goods >= qty:
                seller.goods = seller_goods + qty
                buyer.goods = buyer_goods - qty
                # Restore resources
                seller.resources -= value
                buyer.resources += value
        self._last_trade_log.pop()

    def enforce_scarcity(self, economy: 'Economy') -> None:
        """ML-4: Scarcity -- naturally achieved through resource_ceiling"""
        pass  # Scarcity is a natural result of conservation + limited inflow, no extra enforcement needed

    def enforce_survival_baseline(self, economy: 'Economy') -> None:
        """ML-5: Survival baseline -- consume subsistence resources, check alive/dead"""
        for agent in economy.agents:
            if not agent.alive:
                continue
            agent.resources -= self.subsistence_cost
            if agent.resources < 0:
                agent.crisis_ticks += 1
                if agent.crisis_ticks >= self.crisis_threshold_ticks:
                    agent.alive = False
                    economy.record_death(agent)
            else:
                agent.crisis_ticks = 0

    def enforce_demographic_dynamics(self, economy: 'Economy') -> None:
        """
        ML-9: Demographic dynamics -- reproduction mechanism

        Biological fact: populations expand when resources are abundant,
        reproduction rate declines when resources are scarce.
        Not an economic assumption, but a fundamental property of any living system.

        Execution timing: after institutional rules, before time advancement.
        This ensures institutional allocation outcomes affect reproduction decisions
        (taxation, redistribution change reproducible resources).

        Mechanism design:
        - Reproduction threshold: resources > reproduction_threshold (= 5x subsistence)
        - Reproduction probability: increases with resource abundance, with cap
        - Reproduction cost: parent transfers 25% of resources to offspring (ML-1 conservation)
        - Reproduction penalty: parent's labor capacity reduced 40% next tick (biological cost)
        - Cooldown: no reproduction for 40 ticks after reproducing
        - Trait inheritance: offspring inherits 80% parent skill + 20% random mutation
        - Carrying capacity: suppresses reproduction when population exceeds ceiling
        """
        if not self.demographic_enabled:
            return

        alive_agents = [a for a in economy.agents if a.alive]
        if not alive_agents:
            return

        n_alive = len(alive_agents)
        new_agents = []

        for agent in alive_agents:
            # Cooldown check
            if agent.reproduction_cooldown > 0:
                agent.reproduction_cooldown -= 1
                continue

            # Reproduction threshold: resources must exceed threshold
            if agent.resources <= self.reproduction_threshold:
                continue

            # Carrying capacity suppression: probability decays linearly when population exceeds 80% ceiling
            capacity_pressure = 1.0
            if n_alive > self.population_ceiling * 0.8:
                capacity_pressure = max(0.0, 1.0 - (n_alive - self.population_ceiling * 0.8) /
                                       (self.population_ceiling * 0.2))
            if capacity_pressure <= 0:
                continue

            # Reproduction probability: higher resources -> higher probability (with cap)
            surplus_ratio = min((agent.resources / self.reproduction_threshold - 1.0), 5.0)
            p_reproduce = self.reproduction_base_prob * surplus_ratio * capacity_pressure

            if np.random.random() >= p_reproduce:
                continue

            # === Reproduction occurs ===

            # 1. Resource transfer (ML-1 conservation: parent loses, offspring gains)
            transfer = agent.resources * self.reproduction_cost_ratio
            agent.resources -= transfer

            # 2. Labor penalty (takes effect next tick)
            agent.reproduction_penalty = 0.4
            agent.reproduction_cooldown = self.reproduction_cooldown

            # 3. Offspring traits: inheritance + mutation
            child_skill = max(0.01, min(0.99,
                agent.skill * self.inheritance_ratio +
                np.random.beta(2, 5) * (1 - self.inheritance_ratio) +
                np.random.normal(0, self.mutation_strength)
            ))

            child_risk = max(0.01, min(0.99,
                agent.risk_tolerance * self.inheritance_ratio +
                np.random.beta(2, 2) * (1 - self.inheritance_ratio) +
                np.random.normal(0, self.mutation_strength * 0.5)
            ))

            # 4. Create offspring
            child = MicroIndividual(
                initial_resources=transfer,
                skill=child_skill,
                risk_tolerance=child_risk,
            )
            # Offspring defaults to necessities sector (new individuals start from survival)
            child.sector = 'necessities'

            new_agents.append(child)
            economy.record_birth(child, parent_id=agent.id)

        # Batch add to agent list (start participating in economy next tick)
        economy.agents.extend(new_agents)

    def enforce_all(self, economy: 'Economy') -> None:
        """Enforce all meta-laws (in correct order)"""
        self.enforce_labor_conservation(economy)
        self.enforce_survival_baseline(economy)
        self.enforce_conservation(economy)
        self.enforce_time_irreversibility(economy)
