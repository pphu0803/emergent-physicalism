"""
Micro-Individual

The sole agent type in the system.
No "rational economic agent" assumption -- acts on local information and simple heuristics.

Built-in goals (minimal behavioral axioms, non-economic):
1. Survival first: resources must not fall below subsistence level
2. Risk avoidance: avoid resources dropping to dangerous levels
3. Local imitation: observe neighbors, imitate successful strategies
4. Conditional reciprocity: sustain cooperation with mutually beneficial partners
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Memory:
    """Agent experience memory"""
    max_memory: int = 20
    # (partner_id, action_type, my_payoff, tick)
    interactions: List[Tuple[int, str, float, int]] = field(default_factory=list)
    # Average payoff from recent N interactions
    recent_returns: List[float] = field(default_factory=list)

    def add_interaction(self, other_id: int, action: str, payoff: float, tick: int):
        self.interactions.append((other_id, action, payoff, tick))
        if len(self.interactions) > self.max_memory:
            self.interactions.pop(0)
        self.recent_returns.append(payoff)
        if len(self.recent_returns) > self.max_memory:
            self.recent_returns.pop(0)

    def avg_recent_return(self) -> float:
        if not self.recent_returns:
            return 0.0
        return np.mean(self.recent_returns)

    def trust_score(self, other_id: int, same_tag: bool = False, tick: int = 0,
                    reciprocity_balance: float = 0.0) -> float:
        """
        Compute trust based on reciprocity ledger and tag (ML-10 trust channel)

        trust = sigmoid(alpha * balance + beta * same_tag - gamma * time_since_last)
        """
        if reciprocity_balance != 0.0 or same_tag:
            time_since = tick - self._last_interaction_tick(other_id)
            input_val = 0.1 * reciprocity_balance + 0.3 * (1 if same_tag else 0) - 0.005 * time_since
            return float(1.0 / (1.0 + np.exp(-input_val)))
        # No reciprocity record and different tag: use interaction history
        interactions = [(a, p) for oid, a, p, _ in self.interactions if oid == other_id]
        if not interactions:
            return 0.5
        positive = sum(1 for a, p in interactions if p > 0)
        return positive / len(interactions)

    def _last_interaction_tick(self, other_id: int) -> int:
        """Get the tick of last interaction with a given agent"""
        last_tick = 0
        for _, _, _, t in self.interactions:
            if t > last_tick:
                last_tick = t
        return last_tick


class MicroIndividual:
    """Micro-Individual -- the sole agent type in the system"""

    _id_counter = 0

    def __init__(
        self,
        initial_resources: float = 10.0,
        skill: Optional[float] = None,
        risk_tolerance: Optional[float] = None,
        explore_rate: float = 0.05,
        tag: int = 0,
    ):
        MicroIndividual._id_counter += 1
        self.id = MicroIndividual._id_counter

        # Material resources
        self.resources = initial_resources
        self.initial_resources = initial_resources  # Record initial endowment (for death analysis)

        # Labor (ML-2: bounded, non-transferable)
        self.labor_remaining = 1.0

        # Individual traits (sources of heterogeneity)
        self.skill = skill if skill is not None else np.random.beta(2, 5)  # Most agents have low skill
        self.risk_tolerance = risk_tolerance if risk_tolerance is not None else np.random.beta(2, 2)

        # ML-6: Bounded cognition
        self.memory = Memory()
        self.observation_range = 5  # Can only observe the N most recent interaction partners
        self.price_belief = 1.0  # Local belief about "fair price"

        # ML-7: Random exploration
        self.explore_rate = explore_rate

        # State
        self.alive = True
        self.crisis_ticks = 0
        self.employer_id: Optional[int] = None
        self.employees: List[int] = []

        # Production tools/capital (emerges from accumulation)
        self.tools = 0.0  # Tool level, improves production efficiency
        self.is_employer = False
        self._coercion_debuff_ticks: int = 0  # Productivity debuff duration after coercion attack

        # Learning by doing: experience accumulation
        # Based on cognitive psychology "repeated practice -> proficiency" curve
        # Not an economic agent assumption, but a basic principle of biological learning
        self.production_experience = 0.0  # Cumulative production experience
        self.effective_skill = self.skill  # Effective skill = innate skill * experience bonus

        # Multi-sector: current sector
        self.sector: str = 'necessities'  # Default to necessities sector
        self.sector_experience: Dict[str, float] = {'necessities': 0.0, 'capital_goods': 0.0, 'luxuries': 0.0}
        self.sector_switch_cooldown: int = 0  # Sector switch cooldown

        # ML-9: Demographic dynamics -- reproduction
        self.reproduction_cooldown: int = 0  # Reproduction cooldown (ticks)
        self.reproduction_penalty: float = 0.0  # Reproduction labor penalty (takes effect next tick, 0~0.5)

        # ML-10: Credit/debt
        self.debt: float = 0.0  # Outstanding debt (positive = owes)
        self.credit_score: float = 0.5  # Credit score (0~1)
        self.loans: List[Dict[str, Any]] = []  # Active loans: [{lender_id, principal, interest_rate, remaining, maturity_tick}]
        self.default_count: int = 0  # Number of defaults

        # ML-10: Trust channel
        self.tag: int = tag  # Exogenous tag (0, 1, 2, ...)
        self.reciprocity_ledger: Dict[int, float] = {}  # agent_id -> net reciprocity balance

        # Behavior mode (dynamically computed each tick by choose_sector)
        self.behavior_mode: str = 'survival'  # 'survival' | 'security' | 'profit'

    def compute_behavior_mode(self, subsistence_cost: float) -> str:
        """
        Compute behavior mode based on current resources relative to subsistence.

        Internalized upgrade of ML-5 (survival baseline): subsistence is not just
        a passive death timer, but the primary driver of behavioral priorities.

        Three-tier behavior modes:
        - Survival mode: resources < subsistence * 2
          Lock to necessities, block non-survival signals, extreme risk aversion
        - Security mode: subsistence * 2 <= resources < subsistence * 5
          Allow capital goods, allow cautious investment, ensure necessities self-sufficiency
        - Profit mode: resources >= subsistence * 5
          All sectors open, pursue high returns, allow speculation

        Neuroscience basis: brain reward system weighting of future rewards
        varies dynamically with current resource scarcity.
        """
        if self.resources < subsistence_cost * 2:
            return 'survival'
        elif self.resources < subsistence_cost * 5:
            return 'security'
        else:
            return 'profit'

    def _allowed_sectors(self, sector_thresholds: Dict[str, float]) -> List[str]:
        """Return reachable sectors based on behavior mode"""
        if self.behavior_mode == 'survival':
            return ['necessities']  # Survival mode: only necessities
        elif self.behavior_mode == 'security':
            allowed = ['necessities']  # Security mode: necessities first
            if self.tools >= sector_thresholds.get('capital_goods', 999):
                allowed.append('capital_goods')
            return allowed
        else:
            # Profit mode: all open (subject to entry thresholds)
            return [s for s in ['necessities', 'capital_goods', 'luxuries']
                    if self.tools >= sector_thresholds.get(s, 0)]

    def choose_sector(self, sector_profits: Dict[str, float], sector_thresholds: Dict[str, float], subsistence_cost: float = 0.9) -> None:
        """
        Choose production sector -- driven by behavior mode

        Behavior mode determines what agents can see and where they can go.
        Given the same profit rate signal, agents in different modes make
        radically different choices.

        ML-5 drives behavior mode -> behavior mode constrains sector choice -> ML-6 constrains information processing.
        """
        # 1. Update behavior mode (dynamically computed each tick)
        self.behavior_mode = self.compute_behavior_mode(subsistence_cost)

        # 2. Determine reachable sectors
        allowed = self._allowed_sectors(sector_thresholds)

        # Survival mode: forced lock to necessities, no choice
        if self.behavior_mode == 'survival':
            if self.sector != 'necessities':
                self._switch_sector('necessities')
            return

        # 3. Security mode: necessities self-sufficiency first
        if self.behavior_mode == 'security':
            # If necessities producers are insufficient (<50% of alive), force stay in necessities
            return

        if self.sector_switch_cooldown > 0:
            self.sector_switch_cooldown -= 1
            return

        # 4. Profit mode: sector choice based on profit rate signal (original logic)
        current_profit = sector_profits.get(self.sector, 0.0)
        current_sector_exp = self.sector_experience.get(self.sector, 0.0)

        # Random exploration (ML-7)
        if np.random.random() < self.explore_rate:
            candidates = [s for s in allowed if s in sector_profits]
            if candidates:
                self._switch_sector(np.random.choice(candidates))
            return

        # Compute attractiveness of each reachable sector (within allowed only)
        best_sector = self.sector
        best_score = current_profit + current_sector_exp * 0.01

        for sector_id in allowed:
            if sector_id == self.sector:
                continue

            profit = sector_profits.get(sector_id, 0.0)
            exp_bonus = self.sector_experience.get(sector_id, 0.0) * 0.01
            risk_adjusted_profit = profit * (1 + self.risk_tolerance * 0.5)
            score = risk_adjusted_profit + exp_bonus

            # Only switch when new sector is significantly better
            if score > best_score * 1.15:
                best_sector = sector_id
                best_score = score

        if best_sector != self.sector:
            self._switch_sector(best_sector)

    def _switch_sector(self, new_sector: str) -> None:
        """Execute sector switch"""
        self.sector = new_sector
        self.sector_switch_cooldown = 10  # 10 ticks cooldown
        # Sector switch loses some experience (re-learning cost)
        # But retains cross-sector general experience (50% retention)
        # Specific sector experience resets more
        for s in self.sector_experience:
            if s != new_sector:
                self.sector_experience[s] *= 0.8  # Cross-sector experience retains 80%
        self.production_experience *= 0.7  # Production experience depreciates 30%
        self._update_effective_skill()

    def _update_effective_skill(self) -> None:
        """Update effective skill (based on cross-sector combined experience + coercion debuff)"""
        sector_exp = self.sector_experience.get(self.sector, 0.0)
        experience_bonus = 1 + 0.5 * (1 - np.exp(-sector_exp * 0.05))
        self.effective_skill = self.skill * experience_bonus
        if self._coercion_debuff_ticks > 0:
            self._coercion_debuff_ticks -= 1
            self.effective_skill *= 0.5

    def _update_experience(self, labor_input: float, learning_rate: float = 0.02) -> None:
        """Learning by doing: experience accumulation from production (per sector)"""
        marginal_learning = learning_rate / (1 + self.production_experience * 0.1)
        self.production_experience += labor_input * (1 + marginal_learning)

        # Sector experience accumulation
        if self.sector in self.sector_experience:
            self.sector_experience[self.sector] += labor_input * (1 + marginal_learning)

        self._update_effective_skill()

    def produce(self, natural_resources_available: float, learning_rate: float = 0.02) -> float:
        """
        Use labor and tools to produce goods
        Output = effective_skill * (1 + tool_bonus) * labor_input * (1 + noise)

        Effective skill = innate skill * experience bonus (learning by doing)
        """
        if not self.alive or self.labor_remaining <= 0:
            return 0.0

        # Labor input
        labor_input = min(self.labor_remaining, 1.0)
        self.labor_remaining -= labor_input

        # Learning by doing: experience from practice
        self._update_experience(labor_input, learning_rate=learning_rate)

        # Output computation (using effective skill, not innate skill)
        tool_bonus = 0.3 * self.tools  # Efficiency gain from tools
        base_output = self.effective_skill * (1 + tool_bonus) * labor_input

        # ML-7: Random noise
        noise = np.random.normal(0, 0.1)
        output = max(0, base_output * (1 + noise))

        # Consume some resources as production cost
        material_cost = 0.1 * labor_input
        self.resources -= material_cost

        return output

    def decide_trade(
        self,
        own_stock: float,
        market_price: float,
        partner_trust: float,
    ) -> Tuple[str, float]:
        """
        Decide whether to trade, direction, and quantity
        Returns: ('buy'|'sell'|'skip', quantity)
        """
        if not self.alive:
            return ('skip', 0.0)

        # ML-7: Random exploration
        if np.random.random() < self.explore_rate:
            qty = np.random.uniform(0, max(0.1, own_stock * 0.3))
            direction = 'sell' if np.random.random() < 0.5 else 'buy'
            return (direction, qty)

        # Survival pressure driven
        if self.resources < self._danger_zone():
            # When resources are dangerously low, tend to sell inventory for resources
            if own_stock > 0:
                sell_qty = own_stock * (0.5 + 0.3 * (1 - self.risk_tolerance))
                return ('sell', sell_qty)

        # Imitate successful strategy
        if self.memory.avg_recent_return() > 0:
            return ('buy', min(own_stock * 0.2, self.resources * 0.1))

        # Trust threshold
        if partner_trust < 0.3:
            return ('skip', 0.0)

        return ('skip', 0.0)

    def decide_employment(self, wage_offer: float) -> bool:
        """
        Decide whether to accept employment
        Based on current resources and risk aversion
        """
        if not self.alive:
            return False

        # Not inclined to be employed when resources are abundant
        if self.resources > 20:
            return False

        # More likely to accept when survival pressure is high
        desperation = max(0, 1 - self.resources / 10)
        effective_wage = wage_offer * (1 + desperation * 0.5)

        # Compare with historical returns
        if self.memory.avg_recent_return() > effective_wage:
            return False

        return effective_wage > self.subsistence_wage()

    def decide_hire(self, candidate_id: int, candidate_trust: float) -> bool:
        """Decide whether to hire someone"""
        if not self.alive:
            return False
        if self.resources < 5:  # Employment requires minimum capital threshold
            return False
        if candidate_trust < 0.3:
            return False
        return True

    def decide_invest_in_tools(self) -> float:
        """Decide how much resources to invest in tools -- constrained by behavior mode"""
        if not self.alive:
            return 0.0

        # Survival mode: no investment (all resources for survival)
        if self.behavior_mode == 'survival':
            return 0.0

        # Invest when resources are abundant and production efficiency can improve
        buffer = self.resources - 8
        if buffer > 0 and self.tools < 10:
            marginal_return = 0.3 / (1 + self.tools * 0.8)
            if marginal_return > 0.03:
                skill_factor = self.skill / 0.2
                invest = buffer * 0.15 * skill_factor

                # Security mode: halve investment cap
                if self.behavior_mode == 'security':
                    invest *= 0.5

                return max(0, invest * (1 - self.risk_tolerance * 0.5))
        return 0.0

    def update_price_belief(self, observed_price: float) -> None:
        """ML-6: Update price belief based on local observation (Bayesian-style update)"""
        alpha = 0.1
        self.price_belief = (1 - alpha) * self.price_belief + alpha * observed_price

    @property
    def power(self) -> float:
        """ML-11: Physical power = f(resources, tools, effective_skill)

        sqrt(resources) ensures diminishing returns:
        - resources=1 -> contribution 0.5, resources=10 -> 1.58, resources=100 -> 5.0
        Tools and skill provide additional bonuses, but resources remain fundamental.
        """
        return 1.0 + np.sqrt(max(0, self.resources)) * 0.5 + self.tools * 0.2 + self.effective_skill * 1.0

    def update_reciprocity(self, other_id: int, value_given: float, value_received: float) -> None:
        """Update reciprocity ledger: positive balance = partner owes me net"""
        net = value_received - value_given
        self.reciprocity_ledger[other_id] = self.reciprocity_ledger.get(other_id, 0.0) + net

    def decide_coercion(
        self,
        possible_targets: list,
        subsistence_cost: float,
        coercion_cost: float,
    ) -> Optional[int]:
        """
        ML-11: Decide whether to attempt coerced transfer

        Expected gain vs expected cost. ML-6 bounded cognition: attacker can only observe
        the target's social network breadth (reciprocity partners), not who will help.
        The stronger the target's social network -> higher expected defensive resistance -> deterrent effect.
        """
        if not self.alive or self.resources < coercion_cost:
            return None

        best_target = None
        best_expected_gain = 0.0

        for target in possible_targets:
            if target.id == self.id or not target.alive:
                continue

            my_power = self.power

            # Target's effective defensive power = individual power + social network deterrence
            # ML-6 bounded cognition: attacker sees "how many reciprocity partners this agent has"
            # Natural inference: more partners -> more likely to receive help -> more formidable
            # Social power = sum of positive reciprocity balances toward target from others * decay
            # Decay from ML-1 (helping costs resources) and ML-2a (labor is bounded)
            social_defense = 0.0
            for a in possible_targets:
                if a.id != target.id and a.alive:
                    social_defense += max(0, a.reciprocity_ledger.get(target.id, 0.0))
            # Decay 0.1: consistent with ML-10 trust formula alpha -- not every partner will help
            their_power = target.power + social_defense * 0.1

            success_prob = 1.0 / (1.0 + np.exp(-(my_power - their_power * 1.2) * 0.5))

            expected_transfer = success_prob * min(
                target.resources * 0.15,
                my_power * 0.5,
            )

            expected_cost = coercion_cost
            cross_tag_bonus = 1.3 if (target.tag != self.tag) else 1.0
            expected_gain = (expected_transfer - expected_cost) * cross_tag_bonus

            if expected_gain > best_expected_gain:
                best_expected_gain = expected_gain
                best_target = target.id

        # Survival mode: lower threshold (more willing to take risks)
        # Profit mode: higher threshold (better alternatives exist)
        threshold = 0.0 if self.behavior_mode == 'survival' else coercion_cost * 2

        if best_expected_gain > threshold and best_target is not None:
            return best_target
        return None

    def _danger_zone(self) -> float:
        """Resource danger zone threshold"""
        return 3.0 * (1 + self.risk_tolerance)

    def subsistence_wage(self) -> float:
        """Minimum acceptable wage"""
        return 1.0 + self.risk_tolerance

    def decide_borrow(
        self,
        market_interest_rate: float,
        subsistence_cost: float,
        max_debt_ratio: float,
    ) -> float:
        """
        Decide how much to borrow (first principles: ML-5 survival baseline driven)

        The only legitimate motive for borrowing: current resources insufficient
        for survival or investment. Not speculation, but "must survive now" or
        "borrowing can produce more output".
        """
        if not self.alive:
            return 0.0

        net_wealth = max(0, self.resources + self.tools) - self.debt
        max_debt = max_debt_ratio * max(0.1, net_wealth + self.resources)

        if self.debt >= max_debt:
            return 0.0

        borrowable = max(0, max_debt - self.debt)

        # Survival motive: borrow up to safety line when resources are below it
        safety_buffer = subsistence_cost * 3
        if self.resources < safety_buffer:
            need = safety_buffer - self.resources
            return min(need, borrowable)

        # Investment motive: when investment return (experience-augmented production efficiency)
        # > interest cost, borrowing amplifies returns. Only profit mode considers this.
        # Effective skill represents production efficiency, tool bonus represents capital return
        expected_return = (self.effective_skill * (1 + 0.3 * self.tools) - 0.1) / self.resources if self.resources > 0 else 0
        if expected_return > market_interest_rate and self.behavior_mode == 'profit':
            # Borrowing amount positively correlated with risk preference, negatively with resource level (diminishing marginal utility)
            desired = borrowable * self.risk_tolerance * 0.3
            return desired

        return 0.0

    def decide_lend(
        self,
        market_interest_rate: float,
        subsistence_cost: float,
    ) -> float:
        """
        Decide how much to lend (first principles: intertemporal allocation of surplus resources)

        Lending = transferring current idle resources to the future (in exchange for interest).
        Only consider lending when surplus covers the survival safety net.
        """
        if not self.alive:
            return 0.0

        # Surplus = resources - debt service buffer - survival safety net
        safety_buffer = subsistence_cost * 3
        surplus = max(0, self.resources - safety_buffer - self.debt * 0.1)

        if surplus <= 0:
            return 0.0

        # Lending ratio positively correlated with interest rate (higher rate -> more willing to lend)
        # Positively correlated with risk preference (higher risk tolerance -> more willing to bear uncertainty)
        lend_ratio = min(0.5, market_interest_rate * 3.0) * self.risk_tolerance
        return surplus * lend_ratio

    def update_credit_score(self, repaid: bool) -> None:
        """Update credit score based on repayment/default history"""
        if repaid:
            self.credit_score = min(1.0, self.credit_score + 0.05)
            self.default_count = 0
        else:
            self.credit_score = max(0.0, self.credit_score - 0.15)
            self.default_count += 1
