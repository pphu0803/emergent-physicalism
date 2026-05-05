"""
Economy

Hosts all agents and institutional rules, manages per-timestep execution flow.
Economic structure (markets, employment relationships, capital concentration) emerges from here.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .agent import MicroIndividual
from .meta_laws import MetaLawEnforcer
from .sectors import Sector, create_default_sectors


@dataclass
class EconomyState:
    """Snapshot of economic system state -- for analyzing emergent quantities"""
    tick: int = 0
    population: int = 0
    total_resources: float = 0.0
    gini_coefficient: float = 0.0
    top1_share: float = 0.0
    top10_share: float = 0.0
    bottom50_share: float = 0.0
    mean_wealth: float = 0.0
    median_wealth: float = 0.0
    num_employers: int = 0
    avg_tools: float = 0.0
    avg_experience: float = 0.0
    avg_effective_skill: float = 0.0
    # Multi-sector indicators
    sector_profit_rates: Dict[str, float] = field(default_factory=dict)
    sector_producers: Dict[str, int] = field(default_factory=dict)
    profit_rate_std: float = 0.0  # Cross-sector profit rate std (core HL-2 indicator)
    total_production: float = 0.0
    total_trades: float = 0.0
    total_deaths: int = 0
    hhi: float = 0.0  # Herfindahl index (employer market concentration)
    # ML-9: Demographic indicators
    births_this_tick: int = 0
    deaths_this_tick: int = 0
    total_births: int = 0
    # ML-10: Credit/debt indicators
    total_credit: float = 0.0  # Total outstanding loans
    avg_debt: float = 0.0  # Average debt
    avg_credit_score: float = 0.0  # Average credit score
    default_rate: float = 0.0  # Default rate this tick
    credit_gini: float = 0.0  # Credit concentration
    total_defaults: int = 0  # Cumulative defaults
    total_interest_paid: float = 0.0  # Cumulative interest paid
    # ML-10/11: Trust and coercion indicators
    coercion_attempts: int = 0  # Coercion attempts this tick
    coercion_successes: int = 0  # Coercion successes this tick
    avg_reciprocity_size: float = 0.0  # Average reciprocity ledger size
    ingroup_trade_ratio: float = 0.0  # Same-tag trade ratio
    cross_tag_coercion_ratio: float = 0.0
    coercion_victim_debuff_count: int = 0


class Economy:
    """Economy -- the stage where all agents interact"""

    def __init__(
        self,
        num_agents: int = 100,
        initial_endowment: float = 10.0,
        meta_law_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.tick = 0
        self.agents: List[MicroIndividual] = []
        self.rules: list = []  # Institutional rule list

        # Meta-law enforcer
        ml_config = meta_law_config or {}
        self.meta_laws = MetaLawEnforcer(**ml_config)

        # Market state
        self.goods_market: List[Dict[str, Any]] = []  # Goods supply
        self.labor_market: List[Dict[str, Any]] = []   # Labor supply
        self.current_market_price: float = 1.0  # Market price

        # Multi-sector
        self.sectors = create_default_sectors()

        # Statistics
        self.state_history: List[EconomyState] = []
        self.death_log: List[Dict[str, Any]] = []
        self.birth_log: List[Dict[str, Any]] = []
        self.production_log: List[float] = []

        # Per-tick counters (ML-9)
        self._tick_births: int = 0
        self._tick_deaths: int = 0
        self._total_births: int = 0

        # ML-10: Credit/debt counters
        self._tick_defaults: int = 0
        self._total_defaults: int = 0
        self._tick_interest_paid: float = 0.0
        self._total_interest_paid: float = 0.0
        self._market_interest_rate: float = 0.02  # Current market interest rate

        # ML-10/11: Trust and coercion counters
        self._tick_coercion_attempts: int = 0
        self._tick_coercion_successes: int = 0
        self._tick_cross_tag_coercion: int = 0
        self._tick_ingroup_trade_ratio: float = 0.0

        # Initialize agents
        MicroIndividual._id_counter = 0
        num_tags = ml_config.get('num_tags', 0) if ml_config else 0
        for _ in range(num_agents):
            # Heterogeneous initial endowments (but not extreme)
            endowment = initial_endowment * np.random.lognormal(0, 0.3)
            endowment = max(2.0, min(endowment, 50.0))
            tag = np.random.randint(0, num_tags) if num_tags > 0 else 0
            agent = MicroIndividual(initial_resources=endowment, tag=tag)
            self.agents.append(agent)

    def add_rule(self, rule) -> None:
        """Add an institutional rule"""
        self.rules.append(rule)

    def total_resources(self) -> float:
        """ML-1: Compute total system resources"""
        return sum(a.resources + a.tools for a in self.agents if a.alive)

    def record_death(self, agent: MicroIndividual) -> None:
        """Record agent death"""
        self._tick_deaths += 1
        self.death_log.append({
            'tick': self.tick,
            'agent_id': agent.id,
            'resources': agent.resources,
            'skill': agent.skill,
            'was_employer': agent.is_employer,
            'initial_resources': getattr(agent, 'initial_resources', agent.resources),
        })

    def record_birth(self, child: MicroIndividual, parent_id: int) -> None:
        """Record new agent birth (ML-9)"""
        self._tick_births += 1
        self._total_births += 1
        self.birth_log.append({
            'tick': self.tick,
            'child_id': child.id,
            'parent_id': parent_id,
            'child_skill': child.skill,
            'child_resources': child.resources,
        })

    def _resolve_goods_market(self) -> None:
        """Goods market trade matching"""
        alive_agents = [a for a in self.agents if a.alive]

        # Each agent attempts production
        total_production = 0.0
        for agent in alive_agents:
            output = agent.produce(
                self.meta_laws.natural_inflow_rate / len(alive_agents) if alive_agents else 0,
                learning_rate=self.meta_laws.learning_rate,
            )
            total_production += output
            agent.goods = getattr(agent, 'goods', 0) + output

        self.production_log.append(total_production)

        # Market matching (simplified: random pairwise trading)
        np.random.shuffle(alive_agents)
        trade_count = 0
        ingroup_trades = 0  # Same-tag trade count

        for i in range(len(alive_agents)):
            for j in range(i + 1, len(alive_agents)):
                a, b = alive_agents[i], alive_agents[j]
                if not a.alive or not b.alive:
                    continue

                a_goods = getattr(a, 'goods', 0)
                b_goods = getattr(b, 'goods', 0)

                if a_goods <= 0 and b_goods <= 0:
                    continue

                # Negotiate trade based on price beliefs
                mid_price = (a.price_belief + b.price_belief) / 2
                self.current_market_price = 0.9 * self.current_market_price + 0.1 * mid_price

                # ML-10 trust channel: tag affects trust evaluation
                same_tag = (a.tag == b.tag)
                a_trust = a.memory.trust_score(
                    b.id, same_tag=same_tag, tick=self.tick,
                    reciprocity_balance=a.reciprocity_ledger.get(b.id, 0.0))
                a_action, a_qty = a.decide_trade(a_goods, self.current_market_price, a_trust)

                b_trust = b.memory.trust_score(
                    a.id, same_tag=same_tag, tick=self.tick,
                    reciprocity_balance=b.reciprocity_ledger.get(a.id, 0.0))
                b_action, b_qty = b.decide_trade(b_goods, self.current_market_price, b_trust)

                # Execute trade (one buys, one sells)
                trade_qty = 0
                trade_value = 0

                if a_action == 'sell' and b_action == 'buy' and a_goods > 0:
                    trade_qty = min(a_qty, b_qty, a_goods)
                    trade_value = trade_qty * self.current_market_price
                elif b_action == 'sell' and a_action == 'buy' and b_goods > 0:
                    trade_qty = min(a_qty, b_qty, b_goods)
                    trade_value = trade_qty * self.current_market_price

                if trade_qty > 0 and trade_value > 0:
                    # Execute material transfer (ML-1: conservation)
                    seller = a if a_action == 'sell' else b
                    buyer = a if a_action == 'buy' else b

                    seller.goods = getattr(seller, 'goods', 0) - trade_qty
                    buyer.goods = getattr(buyer, 'goods', 0) + trade_qty
                    seller.resources += trade_value
                    buyer.resources -= trade_value

                    # Record trade (for ML-3 relaxation: time irreversibility loosening)
                    self.meta_laws._last_trade_log.append({
                        'seller_id': seller.id,
                        'buyer_id': buyer.id,
                        'qty': trade_qty,
                        'value': trade_value,
                        'tick': self.tick,
                    })
                    if len(self.meta_laws._last_trade_log) > 1000:
                        self.meta_laws._last_trade_log = self.meta_laws._last_trade_log[-500:]

                    # Record interaction memory
                    seller.memory.add_interaction(buyer.id, 'sell', trade_value, self.tick)
                    buyer.memory.add_interaction(seller.id, 'buy', trade_qty, self.tick)

                    # ML-10: Update reciprocity ledger
                    seller.update_reciprocity(buyer.id, trade_qty, trade_value)
                    buyer.update_reciprocity(seller.id, trade_value, trade_qty)

                    # Update price beliefs
                    seller.update_price_belief(self.current_market_price)
                    buyer.update_price_belief(self.current_market_price)

                    trade_count += 1
                    if seller.tag == buyer.tag:
                        ingroup_trades += 1

        self._tick_ingroup_trade_ratio = ingroup_trades / max(1, trade_count)
        return trade_count

    def _resolve_labor_market(self) -> None:
        """Labor market -- employment relationship formation"""
        alive_agents = [a for a in self.agents if a.alive and not a.is_employer]
        employer_candidates = [a for a in self.agents if a.alive and a.is_employer]

        for worker in alive_agents:
            if worker.employer_id is not None:
                continue

            # Find an employer
            np.random.shuffle(employer_candidates)
            for employer in employer_candidates:
                if not employer.alive or employer.resources < 3:
                    continue

                wage = np.random.uniform(0.5, 2.0)
                if worker.decide_employment(wage):
                    trust = employer.memory.trust_score(worker.id)
                    if employer.decide_hire(worker.id, trust):
                        worker.employer_id = employer.id
                        employer.employees.append(worker.id)
                        # Employer pays wage
                        employer.resources -= wage
                        worker.resources += wage
                        worker.memory.add_interaction(employer.id, 'employed', wage, self.tick)
                        break

    def _resolve_sector_choice(self) -> None:
        """Agent sector choice"""
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return

        # Collect sector profit rate signals (agents only see local signals -- ML-6)
        sector_profits = {sid: s.profit_rate for sid, s in self.sectors.items()}
        sector_thresholds = {sid: s.entry_threshold for sid, s in self.sectors.items()}

        for agent in alive:
            agent.choose_sector(sector_profits, sector_thresholds, self.meta_laws.subsistence_cost)

    def _resolve_sector_production(self) -> None:
        """Sector-level production -- each agent produces in its current sector"""
        alive = [a for a in self.agents if a.alive]
        total_production = 0.0

        # Reset sector statistics
        for sector in self.sectors.values():
            sector.reset_stats()

        # Sector output and demand
        sector_outputs: Dict[str, float] = {sid: 0.0 for sid in self.sectors}
        sector_demands: Dict[str, float] = {sid: 0.0 for sid in self.sectors}

        for agent in alive:
            if not agent.alive or agent.labor_remaining <= 0:
                continue

            sector = self.sectors.get(agent.sector)
            if sector is None:
                continue

            # Check entry barrier
            if agent.tools < sector.entry_threshold:
                # Doesn't meet barrier, fall back to necessities
                agent.sector = 'necessities'
                sector = self.sectors['necessities']

            labor_input = min(agent.labor_remaining, 1.0)
            agent.labor_remaining -= labor_input

            # Learning by doing
            agent._update_experience(labor_input, learning_rate=self.meta_laws.learning_rate)

            # Sector-specific output formula
            tool_bonus = sector.capital_intensity * agent.tools
            base_output = agent.effective_skill * sector.labor_intensity * (1 + tool_bonus) * labor_input
            noise = np.random.normal(0, 0.1)
            output = max(0, base_output * (1 + noise))

            # Production cost
            cost = sector.production_cost * labor_input + 0.1 * labor_input
            agent.resources -= cost

            # Output as goods (priced at sector price)
            revenue = output * sector.current_price
            agent.goods = getattr(agent, 'goods', 0) + output

            # Update sector statistics
            sector.total_output += output
            sector.total_revenue += revenue
            sector.total_cost += cost + max(0, agent.resources * 0.01)  # Opportunity cost
            sector.num_producers += 1
            sector_outputs[agent.sector] += output

            total_production += output

        self.production_log.append(total_production)

        # Compute sector demand (based on alive agents' wealth level)
        mean_wealth = np.mean([a.resources for a in alive]) if alive else 0
        for sid, sector in self.sectors.items():
            # Demand = base demand * (1 + income elasticity * (mean wealth - survival line))
            wealth_effect = sector.demand_income_elasticity * max(0, mean_wealth - 5) * 0.1
            sector_demands[sid] = sector.demand_base * (1 + wealth_effect) * len(alive)

        # Update sector prices
        for sid in self.sectors:
            supply = sector_outputs.get(sid, 0)
            demand = sector_demands.get(sid, 0)
            self.sectors[sid].update_price(supply, demand)
            self.sectors[sid].compute_profit_rate()

    def _resolve_investment(self) -> None:
        """Agent investment in tools/capital accumulation"""
        for agent in self.agents:
            if not agent.alive:
                continue
            invest_amount = agent.decide_invest_in_tools()
            if invest_amount > 0 and agent.resources >= invest_amount:
                agent.resources -= invest_amount
                agent.tools += invest_amount * 0.3  # Investment conversion rate (low rate creates differentiation)

    def _enforce_tool_depreciation(self, depreciation_rate: float = None) -> None:
        """Tool depreciation -- physical reality: tools wear out, become obsolete"""
        if depreciation_rate is None:
            depreciation_rate = self.meta_laws.depreciation_rate
        for agent in self.agents:
            if not agent.alive or agent.tools <= 0:
                continue
            depreciation = agent.tools * depreciation_rate
            agent.tools = max(0, agent.tools - depreciation)

    def _resolve_credit_market(self) -> None:
        """
        ML-10: Credit/debt market

        First principles:
        Credit = intertemporal resource transfer promise. Based on ML-3 (time irreversibility),
        individuals can exchange current resources for future resources.
        Interest is the time cost of waiting.

        Design principles:
        - No system-level regulation (no central bank, no interest rate control)
        - Credit tightening/expansion emerges entirely from individual agent decisions
        - Lenders independently judge risk based on ML-6 (bounded information)
        - Default is a natural result of information asymmetry (ML-6) + randomness (ML-7)

        Physical constraints (ML-1 conservation):
        - On lending: lender.resources -= amount, borrower.resources += amount
        - On interest: borrower.resources -= interest, lender.resources += interest
        - On default: matter is not created or destroyed, just transfer fails
        """
        if not self.meta_laws.credit_enabled:
            return

        alive = [a for a in self.agents if a.alive]
        if len(alive) < 2:
            return

        ml = self.meta_laws
        self._tick_defaults = 0
        self._tick_interest_paid = 0.0

        # === 1. Interest payments ===
        for agent in alive:
            if agent.debt <= 0:
                continue

            interest_this_tick = 0.0
            loans_to_remove = []

            for i, loan in enumerate(agent.loans):
                loan_interest = loan['remaining'] * loan['interest_rate']
                interest_this_tick += loan_interest

                # Maturity check
                if self.tick >= loan['maturity_tick']:
                    repayment = loan['remaining'] + loan_interest
                    if agent.resources >= repayment:
                        # Normal repayment (ML-1 conservation)
                        agent.resources -= repayment
                        lender = next((a for a in alive if a.id == loan['lender_id']), None)
                        if lender and lender.alive:
                            lender.resources += repayment
                        agent.update_credit_score(repaid=True)
                    else:
                        # Default: cannot fully repay
                        self._tick_defaults += 1
                        self._total_defaults += 1
                        # Partial recovery (proportional to borrower's remaining resources)
                        lender = next((a for a in alive if a.id == loan['lender_id']), None)
                        if lender and lender.alive:
                            recovery = min(agent.resources * 0.5, loan['remaining'])
                            lender.resources += recovery
                            agent.resources -= recovery
                        agent.debt -= loan['remaining']
                        agent.debt = max(0, agent.debt)
                        agent.update_credit_score(repaid=False)
                    loans_to_remove.append(i)

            for i in sorted(loans_to_remove, reverse=True):
                agent.loans.pop(i)

            # Interest payment (ML-1 conservation: interest transfers from borrower to lender)
            if interest_this_tick > 0 and agent.debt > 0:
                actual_interest = min(interest_this_tick, agent.resources)
                agent.resources -= actual_interest
                self._tick_interest_paid += actual_interest
                self._total_interest_paid += actual_interest

                total_remaining = sum(l['remaining'] for l in agent.loans)
                if total_remaining > 0:
                    for loan in agent.loans:
                        share = loan['remaining'] / total_remaining
                        interest_portion = actual_interest * share
                        lender = next((a for a in alive if a.id == loan['lender_id']), None)
                        if lender and lender.alive:
                            lender.resources += interest_portion

        # === 2. Lender death -> loans naturally expire ===
        # Lender death means no one to collect, debt relationship terminates.
        # This benefits the borrower (freely obtained previously borrowed resources),
        # but the lender loses all principal and interest.
        dead_lender_ids = set(a.id for a in self.agents if not a.alive)
        if dead_lender_ids:
            for agent in alive:
                loans_to_cancel = [
                    i for i, loan in enumerate(agent.loans)
                    if loan['lender_id'] in dead_lender_ids
                ]
                for i in sorted(loans_to_cancel, reverse=True):
                    cancelled_loan = agent.loans.pop(i)
                    agent.debt = max(0, agent.debt - cancelled_loan['remaining'])

        # === 3. Market interest rate determined by supply and demand (no system regulation) ===
        total_borrow_demand = 0.0
        total_lend_supply = 0.0

        for agent in alive:
            total_borrow_demand += agent.decide_borrow(
                self._market_interest_rate, ml.subsistence_cost, ml.max_debt_ratio
            )
            total_lend_supply += agent.decide_lend(
                self._market_interest_rate, ml.subsistence_cost
            )

        if total_lend_supply > 0:
            pressure = (total_borrow_demand - total_lend_supply) / total_lend_supply
            self._market_interest_rate += pressure * 0.002

        self._market_interest_rate = np.clip(
            self._market_interest_rate, 0.005, ml.max_interest_rate
        )

        # === 4. Credit matching (individual agent decisions, no system intervention) ===
        borrowers = []
        lenders = []

        for agent in alive:
            borrow_amt = agent.decide_borrow(
                self._market_interest_rate, ml.subsistence_cost, ml.max_debt_ratio
            )
            if borrow_amt > 0.05:
                borrowers.append((agent, borrow_amt))

            lend_amt = agent.decide_lend(
                self._market_interest_rate, ml.subsistence_cost
            )
            if lend_amt > 0.05 and agent.resources > lend_amt + ml.subsistence_cost * 2:
                lenders.append((agent, lend_amt))

        # Sort by demand (resource-poor first) and by credit score (high credit first)
        borrowers.sort(key=lambda x: x[0].resources)
        lenders.sort(key=lambda x: x[0].credit_score, reverse=True)

        lender_funds = {l[0].id: l[1] for l in lenders}
        lender_map = {l[0].id: l[0] for l in lenders}

        for borrower, borrow_amt in borrowers:
            # Debt ceiling constraint (ML-10 physical constraint)
            net_wealth = max(0.1, borrower.resources + borrower.tools) - borrower.debt
            effective_max = ml.max_debt_ratio * max(0.1, net_wealth + borrower.resources)
            borrow_amt = min(borrow_amt, max(0, effective_max - borrower.debt))

            if borrow_amt <= 0:
                continue

            remaining = borrow_amt
            matched_lenders = list(lender_map.values())
            np.random.shuffle(matched_lenders)

            for lender in matched_lenders:
                if remaining <= 0.01:
                    break
                if lender.id == borrower.id:
                    continue

                available = lender_funds.get(lender.id, 0)
                if available <= 0:
                    continue

                # Lender decides lending amount based on borrower credit score
                # This embodies ML-6 (bounded information): lender can only judge from observable credit signals
                lend_amt = min(available, remaining)
                lend_amt *= borrower.credit_score

                if lend_amt < 0.01:
                    continue

                # Interest rate = market rate * individual risk adjustment
                # Low credit -> high risk premium (lender demands higher return)
                interest_rate = self._market_interest_rate * (1.0 + 0.3 * (1 - borrower.credit_score))
                interest_rate = min(interest_rate, ml.max_interest_rate)

                loan_duration = np.random.randint(ml.loan_duration_min, ml.loan_duration_max + 1)

                # Execute lending (ML-1 conservation)
                lender.resources -= lend_amt
                borrower.resources += lend_amt
                borrower.debt += lend_amt
                lender_funds[lender.id] = available - lend_amt

                # Record interaction (ML-6 memory)
                lender.memory.add_interaction(borrower.id, 'lend', 0, self.tick)
                borrower.memory.add_interaction(lender.id, 'borrow', lend_amt, self.tick)

                borrower.loans.append({
                    'lender_id': lender.id,
                    'principal': lend_amt,
                    'interest_rate': interest_rate,
                    'remaining': lend_amt,
                    'maturity_tick': self.tick + loan_duration,
                })

                remaining -= lend_amt

    def _resolve_coercion(self) -> None:
        """
        ML-11: Coercion channel

        Agents can attempt involuntary resource transfers against other agents.
        Success depends on relative power, both sides incur costs.
        Institutional rules can suppress coercion by raising costs or banning it.
        """
        if not self.meta_laws.coercion_enabled:
            return

        alive = [a for a in self.agents if a.alive]
        if len(alive) < 2:
            return

        ml = self.meta_laws
        coercion_results = []  # (attacker, target_id)

        for agent in alive:
            if not agent.alive or agent.resources < ml.coercion_cost:
                continue

            target_id = agent.decide_coercion(
                alive, ml.subsistence_cost, ml.coercion_cost
            )
            if target_id is not None:
                coercion_results.append((agent, target_id))
                self._tick_coercion_attempts += 1

        # Execute coerced transfers
        for attacker, target_id in coercion_results:
            target = next((a for a in alive if a.id == target_id), None)
            if not target or not target.alive:
                continue

            my_power = attacker.power
            their_power = target.power
            success = my_power > their_power * ml.defense_bonus

            # Attack cost (ML-1 conservation: resource consumption)
            attacker.resources -= ml.coercion_cost

            if success:
                transfer = min(
                    target.resources * ml.coercion_transfer_rate,
                    my_power * 0.5,
                )
                target.resources -= transfer
                attacker.resources += transfer

                # Record reciprocity ledger (negative interaction)
                attacker.update_reciprocity(target.id, transfer, 0)
                target.update_reciprocity(attacker.id, 0, transfer)

                # Record interaction memory
                attacker.memory.add_interaction(target.id, 'coerce', transfer, self.tick)
                target.memory.add_interaction(attacker.id, 'coerced', -transfer, self.tick)

                # Coercion toxicity: victim's productivity is weakened
                target._coercion_debuff_ticks = 2

                # Cross-tag coercion tracking
                if attacker.tag != target.tag:
                    self._tick_cross_tag_coercion += 1

                self._tick_coercion_successes += 1
            else:
                # Failure: attacker only loses cost
                attacker.memory.add_interaction(target.id, 'coerce_fail', -ml.coercion_cost, self.tick)
                target.memory.add_interaction(attacker.id, 'defend', 0, self.tick)

    def _update_employer_status(self, top_percentile: float = 0.20, min_tool_threshold: float = 2.0) -> None:
        """
        Dynamically update employer status -- relative threshold

        Employers are not determined by absolute tool level, but by relative position.
        Only agents whose tool level ranks in the top top_percentile can become employers.

        Psychological basis: status is not "what I have" but "how much more I have than others."
        Social comparison instinct makes relative position the key determinant of social roles.

        Also maintains a minimum absolute threshold (min_tool_threshold) to avoid incorrectly
        marking agents with very low tool levels as employers in extreme cases.
        """
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return

        # Collect all agents' tool levels
        tool_levels = [(a.id, a.tools) for a in alive]
        tool_levels.sort(key=lambda x: x[1], reverse=True)

        # Compute dynamic threshold: lowest tool level among top top_percentile
        n = len(alive)
        n_employers = max(1, int(n * top_percentile))
        dynamic_threshold = tool_levels[n_employers - 1][1] if n_employers <= len(tool_levels) else 0

        # Effective threshold = max(dynamic threshold, absolute minimum barrier)
        effective_threshold = max(dynamic_threshold, min_tool_threshold)

        # Update status
        for agent in alive:
            if agent.tools >= effective_threshold:
                agent.is_employer = True
            else:
                agent.is_employer = False
                # Downgraded employers lose all employees
                if agent.employees:
                    for emp_id in agent.employees:
                        emp = next((a for a in alive if a.id == emp_id), None)
                        if emp:
                            emp.employer_id = None
                    agent.employees = []

    def _enforce_employer_production(self) -> None:
        """Employers produce through employees (implementation of capital-labor asymmetry)"""
        for agent in self.agents:
            if not agent.alive or not agent.is_employer:
                continue

            employees = [a for a in self.agents if a.alive and a.employer_id == agent.id]
            for emp in employees:
                # Employee labor output goes to employer (core of employment relationship)
                emp_labor = emp.labor_remaining * 0.8  # Employee contributes 80% of labor
                emp.labor_remaining -= emp_labor

                tool_bonus = 0.3 * agent.tools
                output = emp.effective_skill * (1 + tool_bonus) * emp_labor
                noise = np.random.normal(0, 0.1)
                output = max(0, output * (1 + noise))

                agent.goods = getattr(agent, 'goods', 0) + output
                emp.memory.add_interaction(agent.id, 'employed', 0, self.tick)

    def _compute_state(self) -> EconomyState:
        """Compute all emergent quantities of the current economic state"""
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return EconomyState(tick=self.tick, population=0)

        # Wealth = resources + tools, clipped to non-negative (agents in crisis may have negative resources)
        wealths = np.array([max(0, a.resources + a.tools) for a in alive])
        total = wealths.sum()
        n = len(wealths)

        if total <= 0 or n < 2:
            return EconomyState(tick=self.tick, population=n)

        # Gini coefficient (standard formula, guaranteed in [0,1] range)
        sorted_w = np.sort(wealths)
        cumw = np.cumsum(sorted_w)
        gini = (n + 1 - 2 * np.sum(cumw) / cumw[-1]) / n
        gini = max(0.0, min(1.0, gini))  # Clip to [0,1]

        # Wealth shares (based on non-negative wealth)
        sorted_desc = np.sort(wealths)[::-1]
        top1_share = sorted_desc[:max(1, n // 100)].sum() / total
        top10_share = sorted_desc[:max(1, n // 10)].sum() / total
        bottom50_share = sorted_desc[max(1, n // 2):].sum() / total

        # Employer market concentration (HHI)
        employers = [a for a in alive if a.is_employer]
        if employers:
            employer_sizes = np.array([a.tools for a in employers])
            hhi = ((employer_sizes / employer_sizes.sum()) ** 2).sum() if employer_sizes.sum() > 0 else 0
        else:
            hhi = 0.0

        # Multi-sector indicators
        sector_profits = {sid: s.profit_rate for sid, s in self.sectors.items()}
        sector_producers = {sid: s.num_producers for sid, s in self.sectors.items()}
        profit_rates = [s.profit_rate for s in self.sectors.values()]
        profit_rate_std = np.std(profit_rates) if len(profit_rates) > 1 else 0.0

        # ML-10: Credit/debt indicators
        debts = np.array([a.debt for a in alive])
        total_credit = float(debts.sum())
        avg_debt = float(debts.mean()) if n > 0 else 0.0
        avg_credit_score = float(np.mean([a.credit_score for a in alive])) if alive else 0.0
        default_rate = self._tick_defaults / n if n > 0 else 0.0

        # Credit concentration (Gini of debt distribution among debtors)
        debtors = debts[debts > 0]
        if len(debtors) > 1:
            sorted_d = np.sort(debtors)
            cumd = np.cumsum(sorted_d)
            credit_gini = (len(debtors) + 1 - 2 * np.sum(cumd) / cumd[-1]) / len(debtors)
            credit_gini = max(0.0, min(1.0, credit_gini))
        else:
            credit_gini = 0.0

        return EconomyState(
            tick=self.tick,
            population=n,
            total_resources=total,
            gini_coefficient=gini,
            top1_share=top1_share,
            top10_share=top10_share,
            bottom50_share=bottom50_share,
            mean_wealth=wealths.mean(),
            median_wealth=np.median(wealths),
            num_employers=len(employers),
            avg_tools=np.mean([a.tools for a in alive]),
            avg_experience=np.mean([a.production_experience for a in alive]),
            avg_effective_skill=np.mean([a.effective_skill for a in alive]),
            total_production=sum(self.production_log[-2:]) if len(self.production_log) >= 2 else (self.production_log[-1] if self.production_log else 0),
            total_trades=0,
            total_deaths=len(self.death_log),
            hhi=hhi,
            sector_profit_rates=sector_profits,
            sector_producers=sector_producers,
            profit_rate_std=profit_rate_std,
            births_this_tick=self._tick_births,
            deaths_this_tick=self._tick_deaths,
            total_births=self._total_births,
            total_credit=total_credit,
            avg_debt=avg_debt,
            avg_credit_score=avg_credit_score,
            default_rate=default_rate,
            credit_gini=credit_gini,
            total_defaults=self._total_defaults,
            total_interest_paid=self._total_interest_paid,
            coercion_attempts=self._tick_coercion_attempts,
            coercion_successes=self._tick_coercion_successes,
            avg_reciprocity_size=float(np.mean([len(a.reciprocity_ledger) for a in alive])),
            ingroup_trade_ratio=self._tick_ingroup_trade_ratio,
            cross_tag_coercion_ratio=(
                self._tick_cross_tag_coercion / max(1, self._tick_coercion_successes)
            ),
            coercion_victim_debuff_count=sum(1 for a in alive if a._coercion_debuff_ticks > 0),
        )

    def step(self) -> EconomyState:
        """Execute one complete simulation timestep"""
        # 0. Reset per-tick counters
        self._tick_births = 0
        self._tick_deaths = 0
        self._tick_defaults = 0
        self._tick_interest_paid = 0.0
        self._tick_coercion_attempts = 0
        self._tick_coercion_successes = 0
        self._tick_cross_tag_coercion = 0

        # 1. Enforce meta-laws
        self.meta_laws.enforce_survival_baseline(self)
        self.meta_laws.enforce_conservation(self)
        self.meta_laws.enforce_labor_conservation(self)

        # 2. Economic activity (emergence level)
        self._resolve_sector_choice()        # Sector choice
        self._resolve_sector_production()    # Sector-level production
        self._resolve_investment()           # Investment
        self._enforce_tool_depreciation()    # Tool depreciation (physical wear)
        self._resolve_credit_market()        # Legacy ML-10: credit/debt market (disabled by default)
        self._resolve_coercion()            # ML-11: coercion channel
        self._update_employer_status()       # Dynamic employer threshold (relative status)
        self._resolve_labor_market()          # Employment
        self._enforce_employer_production()   # Employer production
        self._resolve_goods_market()          # Goods trading

        # 3. Enforce institutional rules
        for rule in self.rules:
            rule.enforce(self)

        # 3.5 Demographic dynamics (ML-9: reproduction)
        self.meta_laws.enforce_demographic_dynamics(self)

        # 4. Cleanup and advance time
        self.meta_laws.enforce_time_irreversibility(self)

        # 5. Record state
        state = self._compute_state()
        self.state_history.append(state)

        return state
