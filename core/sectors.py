"""
Production Sectors

Define multiple production sectors in the economy, each with different production characteristics.
Differences in sector profit rates are the prerequisite for HL-2 (profit rate equalization) testing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class Sector:
    """Production sector"""
    id: str
    name: str

    # Production characteristics
    capital_intensity: float = 0.0    # Tool bonus coefficient for output (0~1)
    labor_intensity: float = 1.0       # Skill weight for output
    entry_threshold: float = 0.0       # Entry barrier (minimum tools)
    production_cost: float = 0.05      # Production cost per unit labor

    # Demand characteristics
    demand_base: float = 1.0           # Base demand level
    demand_income_elasticity: float = 0.3  # Income elasticity of demand (0=necessity, 1=luxury)

    # Price
    base_price: float = 1.0            # Base price
    current_price: float = 1.0         # Current market price

    # State statistics (updated each tick)
    total_output: float = 0.0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    num_producers: int = 0
    profit_rate: float = 0.0           # Profit rate = (revenue - cost) / cost

    def __post_init__(self):
        self.price_history: list = []

    def update_price(self, supply: float, demand: float) -> None:
        """Update price based on supply-demand relationship (simple auction mechanism)"""
        if demand > 0 and supply > 0:
            price_pressure = demand / supply
            # Price fluctuates around base_price, driven by supply-demand ratio
            self.current_price = self.base_price * np.clip(price_pressure, 0.3, 3.0)
        self.price_history.append(self.current_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)

    def compute_profit_rate(self) -> float:
        """Compute sector profit rate"""
        if self.total_cost <= 0:
            self.profit_rate = 0.0
        else:
            self.profit_rate = (self.total_revenue - self.total_cost) / self.total_cost
        return self.profit_rate

    def reset_stats(self) -> None:
        """Reset statistics at the start of each tick"""
        self.total_output = 0.0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.num_producers = 0


# ============================================================
# Default sectors
# ============================================================

def create_default_sectors() -> Dict[str, Sector]:
    """Create 3 default production sectors"""
    sectors = {}

    sectors['necessities'] = Sector(
        id='necessities',
        name='Necessities',
        capital_intensity=0.05,     # Tools barely matter
        labor_intensity=1.0,
        entry_threshold=0.0,         # No barrier
        production_cost=0.03,       # Low cost
        demand_base=2.0,            # High base demand
        demand_income_elasticity=0.1,  # Very low elasticity (necessity)
        base_price=0.8,
    )

    sectors['capital_goods'] = Sector(
        id='capital_goods',
        name='Capital Goods',
        capital_intensity=0.4,      # Tools matter a lot
        labor_intensity=0.8,
        entry_threshold=2.0,         # Requires some tools
        production_cost=0.08,       # Medium cost
        demand_base=0.8,
        demand_income_elasticity=0.5,  # Medium elasticity
        base_price=1.5,
    )

    sectors['luxuries'] = Sector(
        id='luxuries',
        name='Luxuries',
        capital_intensity=0.25,     # Tools useful but not critical
        labor_intensity=0.6,
        entry_threshold=3.0,         # High barrier
        production_cost=0.12,       # High cost
        demand_base=0.3,
        demand_income_elasticity=1.2,  # High elasticity (luxury, bought when income is high)
        base_price=3.0,
    )

    return sectors
