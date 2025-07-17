"""
Domain models for electricity price data.
"""

from pydantic import BaseModel
from typing import List, Dict


class PriceRecord(BaseModel):
    """Model for individual price record."""
    timestamp: str
    date: str
    hour: int
    price_eur: float  # wholesale price in EUR/MWh
    price_raw: str
    consumer_price_cents_kwh: float = None  # consumer price in euro cents per kWh
    price_category: str = None  # "cheap", "regular", "expensive", "extremely_expensive"


class ExtremePrice(BaseModel):
    """Model for extreme price records."""
    timestamp: str
    date: str
    hour: int
    price_eur: float  # wholesale price in EUR/MWh
    price_raw: str
    consumer_price_cents_kwh: float  # consumer price in euro cents per kWh
    price_type: str  # "highest" or "lowest"


class HourlyConsumption(BaseModel):
    """Model for hourly consumption breakdown."""
    timestamp: str
    date: str
    hour: int
    consumption_kwh: float
    price_cents_kwh: float
    cost_euros: float
    price_category: str


class ConsumptionCostAnalysis(BaseModel):
    """Model for consumption cost analysis response."""
    total_consumption_kwh: float
    total_cost_euros: float
    average_price_cents_kwh: float
    days_analyzed: int
    period_start: str
    period_end: str
    hourly_breakdown: List[HourlyConsumption]
    cost_by_category: Dict[str, float]  # cost breakdown by price category
    # consumption breakdown by price category
    consumption_by_category: Dict[str, float]
    savings_analysis: Dict[str, float]  # potential savings analysis
    statistics: Dict[str, float]  # additional statistics
