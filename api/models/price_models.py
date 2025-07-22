"""
Domain models for electricity price data.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional


class PriceRecord(BaseModel):
    """Model for individual price record."""
    timestamp: str
    date: str
    hour: int
    price_eur: float  # wholesale price in EUR/MWh
    price_raw: str
    # consumer price in euro cents per kWh
    consumer_price_cents_kwh: Optional[float] = None
    # "cheap", "regular", "expensive", "extremely_expensive"
    price_category: Optional[str] = None

    # Weather data fields
    cloud_cover: Optional[float] = None  # cloud coverage percentage (0-100)
    temperature: Optional[float] = None  # temperature in Celsius
    solar_factor: Optional[float] = None  # solar production factor (0-1)


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
