"""
Domain models for electricity price data.
"""

from pydantic import BaseModel
from typing import List


class PriceRecord(BaseModel):
    """Model for individual price record."""
    timestamp: str
    date: str
    hour: int
    price_eur: float
    price_raw: str


class ExtremePrice(BaseModel):
    """Model for extreme price records."""
    timestamp: str
    date: str
    hour: int
    price_eur: float  # wholesale price in EUR/MWh
    price_raw: str
    consumer_price_cents_kwh: float  # consumer price in euro cents per kWh
    price_type: str  # "highest" or "lowest"
