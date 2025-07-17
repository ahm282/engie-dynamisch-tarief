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
    price_eur: float
    price_raw: str
    price_type: str  # "highest" or "lowest"
