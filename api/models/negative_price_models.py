"""
Models for negative price analysis.
"""

from pydantic import BaseModel
from typing import List


class NegativePriceStats(BaseModel):
    """Model for negative price analysis statistics."""
    total_negative: int
    lowest_price: float
    avg_negative_price: float
    first_negative_date: str
    last_negative_date: str


class HourlyNegativeStats(BaseModel):
    """Model for hourly negative price statistics."""
    hour: int
    negative_count: int
    avg_negative_price: float


class MonthlyNegativeStats(BaseModel):
    """Model for monthly negative price statistics."""
    month: str
    negative_count: int
    avg_negative_price: float


class NegativePriceAnalysis(BaseModel):
    """Model for complete negative price analysis."""
    overall_stats: NegativePriceStats
    by_hour: List[HourlyNegativeStats]
    by_month: List[MonthlyNegativeStats]
