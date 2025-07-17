"""
Models for expensive price analysis.
"""

from pydantic import BaseModel
from typing import List


class ExpensivePriceStats(BaseModel):
    """Model for expensive price analysis statistics."""
    total_expensive: int
    highest_price: float
    avg_expensive_price: float
    first_expensive_date: str
    last_expensive_date: str
    threshold_eur: float


class HourlyExpensiveStats(BaseModel):
    """Model for hourly expensive price statistics."""
    hour: int
    expensive_count: int
    avg_expensive_price: float


class MonthlyExpensiveStats(BaseModel):
    """Model for monthly expensive price statistics."""
    month: str
    expensive_count: int
    avg_expensive_price: float


class ExpensivePriceAnalysis(BaseModel):
    """Model for complete expensive price analysis."""
    overall_stats: ExpensivePriceStats
    by_hour: List[HourlyExpensiveStats]
    by_month: List[MonthlyExpensiveStats]
