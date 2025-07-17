"""
Statistics models for price analysis.
"""

from pydantic import BaseModel
from typing import List


class DailyStats(BaseModel):
    """Model for daily statistics."""
    date: str
    hours_count: int
    min_price: float
    max_price: float
    avg_price: float
    price_range: float
    negative_hours: int


class HourlyStats(BaseModel):
    """Model for hourly statistics."""
    hour: int
    total_occurrences: int
    min_price: float
    max_price: float
    avg_price: float
    negative_occurrences: int


class DatabaseStats(BaseModel):
    """Model for database statistics."""
    total_records: int
    date_range_start: str
    date_range_end: str
    min_price: float
    max_price: float
    avg_price: float
    negative_prices_count: int
