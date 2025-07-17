"""
Models package for API data structures.
Imports all models for easy access.
"""

# Price models
from .price_models import PriceRecord, ExtremePrice, HourlyConsumption, ConsumptionCostAnalysis

# Statistics models
from .stats_models import DailyStats, HourlyStats, DatabaseStats

# Negative price models
from .negative_price_models import (
    NegativePriceStats,
    HourlyNegativeStats,
    MonthlyNegativeStats,
    NegativePriceAnalysis
)

# Expensive price models
from .expensive_price_models import (
    ExpensivePriceStats,
    HourlyExpensiveStats,
    MonthlyExpensiveStats,
    ExpensivePriceAnalysis
)

# Response models
from .response_models import CurrentPricesResponse, APIInfo, HealthResponse

__all__ = [
    # Price models
    "PriceRecord",
    "ExtremePrice",
    "HourlyConsumption",
    "ConsumptionCostAnalysis",

    # Statistics models
    "DailyStats",
    "HourlyStats",
    "DatabaseStats",

    # Negative price models
    "NegativePriceStats",
    "HourlyNegativeStats",
    "MonthlyNegativeStats",
    "NegativePriceAnalysis",

    # Expensive price models
    "ExpensivePriceStats",
    "HourlyExpensiveStats",
    "MonthlyExpensiveStats",
    "ExpensivePriceAnalysis",

    # Response models
    "CurrentPricesResponse",
    "APIInfo",
    "HealthResponse"
]
