"""
Services package for business logic layer.
Imports all services for easy access.
"""

# Base service
from .base_service import BaseService

# Individual services
from .price_service import PriceService
from .statistics_service import StatisticsService
from .extreme_analysis_service import ExtremeAnalysisService
from .negative_price_analysis_service import NegativePriceAnalysisService
from .expensive_price_analysis_service import ExpensivePriceAnalysisService

# Aggregate service for backwards compatibility
from ..repositories import ElectricityPriceRepositoryAggregate


class ElectricityPriceService:
    """
    Aggregate service that combines all electricity price services.
    Maintains backwards compatibility while using the new modular services.
    """

    def __init__(self, repository: ElectricityPriceRepositoryAggregate = None):
        """Initialize aggregate service with all sub-services."""
        repo = repository or ElectricityPriceRepositoryAggregate()

        # Initialize individual services with specialized repositories
        self.price_service = PriceService()
        self.statistics_service = StatisticsService()
        self.extreme_analysis_service = ExtremeAnalysisService()
        self.negative_price_service = NegativePriceAnalysisService()
        self.expensive_price_service = ExpensivePriceAnalysisService()

    # Delegate methods to appropriate services
    def get_database_stats(self):
        """Get overall database statistics."""
        return self.statistics_service.get_database_stats()

    def get_prices(self, start_date=None, end_date=None, days_back=None,
                   hour=None, limit=100, order="desc"):
        """Get electricity price data with various filters."""
        return self.price_service.get_prices(
            start_date=start_date,
            end_date=end_date,
            days_back=days_back,
            hour=hour,
            limit=limit,
            order=order
        )

    def get_daily_stats(self, days_back=30):
        """Get daily price statistics."""
        return self.statistics_service.get_daily_stats(days_back=days_back)

    def get_hourly_stats(self):
        """Get hourly price patterns across all data."""
        return self.statistics_service.get_hourly_stats()

    def get_extreme_prices(self, limit=10):
        """Get highest and lowest prices."""
        return self.extreme_analysis_service.get_extreme_prices(limit=limit)

    def get_negative_price_analysis(self):
        """Get comprehensive analysis of negative price occurrences."""
        return self.negative_price_service.get_negative_price_analysis()

    def get_expensive_price_analysis(self, threshold=1.500):
        """Get comprehensive analysis of expensive consumer price occurrences."""
        return self.expensive_price_service.get_expensive_price_analysis(threshold=threshold)

    def get_current_prices(self, hours=24):
        """Get the most recent price data."""
        return self.price_service.get_current_prices(hours=hours)


__all__ = [
    # Base service
    "BaseService",

    # Individual services
    "PriceService",
    "StatisticsService",
    "ExtremeAnalysisService",
    "NegativePriceAnalysisService",
    "ExpensivePriceAnalysisService",

    # Aggregate service
    "ElectricityPriceService"
]
