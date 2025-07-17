"""
Aggregate repository that combines all specialized repositories.
Maintains backwards compatibility while delegating to specialized repositories.
"""

import pandas as pd
from typing import Optional, Dict, Any

from .base_repository import BaseRepository
from .price_data_repository import PriceDataRepository
from .statistics_repository import StatisticsRepository
from .extreme_price_repository import ExtremePriceRepository
from .negative_price_repository import NegativePriceRepository


class ElectricityPriceRepositoryAggregate(BaseRepository):
    """
    Aggregate repository that delegates to specialized repositories.
    Maintains the same interface as the original ElectricityPriceRepository.
    """

    def __init__(self):
        self.price_data_repo = PriceDataRepository()
        self.statistics_repo = StatisticsRepository()
        self.extreme_price_repo = ExtremePriceRepository()
        self.negative_price_repo = NegativePriceRepository()

    # Basic operations delegated to PriceDataRepository
    def find_all(self) -> pd.DataFrame:
        """Find all electricity price records."""
        return self.price_data_repo.find_all()

    def find_by_id(self, record_id: Any) -> Optional[pd.Series]:
        """Find price record by timestamp."""
        return self.price_data_repo.find_by_id(record_id)

    def count(self) -> int:
        """Count total price records."""
        return self.price_data_repo.count()

    def find_with_filters(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
        hour: Optional[int] = None,
        limit: int = 100,
        order: str = "desc"
    ) -> pd.DataFrame:
        """Find price records with various filters."""
        return self.price_data_repo.find_with_filters(
            start_date=start_date,
            end_date=end_date,
            days_back=days_back,
            hour=hour,
            limit=limit,
            order=order
        )

    def find_current_prices(self, hours: int = 24) -> pd.DataFrame:
        """Find most recent price data."""
        return self.price_data_repo.find_current_prices(hours)

    # Statistical operations delegated to StatisticsRepository
    def find_daily_stats(self, days_back: int = 30) -> pd.DataFrame:
        """Find daily statistics."""
        return self.statistics_repo.find_daily_stats(days_back)

    def find_hourly_stats(self) -> pd.DataFrame:
        """Find hourly statistics across all data."""
        return self.statistics_repo.find_hourly_stats()

    def find_database_stats(self) -> pd.Series:
        """Find overall database statistics."""
        return self.statistics_repo.find_database_stats()

    # Extreme price operations delegated to ExtremePriceRepository
    def find_extreme_prices(self, limit: int = 10) -> Dict[str, pd.DataFrame]:
        """Find highest and lowest prices."""
        return self.extreme_price_repo.find_extreme_prices(limit)

    # Negative price operations delegated to NegativePriceRepository
    def find_negative_price_stats(self) -> Dict[str, pd.DataFrame]:
        """Find negative price analysis data."""
        return self.negative_price_repo.find_negative_price_stats()
