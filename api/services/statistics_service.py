"""
Service for statistics and analysis operations.
"""

from typing import List
from fastapi import HTTPException

from .base_service import BaseService
from ..repositories import StatisticsRepository
from ..models import DailyStats, HourlyStats, DatabaseStats


class StatisticsService(BaseService):
    """Service for statistics and analysis operations."""

    def __init__(self, repository: StatisticsRepository = None):
        """Initialize service with repository dependency injection."""
        super().__init__(repository or StatisticsRepository())

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for statistics queries."""
        days_back = kwargs.get('days_back')

        if days_back is not None and (days_back < 1 or days_back > 365):
            raise HTTPException(
                status_code=400,
                detail="Days back must be between 1 and 365"
            )

        return True

    def get_database_stats(self) -> DatabaseStats:
        """Get overall database statistics."""
        try:
            result = self.repository.find_database_stats()

            return DatabaseStats(
                total_records=int(result['total_records']),
                date_range_start=result['date_start'],
                date_range_end=result['date_end'],
                min_price=round(float(result['min_price']), 3),
                max_price=round(float(result['max_price']), 3),
                avg_price=round(float(result['avg_price']), 3),
                negative_prices_count=int(result['negative_count'])
            )

        except Exception as e:
            self.handle_exception(e, "Error retrieving database statistics")

    def get_daily_stats(self, days_back: int = 30) -> List[DailyStats]:
        """Get daily price statistics."""
        try:
            # Validate input
            self.validate_input(days_back=days_back)

            df = self.repository.find_daily_stats(days_back=days_back)

            # Convert to Pydantic models
            stats = []
            for _, row in df.iterrows():
                stats.append(DailyStats(
                    date=row['date'],
                    hours_count=int(row['hours_count']),
                    min_price=round(float(row['min_price']), 3),
                    max_price=round(float(row['max_price']), 3),
                    avg_price=round(float(row['avg_price']), 3),
                    price_range=round(float(row['price_range']), 3),
                    negative_hours=int(row['negative_hours'])
                ))

            return stats

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving daily statistics")

    def get_hourly_stats(self) -> List[HourlyStats]:
        """Get hourly price patterns across all data."""
        try:
            df = self.repository.find_hourly_stats()

            # Convert to Pydantic models
            stats = []
            for _, row in df.iterrows():
                stats.append(HourlyStats(
                    hour=int(row['hour']),
                    total_occurrences=int(row['total_occurrences']),
                    min_price=round(float(row['min_price']), 3),
                    max_price=round(float(row['max_price']), 3),
                    avg_price=round(float(row['avg_price']), 3),
                    negative_occurrences=int(row['negative_occurrences'])
                ))

            return stats

        except Exception as e:
            self.handle_exception(e, "Error retrieving hourly statistics")
