"""
Controller for statistics and analysis endpoints.
"""

from fastapi import Query, HTTPException, Depends
from typing import List

from .base_controller import BaseController
from ..services import StatisticsService
from ..repositories import ElectricityPriceRepository
from ..models import DailyStats, HourlyStats, DatabaseStats


def get_statistics_service() -> StatisticsService:
    """Dependency injection for StatisticsService."""
    repository = ElectricityPriceRepository()
    return StatisticsService(repository)


class StatisticsController(BaseController):
    """Controller for statistics and analysis endpoints."""

    def _setup_routes(self):
        """Setup routes for statistics operations."""

        @self.router.get("/database-info", response_model=DatabaseStats)
        async def get_database_info(
            service: StatisticsService = Depends(get_statistics_service)
        ):
            """Get overall database statistics."""
            try:
                return service.get_database_stats()
            except Exception as e:
                self.handle_exception(e, "Error retrieving database info")

        @self.router.get("/daily-stats", response_model=List[DailyStats])
        async def get_daily_stats(
            days_back: int = Query(
                30, description="Number of days to analyze"),
            service: StatisticsService = Depends(get_statistics_service)
        ):
            """Get daily price statistics."""
            try:
                return service.get_daily_stats(days_back=days_back)
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving daily statistics")

        @self.router.get("/hourly-stats", response_model=List[HourlyStats])
        async def get_hourly_stats(
            service: StatisticsService = Depends(get_statistics_service)
        ):
            """Get hourly price patterns across all data."""
            try:
                return service.get_hourly_stats()
            except Exception as e:
                self.handle_exception(e, "Error retrieving hourly statistics")
