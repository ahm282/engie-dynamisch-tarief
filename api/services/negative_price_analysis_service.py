"""
Service for negative price analysis operations.
"""

from typing import Dict, Any, List
from fastapi import HTTPException

from .base_service import BaseService
from ..repositories import NegativePriceRepository
from ..models import NegativePriceStats, HourlyNegativeStats, MonthlyNegativeStats


class NegativePriceAnalysisService(BaseService):
    """Service for negative price analysis operations."""

    def __init__(self, repository: NegativePriceRepository = None):
        """Initialize service with repository dependency injection."""
        super().__init__(repository or NegativePriceRepository())

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for negative price analysis."""
        # No specific validation needed for negative price analysis
        return True

    def get_negative_price_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of negative price occurrences."""
        try:
            result = self.repository.find_negative_price_stats()
            stats_df = result["overall"]
            hourly_df = result["hourly"]
            monthly_df = result["monthly"]

            # Convert to response format
            overall_stats = None
            if not stats_df.empty:
                stats_row = stats_df.iloc[0]
                overall_stats = NegativePriceStats(
                    total_negative=int(stats_row['total_negative']),
                    lowest_price=round(float(stats_row['lowest_price']), 3),
                    avg_negative_price=round(
                        float(stats_row['avg_negative_price']), 3),
                    first_negative_date=stats_row['first_negative_date'],
                    last_negative_date=stats_row['last_negative_date']
                )

            hourly_stats = []
            for _, row in hourly_df.iterrows():
                hourly_stats.append(HourlyNegativeStats(
                    hour=int(row['hour']),
                    negative_count=int(row['negative_count']),
                    avg_negative_price=round(
                        float(row['avg_negative_price']), 3)
                ))

            monthly_stats = []
            for _, row in monthly_df.iterrows():
                monthly_stats.append(MonthlyNegativeStats(
                    month=row['month'],
                    negative_count=int(row['negative_count']),
                    avg_negative_price=round(
                        float(row['avg_negative_price']), 3)
                ))

            return {
                "overall_stats": overall_stats,
                "by_hour": hourly_stats,
                "by_month": monthly_stats
            }

        except Exception as e:
            self.handle_exception(
                e, "Error retrieving negative price analysis")

    def get_negative_price_summary(self) -> Dict[str, Any]:
        """Get a summary of negative price occurrences."""
        try:
            analysis = self.get_negative_price_analysis()
            overall = analysis["overall_stats"]

            if overall is None:
                return {"message": "No negative prices found in the database"}

            # Find peak negative hour
            hourly_stats = analysis["by_hour"]
            peak_hour = max(
                hourly_stats, key=lambda x: x.negative_count) if hourly_stats else None

            # Find most recent negative month
            monthly_stats = analysis["by_month"]
            recent_month = monthly_stats[0] if monthly_stats else None

            return {
                "summary": {
                    "total_negative_hours": overall.total_negative,
                    "percentage_of_total": round((overall.total_negative / self._get_total_records()) * 100, 3),
                    "lowest_price_ever": round(overall.lowest_price, 3),
                    "average_negative_price": round(overall.avg_negative_price, 3)
                },
                "peak_negative_hour": {
                    "hour": peak_hour.hour,
                    "count": peak_hour.negative_count,
                    "avg_price": round(peak_hour.avg_negative_price, 3),
                    "description": f"Hour {peak_hour.hour}:00 has the most negative price occurrences ({peak_hour.negative_count} times) with avg {round(peak_hour.avg_negative_price, 1)} €/MWh"
                } if peak_hour else None,
                "recent_month": {
                    "month": recent_month.month,
                    "count": recent_month.negative_count,
                    "avg_price": round(recent_month.avg_negative_price, 3)
                } if recent_month else None
            }

        except Exception as e:
            self.handle_exception(e, "Error retrieving negative price summary")

    def _get_total_records(self) -> int:
        """Helper method to get total records count."""
        try:
            # Use the count method which counts all records, not just negative ones
            from ..repositories import StatisticsRepository
            stats_repo = StatisticsRepository()
            return stats_repo.count()
        except Exception:
            return 1  # Avoid division by zero
