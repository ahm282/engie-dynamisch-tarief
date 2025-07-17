"""
Service for expensive price analysis operations.
"""

from typing import Dict, Any, List, Optional
from fastapi import HTTPException

from .base_service import BaseService
from ..repositories import ExpensivePriceRepository
from ..models import ExpensivePriceStats, HourlyExpensiveStats, MonthlyExpensiveStats


class ExpensivePriceAnalysisService(BaseService):
    """Service for expensive price analysis operations."""

    def __init__(self, repository: ExpensivePriceRepository = None):
        """Initialize service with repository dependency injection."""
        super().__init__(repository or ExpensivePriceRepository())

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for expensive price analysis."""
        threshold = kwargs.get('threshold')
        if threshold is not None and threshold < 0:
            raise HTTPException(
                status_code=400,
                detail="Threshold must be a non-negative number"
            )
        return True

    def get_expensive_price_analysis(self, threshold: float = 1.500) -> Dict[str, Any]:
        """Get comprehensive analysis of expensive consumer price occurrences."""
        try:
            self.validate_input(threshold=threshold)

            result = self.repository.find_expensive_price_stats(threshold)
            stats_df = result["overall"]
            hourly_df = result["hourly"]
            monthly_df = result["monthly"]

            # Convert to response format
            overall_stats = None
            if not stats_df.empty:
                stats_row = stats_df.iloc[0]
                overall_stats = ExpensivePriceStats(
                    total_expensive=int(stats_row['total_expensive']),
                    highest_price=round(float(stats_row['highest_price']), 3),
                    avg_expensive_price=round(
                        float(stats_row['avg_expensive_price']), 3),
                    first_expensive_date=stats_row['first_expensive_date'],
                    last_expensive_date=stats_row['last_expensive_date'],
                    threshold_cents=round(
                        float(stats_row['threshold_cents']), 3)
                )

            hourly_stats = []
            for _, row in hourly_df.iterrows():
                hourly_stats.append(HourlyExpensiveStats(
                    hour=int(row['hour']),
                    expensive_count=int(row['expensive_count']),
                    avg_expensive_price=round(
                        float(row['avg_expensive_price']), 3)
                ))

            monthly_stats = []
            for _, row in monthly_df.iterrows():
                monthly_stats.append(MonthlyExpensiveStats(
                    month=row['month'],
                    expensive_count=int(row['expensive_count']),
                    avg_expensive_price=round(
                        float(row['avg_expensive_price']), 3)
                ))

            return {
                "overall_stats": overall_stats,
                "by_hour": hourly_stats,
                "by_month": monthly_stats
            }

        except Exception as e:
            self.handle_exception(
                e, "Error retrieving expensive price analysis")

    def get_expensive_price_summary(self, threshold: float = 1.500) -> Dict[str, Any]:
        """Get a summary of expensive consumer price occurrences."""
        try:
            self.validate_input(threshold=threshold)

            analysis = self.get_expensive_price_analysis(threshold)
            overall = analysis["overall_stats"]

            if overall is None:
                return {"message": f"No prices above {threshold} EUR found in the database"}

            # Find peak expensive hour
            hourly_stats = analysis["by_hour"]
            peak_hour = max(
                hourly_stats, key=lambda x: x.expensive_count) if hourly_stats else None

            # Find most recent expensive month
            monthly_stats = analysis["by_month"]
            recent_month = monthly_stats[0] if monthly_stats else None

            return {
                "summary": {
                    "total_expensive_hours": overall.total_expensive,
                    "percentage_of_total": round((overall.total_expensive / self._get_total_records()) * 100, 3),
                    "highest_price_ever": round(overall.highest_price, 3),
                    "average_expensive_price": round(overall.avg_expensive_price, 3),
                    "threshold_used": round(overall.threshold_cents, 3)
                },
                "peak_expensive_hour": {
                    "hour": peak_hour.hour,
                    "count": peak_hour.expensive_count,
                    "avg_price": round(peak_hour.avg_expensive_price, 3)
                } if peak_hour else None,
                "recent_month": {
                    "month": recent_month.month,
                    "count": recent_month.expensive_count,
                    "avg_price": round(recent_month.avg_expensive_price, 3)
                } if recent_month else None
            }

        except Exception as e:
            self.handle_exception(
                e, "Error retrieving expensive price summary")

    def get_top_expensive_prices(self, limit: int = 10) -> Dict[str, Any]:
        """Get the most expensive price records."""
        try:
            if limit <= 0 or limit > 100:
                limit = 10

            df = self.repository.find_top_expensive_prices(limit)

            if df.empty:
                return {"message": "No price data found"}

            records = []
            for _, row in df.iterrows():
                records.append({
                    "timestamp": row['timestamp'],
                    "date": row['date'],
                    "hour": int(row['hour']),
                    "price_eur": round(float(row['price_eur']), 3),
                    "price_raw": row['price_raw']
                })

            return {
                "top_expensive_prices": records,
                "count": len(records),
                "highest_price": round(float(df['price_eur'].max()), 3),
                "avg_of_top_prices": round(float(df['price_eur'].mean()), 3)
            }

        except Exception as e:
            self.handle_exception(e, "Error retrieving top expensive prices")

    def get_price_percentiles(self) -> Dict[str, Any]:
        """Get price percentiles to help determine expensive thresholds."""
        try:
            percentiles = self.repository.get_price_percentiles()

            # Round all values to 3 decimal places
            for key, value in percentiles.items():
                if isinstance(value, (int, float)):
                    percentiles[key] = round(float(value), 3)

            return {
                "price_distribution": percentiles,
                "suggested_thresholds": {
                    "moderate_expensive": round(percentiles.get('p75', 0.200), 3),
                    "high_expensive": round(percentiles.get('p90', 0.300), 3),
                    "extreme_expensive": round(percentiles.get('p95', 0.400), 3)
                }
            }

        except Exception as e:
            self.handle_exception(e, "Error retrieving price percentiles")

    def _get_total_records(self) -> int:
        """Helper method to get total records count."""
        try:
            # Use the count method which counts all records, not just expensive ones
            from ..repositories import StatisticsRepository
            stats_repo = StatisticsRepository()
            return stats_repo.count()
        except Exception:
            return 1  # Avoid division by zero
