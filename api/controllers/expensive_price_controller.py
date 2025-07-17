"""
Controller for expensive price analysis endpoints.
"""

from fastapi import HTTPException, Depends, Query
from typing import Optional

from .base_controller import BaseController
from ..services import ExpensivePriceAnalysisService
from ..repositories import ExpensivePriceRepository


def get_expensive_price_service() -> ExpensivePriceAnalysisService:
    """Dependency injection for ExpensivePriceAnalysisService."""
    repository = ExpensivePriceRepository()
    return ExpensivePriceAnalysisService(repository)


class ExpensivePriceController(BaseController):
    """Controller for expensive price analysis endpoints."""

    def _setup_routes(self):
        """Setup routes for expensive price analysis operations."""

        @self.router.get("/expensive-prices")
        async def get_expensive_price_analysis(
            threshold: Optional[float] = Query(
                0.200, description="Minimum price per kWh threshold in EUR to consider expensive"),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get comprehensive analysis of expensive price occurrences."""
            try:
                return service.get_expensive_price_analysis(threshold)
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving expensive price analysis")

        @self.router.get("/expensive-prices/summary")
        async def get_expensive_price_summary(
            threshold: Optional[float] = Query(
                0.200, description="Minimum price threshold in EUR to consider expensive"),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get a summary of expensive price occurrences."""
            try:
                return service.get_expensive_price_summary(threshold)
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving expensive price summary")

        @self.router.get("/expensive-prices/top")
        async def get_top_expensive_prices(
            limit: Optional[int] = Query(
                10, description="Number of top expensive prices to return (max 100)"),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get the highest price records ever recorded."""
            try:
                return service.get_top_expensive_prices(limit)
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving top expensive prices")

        @self.router.get("/expensive-prices/trends")
        async def get_expensive_price_trends(
            threshold: Optional[float] = Query(
                0.200, description="Minimum price threshold in EUR to consider expensive"),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get trends and insights about expensive prices."""
            try:
                analysis = service.get_expensive_price_analysis(threshold)

                # Build trends response
                hourly_stats = analysis.get('by_hour', [])
                monthly_stats = analysis.get('by_month', [])

                # Find peak hours for expensive prices
                peak_hours = sorted(
                    hourly_stats, key=lambda x: x.expensive_count, reverse=True)[:3]

                # Find recent trend (last 3 months)
                recent_months = monthly_stats[:3] if len(
                    monthly_stats) >= 3 else monthly_stats

                return {
                    "peak_expensive_hours": [
                        {
                            "hour": hour.hour,
                            "expensive_count": hour.expensive_count,
                            "avg_price": hour.avg_expensive_price,
                            "description": f"Hour {hour.hour}:00 - {hour.expensive_count} expensive occurrences"
                        } for hour in peak_hours
                    ],
                    "recent_monthly_trend": [
                        {
                            "month": month.month,
                            "expensive_count": month.expensive_count,
                            "avg_price": month.avg_expensive_price,
                            "description": f"{month.month} - {month.expensive_count} expensive hours"
                        } for month in recent_months
                    ],
                    "threshold_used": threshold,
                    "insights": {
                        "most_expensive_hour": peak_hours[0].hour if peak_hours else None,
                        "recent_expensive_activity": len(recent_months) > 0,
                        "avg_expensive_price": analysis.get('overall_stats').avg_expensive_price if analysis.get('overall_stats') else None
                    }
                }

            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving expensive price trends")

        @self.router.get("/expensive-prices/percentiles")
        async def get_price_percentiles(
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get price percentiles and suggested thresholds for expensive price analysis."""
            try:
                return service.get_price_percentiles()
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving price percentiles")
