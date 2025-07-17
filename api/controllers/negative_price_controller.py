"""
Controller for negative price analysis endpoints.
"""

from fastapi import HTTPException, Depends

from .base_controller import BaseController
from ..services import NegativePriceAnalysisService
from ..repositories import ElectricityPriceRepository


def get_negative_price_service() -> NegativePriceAnalysisService:
    """Dependency injection for NegativePriceAnalysisService."""
    repository = ElectricityPriceRepository()
    return NegativePriceAnalysisService(repository)


class NegativePriceController(BaseController):
    """Controller for negative price analysis endpoints."""

    def _setup_routes(self):
        """Setup routes for negative price analysis operations."""

        @self.router.get("/negative-prices")
        async def get_negative_price_analysis(
            service: NegativePriceAnalysisService = Depends(
                get_negative_price_service)
        ):
            """Get comprehensive analysis of negative price occurrences."""
            try:
                return service.get_negative_price_analysis()
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving negative price analysis")

        @self.router.get("/negative-prices/summary")
        async def get_negative_price_summary(
            service: NegativePriceAnalysisService = Depends(
                get_negative_price_service)
        ):
            """Get a summary of negative price occurrences."""
            try:
                return service.get_negative_price_summary()
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving negative price summary")

        @self.router.get("/negative-prices/trends")
        async def get_negative_price_trends(
            service: NegativePriceAnalysisService = Depends(
                get_negative_price_service)
        ):
            """Get trends and insights about negative prices."""
            try:
                analysis = service.get_negative_price_analysis()

                # Build trends response
                hourly_stats = analysis.get('by_hour', [])
                monthly_stats = analysis.get('by_month', [])

                # Find peak hours for negative prices
                peak_hours = sorted(
                    hourly_stats, key=lambda x: x.negative_count, reverse=True)[:3]

                # Find recent trend (last 3 months)
                recent_months = monthly_stats[:3] if len(
                    monthly_stats) >= 3 else monthly_stats

                return {
                    "peak_negative_hours": [
                        {
                            "hour": hour.hour,
                            "negative_count": hour.negative_count,
                            "avg_price": hour.avg_negative_price,
                            "description": f"Hour {hour.hour}:00 - {hour.negative_count} negative occurrences"
                        } for hour in peak_hours
                    ],
                    "recent_monthly_trend": [
                        {
                            "month": month.month,
                            "negative_count": month.negative_count,
                            "avg_price": month.avg_negative_price
                        } for month in recent_months
                    ],
                    "insights": {
                        "most_negative_hour": peak_hours[0].hour if peak_hours else None,
                        "is_increasing": len(recent_months) >= 2 and recent_months[0].negative_count > recent_months[1].negative_count if recent_months else False,
                        "seasonal_pattern": "Midday hours (12-15) show highest negative price frequency" if any(h.hour in [12, 13, 14, 15] for h in peak_hours[:2]) else "No clear seasonal pattern"
                    }
                }
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving negative price trends")
