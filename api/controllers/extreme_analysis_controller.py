"""
Controller for extreme price analysis endpoints.
"""

from fastapi import Query, HTTPException, Depends

from .base_controller import BaseController
from ..services import ExtremeAnalysisService
from ..repositories import ExtremePriceRepository


def get_extreme_analysis_service() -> ExtremeAnalysisService:
    """Dependency injection for ExtremeAnalysisService."""
    repository = ExtremePriceRepository()
    return ExtremeAnalysisService(repository)


class ExtremeAnalysisController(BaseController):
    """Controller for extreme price analysis endpoints."""

    def _setup_routes(self):
        """Setup routes for extreme analysis operations."""

        @self.router.get("/extremes")
        async def get_extremes(
            limit: int = Query(
                10, description="Number of records to return for each extreme"),
            service: ExtremeAnalysisService = Depends(
                get_extreme_analysis_service)
        ):
            """Get highest and lowest prices."""
            try:
                return service.get_extreme_prices(limit=limit)
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving extreme prices")

        @self.router.get("/price-ranges")
        async def get_price_ranges(
            days_back: int = Query(
                30, description="Number of days to analyze"),
            service: ExtremeAnalysisService = Depends(
                get_extreme_analysis_service)
        ):
            """Get price ranges analysis for recent period."""
            try:
                # This could be a new method to add to ExtremeAnalysisService
                extremes = service.get_extreme_prices(limit=1)
                highest = extremes['highest_prices'][0] if extremes['highest_prices'] else None
                lowest = extremes['lowest_prices'][0] if extremes['lowest_prices'] else None

                return {
                    "analysis_period_days": days_back,
                    "highest_price": {
                        "price": highest.price_eur,
                        "timestamp": highest.timestamp,
                        "date": highest.date,
                        "hour": highest.hour
                    } if highest else None,
                    "lowest_price": {
                        "price": lowest.price_eur,
                        "timestamp": lowest.timestamp,
                        "date": lowest.date,
                        "hour": lowest.hour
                    } if lowest else None,
                    "price_spread": (highest.price_eur - lowest.price_eur) if (highest and lowest) else None
                }
            except Exception as e:
                self.handle_exception(e, "Error retrieving price ranges")
