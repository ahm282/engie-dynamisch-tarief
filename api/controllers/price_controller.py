"""
Controller for price data endpoints.
"""

from fastapi import Query, HTTPException, Depends
from typing import Optional, List

from .base_controller import BaseController
from ..services import PriceService
from ..repositories import ElectricityPriceRepository
from ..models import PriceRecord


def get_price_service() -> PriceService:
    """Dependency injection for PriceService."""
    repository = ElectricityPriceRepository()
    return PriceService(repository)


class PriceController(BaseController):
    """Controller for price data endpoints."""

    def _setup_routes(self):
        """Setup routes for price data operations."""

        @self.router.get("/prices", response_model=List[PriceRecord])
        async def get_prices(
            start_date: Optional[str] = Query(
                None, description="Start date (YYYY-MM-DD)"),
            end_date: Optional[str] = Query(
                None, description="End date (YYYY-MM-DD)"),
            days_back: Optional[int] = Query(
                None, description="Number of days back from today"),
            hour: Optional[int] = Query(
                None, description="Filter by specific hour (0-23)"),
            limit: Optional[int] = Query(
                None, description="Maximum number of records to return"),
            order: Optional[str] = Query(
                "desc", description="Order by timestamp (asc/desc)"),
            service: PriceService = Depends(get_price_service)
        ):
            """Get electricity price data with various filters."""
            try:
                return service.get_prices(
                    start_date=start_date,
                    end_date=end_date,
                    days_back=days_back,
                    hour=hour,
                    limit=limit,
                    order=order
                )
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving prices")

        @self.router.get("/current-prices")
        async def get_current_prices(
            hours: int = Query(
                24, description="Number of recent hours to retrieve"),
            service: PriceService = Depends(get_price_service)
        ):
            """Get the most recent price data."""
            try:
                return service.get_current_prices(hours=hours)
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving current prices")

        @self.router.get("/all-prices")
        async def get_all_prices(
            start_date: Optional[str] = Query(
                None, description="Start date (YYYY-MM-DD)"),
            end_date: Optional[str] = Query(
                None, description="End date (YYYY-MM-DD)"),
            days_back: Optional[int] = Query(
                None, description="Number of days back from today"),
            hour: Optional[int] = Query(
                None, description="Filter by specific hour (0-23)"),
            order: Optional[str] = Query(
                "desc", description="Order by timestamp (asc/desc)"),
            service: PriceService = Depends(get_price_service)
        ):
            """Get all electricity price data with filters - no limit applied. 
            Warning: This can return very large datasets (17,950+ records)."""
            try:
                return service.get_all_prices(
                    start_date=start_date,
                    end_date=end_date,
                    days_back=days_back,
                    hour=hour,
                    order=order
                )
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving all prices")
