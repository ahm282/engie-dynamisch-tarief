"""
Controller for price data endpoints.

This controller handles all electricity price data retrieval operations including:
- Raw price data with flexible filtering
- Current/recent price queries
- Bulk data exports
- Historical price analysis

Tags:
    - price-data
    - electricity-prices
    - data-retrieval
    - filtering
    - rest-endpoints

Endpoints:
    - GET /prices: Filtered price data with pagination
    - GET /current-prices: Most recent price data
    - GET /all-prices: Bulk price data export
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

        @self.router.get(
            "/prices",
            response_model=List[PriceRecord],
            tags=["Price Data"],
            summary="Get filtered electricity price data",
            description="""
            Retrieve electricity price data with comprehensive filtering options.
            
            This endpoint provides flexible access to historical electricity price data
            with support for date ranges, time-based filtering, and result pagination.
            
            **Filtering Options:**
            - **Date Range**: Filter by start_date and end_date
            - **Recent Data**: Use days_back for recent historical data
            - **Hourly Filter**: Filter by specific hour of the day (0-23)
            - **Pagination**: Limit results and control ordering
            
            **Use Cases:**
            - Historical price analysis
            - Time-series data for charting
            - Peak hour price identification
            - Data export for external analysis
            
            **Performance Notes:**
            - Default limit of 1000 records for performance
            - Use pagination for large datasets
            - Consider date range filtering for better performance
            """,
            response_description="List of electricity price records matching the specified filters"
        )
        async def get_prices(
            start_date: Optional[str] = Query(
                None,
                description="Start date in YYYY-MM-DD format for filtering",
                example="2024-01-01"
            ),
            end_date: Optional[str] = Query(
                None,
                description="End date in YYYY-MM-DD format for filtering",
                example="2024-12-31"
            ),
            days_back: Optional[int] = Query(
                None,
                description="Number of days back from today to retrieve data",
                example=30,
                ge=1,
                le=365
            ),
            hour: Optional[int] = Query(
                None,
                description="Filter by specific hour of the day (0-23)",
                example=18,
                ge=0,
                le=23
            ),
            limit: Optional[int] = Query(
                None,
                description="Maximum number of records to return (default: 1000, max: 10000)",
                example=100,
                ge=1,
                le=10000
            ),
            order: Optional[str] = Query(
                "desc",
                description="Order by timestamp: 'asc' for oldest first, 'desc' for newest first",
                example="desc",
                regex="^(asc|desc)$"
            ),
            service: PriceService = Depends(get_price_service)
        ):
            """
            Get electricity price data with comprehensive filtering options.

            Returns electricity price records filtered by the specified criteria.
            Supports date ranges, recent data queries, hourly filtering, and pagination.
            """
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

        @self.router.get(
            "/current-prices",
            tags=["Price Data"],
            summary="Get current/recent electricity prices",
            description="""
            Retrieve the most recent electricity price data for real-time monitoring.
            
            This endpoint provides quick access to current market conditions and recent
            price trends. Ideal for dashboards, alerts, and real-time applications.
            
            **Features:**
            - Configurable lookback period (default: 24 hours)
            - Always returns the most recent data first
            - Optimized for real-time applications
            - Includes both wholesale and consumer prices
            
            **Use Cases:**
            - Real-time price monitoring dashboards
            - Current market condition alerts
            - Live price widgets
            - Mobile app current price displays
            """,
            response_description="List of recent electricity price records ordered by timestamp"
        )
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
