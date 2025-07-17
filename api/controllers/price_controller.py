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
                description="Start date in YYYY-MM-DD format for filtering"
            ),
            end_date: Optional[str] = Query(
                None,
                description="End date in YYYY-MM-DD format for filtering"
            ),
            days_back: Optional[int] = Query(
                None,
                description="Number of days back from today to retrieve data",
                ge=1,
                le=365
            ),
            hour: Optional[int] = Query(
                None,
                description="Filter by specific hour of the day (0-23)",
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
            summary="Get current/recent electricity prices with categorization",
            description="""
            Retrieve the most recent electricity price data with smart categorization for real-time monitoring.
            
            This endpoint provides quick access to current market conditions with automatic price categorization
            to help users understand whether current prices are cheap, regular, expensive, or extremely expensive.
            
            **Price Categories:**
            - **Cheap**: < 7.5 c€/kWh (below average, great for energy usage)
            - **Regular**: 7.5 - 13.0 c€/kWh (normal pricing, around average)
            - **Expensive**: 13.0 - 20.0 c€/kWh (above average, consider reducing usage)
            - **Extremely Expensive**: > 20.0 c€/kWh (very high, avoid high energy usage)
            
            **Enhanced Features:**
            - Configurable lookback period (default: 24 hours)
            - Consumer price impact in euro cents per kWh
            - Automatic price categorization for each hour
            - Category distribution summary
            - Always returns the most recent data first
            
            **Use Cases:**
            - Real-time price monitoring dashboards
            - Smart home energy management decisions
            - Cost-conscious energy usage planning
            - Mobile app price alerts and recommendations
            """,
            response_description="Recent electricity prices with consumer impact and smart categorization"
        )
        async def get_current_prices(
            hours: int = Query(
                24,
                description="Number of recent hours to retrieve (1-168, default: 24)",
                example=24,
                ge=1,
                le=168
            ),
            service: PriceService = Depends(get_price_service)
        ):
            """
            Get the most recent electricity price data with smart categorization.

            Returns recent electricity prices with automatic categorization (cheap, regular, expensive, extremely expensive)
            and category distribution statistics for quick market condition assessment.
            """
            try:
                return service.get_current_prices(hours=hours)
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving current prices")

        @self.router.get(
            "/all-prices",
            response_model=List[PriceRecord],
            tags=["Price Data"],
            summary="Get all electricity price data without pagination limits",
            description="""
            Retrieve all electricity price data with optional filtering - no pagination limits applied.
            
            **⚠️ Warning:** This endpoint can return very large datasets (17,950+ records). 
            Use filtering options to reduce response size and improve performance.
            
            **Filtering Options:**
            - **Date Range**: Filter by start_date and end_date to limit time range
            - **Recent Data**: Use days_back for recent historical data only
            - **Hourly Filter**: Filter by specific hour of the day (0-23)
            - **Ordering**: Control sort order by timestamp
            
            **Use Cases:**
            - Full historical data exports
            - Comprehensive analysis requiring complete datasets
            - Data migration and backup operations
            - Statistical analysis over entire price history
            
            **Performance Recommendations:**
            - Always use date range filtering for better performance
            - Consider using /prices endpoint with pagination for UI applications
            - Use specific hour filtering when analyzing time-of-day patterns
            """,
            response_description="Complete list of electricity price records matching the specified filters"
        )
        async def get_all_prices(
            start_date: Optional[str] = Query(
                None,
                description="Start date in YYYY-MM-DD format for filtering"
            ),
            end_date: Optional[str] = Query(
                None,
                description="End date in YYYY-MM-DD format for filtering"
            ),
            days_back: Optional[int] = Query(
                None,
                description="Number of days back from today to retrieve data",
                ge=1,
                le=365
            ),
            hour: Optional[int] = Query(
                None,
                description="Filter by specific hour of the day (0-23)",
                ge=0,
                le=23
            ),
            order: Optional[str] = Query(
                "desc",
                description="Order by timestamp: 'asc' for oldest first, 'desc' for newest first",
                regex="^(asc|desc)$"
            ),
            service: PriceService = Depends(get_price_service)
        ):
            """
            Get all electricity price data with optional filtering - no pagination limits.

            Returns complete electricity price datasets filtered by the specified criteria.
            Warning: This can return very large datasets, use filtering to improve performance.
            """
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
