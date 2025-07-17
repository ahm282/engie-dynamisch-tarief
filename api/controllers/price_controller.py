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
from ..models import PriceRecord, ConsumptionCostAnalysis


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
            summary="Get today's electricity prices with categorization",
            description="""
            Retrieve today's electricity price data with smart categorization for real-time monitoring.
            
            This endpoint provides access to current day's market conditions with automatic price categorization
            to help users understand whether current prices are cheap, regular, expensive, or extremely expensive.
            
            **Price Categories:**
            - **Cheap**: < 7.5 c€/kWh (below average, great for energy usage)
            - **Regular**: 7.5 - 13.0 c€/kWh (normal pricing, around average)
            - **Expensive**: 13.0 - 20.0 c€/kWh (above average, consider reducing usage)
            - **Extremely Expensive**: > 20.0 c€/kWh (very high, avoid high energy usage)
            
            **Features:**
            - Returns all 24 hours for today's date
            - Consumer price impact in euro cents per kWh
            - Automatic price categorization for each hour
            - Category distribution summary
            - Data ordered by hour (0-23)
            
            **Use Cases:**
            - Real-time price monitoring dashboards
            - Smart home energy management decisions
            - Cost-conscious energy usage planning for today
            - Mobile app price alerts and recommendations
            """,
            response_description="Today's electricity prices with consumer impact and smart categorization"
        )
        async def get_current_prices(
            service: PriceService = Depends(get_price_service)
        ):
            """
            Get today's electricity price data with smart categorization.

            Returns today's electricity prices with automatic categorization (cheap, regular, expensive, extremely expensive)
            and category distribution statistics for quick market condition assessment.
            """
            try:
                return service.get_today_prices()
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving current prices")

        @self.router.get(
            "/next-day-prices",
            tags=["Price Data"],
            summary="Get next day's (tomorrow's) electricity prices with categorization",
            description="""
            Retrieve tomorrow's electricity price data with smart categorization for planning ahead.
            
            This endpoint provides access to next day's market prices to help users plan their energy consumption
            and understand upcoming market conditions with automatic price categorization.
            
            **Price Categories:**
            - **Cheap**: < 7.5 c€/kWh (below average, great for energy usage)
            - **Regular**: 7.5 - 13.0 c€/kWh (normal pricing, around average)
            - **Expensive**: 13.0 - 20.0 c€/kWh (above average, consider reducing usage)
            - **Extremely Expensive**: > 20.0 c€/kWh (very high, avoid high energy usage)
            
            **Features:**
            - Returns all 24 hours for tomorrow's date (if available)
            - Consumer price impact in euro cents per kWh
            - Automatic price categorization for each hour
            - Category distribution summary
            - Data ordered by hour (0-23)
            - Availability status and informative messages
            
            **Data Availability:**
            - Next day prices are typically published by energy markets around 12:00-14:00 CET
            - If data is not yet available, the endpoint returns an empty result with availability status
            - Check the `available` field in the response to determine data availability
            
            **Use Cases:**
            - Energy usage planning for tomorrow
            - Smart home scheduling and automation
            - Cost optimization for the next day
            - Energy-intensive task planning (dishwasher, washing machine, EV charging)
            """,
            response_description="Tomorrow's electricity prices with consumer impact and smart categorization (if available)"
        )
        async def get_next_day_prices(
            service: PriceService = Depends(get_price_service)
        ):
            """
            Get tomorrow's electricity price data with smart categorization.

            Returns tomorrow's electricity prices with automatic categorization and availability status.
            If data is not yet available, returns appropriate status information.
            """
            try:
                return service.get_next_day_prices()
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving next day prices")

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

        @self.router.get(
            "/consumption-cost",
            response_model=ConsumptionCostAnalysis,
            tags=["Price Data"],
            summary="Calculate electricity costs based on consumption with realistic usage patterns",
            description="""
            Calculate electricity costs based on total consumption using realistic distribution patterns.
            
            This endpoint simulates real-life electricity usage by distributing consumption across
            time periods using realistic patterns that reflect typical household behavior patterns.
            
            **Realistic Consumption Modeling:**
            - **Morning Peak (7-9h)**: Higher consumption during breakfast and preparation
            - **Evening Peak (18-22h)**: Highest consumption during dinner and leisure time
            - **Night Valley (0-6h)**: Lower consumption during sleeping hours
            - **Weekend Variations**: Different patterns on weekends vs weekdays
            - **Random Variations**: ±20% variations to simulate real unpredictability
            
            **Cost Analysis Features:**
            - Detailed hourly breakdown with consumption and costs
            - Cost categorization by price levels (cheap, regular, expensive, extremely expensive)
            - Savings analysis comparing optimal vs actual timing
            - Statistical insights including price volatility and efficiency scores
            
            **Advanced Analytics:**
            - Cost efficiency scoring based on market timing
            - Potential savings identification through optimal consumption timing
            - Price volatility impact assessment
            - Consumption pattern optimization recommendations
            
            **Use Cases:**
            - Household electricity bill estimation and planning
            - Smart home energy management optimization
            - Cost-conscious consumption timing strategies
            - Energy efficiency assessment and improvement planning
            """,
            response_description="Comprehensive consumption cost analysis with hourly breakdown and savings insights"
        )
        async def calculate_consumption_cost(
            consumption_kwh: float = Query(
                ...,
                description="Total electricity consumption in kWh for the analysis period",
                example=200.0,
                gt=0.0,
                le=10000.0
            ),
            days_back: int = Query(
                30,
                description="Number of days to analyze (default: 30, max: 365)",
                example=30,
                ge=1,
                le=365
            ),
            service: PriceService = Depends(get_price_service)
        ):
            """
            Calculate electricity costs based on consumption using realistic usage patterns.

            Returns detailed cost analysis with hourly breakdown, category distribution,
            and savings opportunities based on realistic consumption distribution patterns.
            """
            try:
                return service.calculate_consumption_cost(
                    total_consumption_kwh=consumption_kwh,
                    days_back=days_back
                )
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error calculating consumption costs")
