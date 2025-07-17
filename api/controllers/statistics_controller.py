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

        @self.router.get(
            "/database-info",
            response_model=DatabaseStats,
            tags=["Statistics & Analytics"],
            summary="Get comprehensive database statistics",
            description="""
            Retrieve overall statistics and metadata about the electricity price database.
            
            This endpoint provides key metrics about the data quality, coverage, and completeness
            of the electricity price database for system monitoring and analysis planning.
            
            **Statistical Overview:**
            - Total number of price records in the database
            - Date range coverage (earliest to latest data points)
            - Data completeness metrics and gap analysis
            - Average prices and value distributions
            
            **Database Health Metrics:**
            - Record count validations
            - Data quality indicators
            - Coverage analysis by time periods
            - Statistical summaries for monitoring
            
            **Use Cases:**
            - System health monitoring and alerting
            - Data quality assessment and reporting
            - Analysis planning and scope determination
            - API status and data availability checks
            """,
            response_description="Comprehensive database statistics including record counts, date ranges, and data quality metrics"
        )
        async def get_database_info(
            service: StatisticsService = Depends(get_statistics_service)
        ):
            """
            Get comprehensive database statistics and health metrics.

            Returns detailed statistics about the electricity price database including
            record counts, date coverage, and data quality indicators for monitoring.
            """
            try:
                return service.get_database_stats()
            except Exception as e:
                self.handle_exception(e, "Error retrieving database info")

        @self.router.get(
            "/daily-stats",
            response_model=List[DailyStats],
            tags=["Statistics & Analytics"],
            summary="Get daily electricity price statistics and trends",
            description="""
            Analyze daily electricity price patterns and statistics over a configurable period.
            
            This endpoint provides comprehensive daily-level analysis of electricity prices
            including statistical summaries, trend identification, and pattern recognition.
            
            **Daily Analysis Components:**
            - Daily average, minimum, and maximum prices
            - Price volatility and standard deviation metrics
            - Day-over-day price change analysis
            - Statistical distributions and percentiles
            
            **Trend Analysis Features:**
            - Price movement patterns and correlations
            - Seasonal and cyclical trend identification
            - Outlier detection and anomaly flagging
            - Historical comparison benchmarks
            
            **Configurable Parameters:**
            - Customizable analysis period (default: 30 days)
            - Flexible date range selection
            - Statistical aggregation options
            
            **Use Cases:**
            - Daily price monitoring and reporting
            - Trend analysis for forecasting models
            - Market behavior pattern identification
            - Historical performance benchmarking
            """,
            response_description="Daily electricity price statistics with trends, volatility metrics, and statistical summaries"
        )
        async def get_daily_stats(
            days_back: int = Query(
                30,
                description="Number of days to analyze (default: 30, max: 365)",
                example=30,
                ge=1,
                le=365
            ),
            service: StatisticsService = Depends(get_statistics_service)
        ):
            """
            Get comprehensive daily price statistics and trend analysis.

            Returns detailed daily-level statistics including averages, volatility metrics,
            and trend patterns for the specified analysis period.
            """
            try:
                return service.get_daily_stats(days_back=days_back)
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving daily statistics")

        @self.router.get(
            "/hourly-stats",
            response_model=List[HourlyStats],
            tags=["Statistics & Analytics"],
            summary="Get hourly electricity price patterns and analysis",
            description="""
            Analyze hourly electricity price patterns across the entire historical dataset.
            
            This endpoint reveals time-of-day pricing patterns and hourly market behaviors
            by aggregating all historical data to identify consistent hourly trends.
            
            **Hourly Pattern Analysis:**
            - Average prices for each hour of the day (0-23)
            - Peak and off-peak hour identification
            - Price volatility by hour of day
            - Statistical distributions across time periods
            
            **Market Behavior Insights:**
            - Daily energy demand patterns reflected in pricing
            - Market trading session impacts on hourly prices
            - Seasonal and weekly pattern overlays
            - Grid supply and demand correlation indicators
            
            **Comprehensive Coverage:**
            - Analysis spans entire historical dataset
            - All available price records included
            - Cross-seasonal pattern averaging
            - Long-term trend normalization
            
            **Use Cases:**
            - Energy consumption optimization planning
            - Market trading strategy development
            - Peak demand analysis and forecasting
            - Time-of-use pricing strategy design
            """,
            response_description="Hourly electricity price patterns with statistical analysis across all historical data"
        )
        async def get_hourly_stats(
            service: StatisticsService = Depends(get_statistics_service)
        ):
            """
            Get comprehensive hourly price patterns and market behavior analysis.

            Returns detailed hourly statistics showing time-of-day pricing patterns
            and market behaviors aggregated across all historical data.
            """
            try:
                return service.get_hourly_stats()
            except Exception as e:
                self.handle_exception(e, "Error retrieving hourly statistics")
