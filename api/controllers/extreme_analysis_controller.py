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

        @self.router.get(
            "/extremes",
            tags=["Extreme Market Analysis"],
            summary="Get extreme electricity price analysis (highest and lowest)",
            description="""
            Analyze extreme electricity price events including highest and lowest price occurrences.
            
            This endpoint identifies and analyzes extreme price events that represent significant
            market conditions, supply-demand imbalances, or exceptional market circumstances.
            
            **Extreme Price Analysis:**
            - Highest price events with timestamps and market context
            - Lowest price events including negative price occurrences
            - Statistical analysis of extreme value distributions
            - Frequency and magnitude analysis of outliers
            
            **Market Context Insights:**
            - Extreme price occurrence patterns and triggers
            - Seasonal and temporal correlations
            - Market stress indicators and volatility analysis
            - Historical context and comparative analysis
            
            **Configurable Analysis:**
            - Adjustable record limits for top/bottom results
            - Flexible time period selection
            - Statistical significance thresholds
            - Custom extreme value definitions
            
            **Use Cases:**
            - Market volatility assessment and monitoring
            - Risk management and exposure analysis
            - Trading strategy development and backtesting
            - Market research and academic analysis
            """,
            response_description="Extreme price analysis including highest and lowest prices with market context"
        )
        async def get_extremes(
            limit: int = Query(
                10,
                description="Number of records to return for each extreme (default: 10, max: 100)",
                example=10,
                ge=1,
                le=100
            ),
            service: ExtremeAnalysisService = Depends(
                get_extreme_analysis_service)
        ):
            """
            Get comprehensive extreme price analysis including highest and lowest prices.

            Returns detailed analysis of extreme price events with market context,
            statistical insights, and temporal patterns for risk assessment.
            """
            try:
                return service.get_extreme_prices(limit=limit)
            except HTTPException:
                raise
            except Exception as e:
                self.handle_exception(e, "Error retrieving extreme prices")

        @self.router.get(
            "/price-ranges",
            tags=["Extreme Market Analysis"],
            summary="Get price range analysis for recent periods",
            description="""
            Analyze electricity price ranges and volatility patterns over recent time periods.
            
            This endpoint provides price range analysis focusing on volatility, spread analysis,
            and price distribution patterns to understand market dynamics and trading ranges.
            
            **Price Range Analysis:**
            - Daily, weekly, and monthly price ranges
            - Volatility metrics and standard deviations
            - Price spread analysis and trading ranges
            - Percentile distributions and quartile analysis
            
            **Volatility Assessment:**
            - Intraday price volatility patterns
            - Inter-period volatility comparisons
            - Market stress and stability indicators
            - Risk metrics and uncertainty measures
            
            **Temporal Analysis:**
            - Recent period focus with configurable lookback
            - Trend analysis and pattern identification
            - Seasonal volatility pattern recognition
            - Market regime change detection
            
            **Use Cases:**
            - Trading range identification for strategy development
            - Risk assessment and portfolio management
            - Market volatility monitoring and alerting
            - Price forecasting model calibration
            """,
            response_description="Price range analysis with volatility metrics and temporal patterns"
        )
        async def get_price_ranges(
            days_back: int = Query(
                30,
                description="Number of days to analyze (default: 30, max: 365)",
                example=30,
                ge=1,
                le=365
            ),
            service: ExtremeAnalysisService = Depends(
                get_extreme_analysis_service)
        ):
            """
            Get comprehensive price range analysis for recent periods.

            Returns detailed price range and volatility analysis including trading ranges,
            volatility metrics, and market dynamics for the specified time period.
            """
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
