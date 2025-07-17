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

        @self.router.get(
            "/negative-prices",
            tags=["Negative Price Analysis"],
            summary="Get comprehensive negative price analysis and market insights",
            description="""
            Comprehensive analysis of negative electricity price occurrences with market insights.
            
            This endpoint provides detailed analysis of negative price events, which occur when
            electricity supply exceeds demand, often due to renewable energy oversupply or
            grid constraints requiring emergency price signals.
            
            **Negative Price Analysis:**
            - Historical negative price frequency and magnitude
            - Temporal patterns and seasonal variations
            - Duration analysis of negative price periods
            - Consumer impact and cost benefit calculations
            
            **Market Insights:**
            - Renewable energy correlation analysis
            - Grid constraint indicators and patterns
            - Supply-demand imbalance identification
            - Economic efficiency indicators
            
            **Statistical Analysis:**
            - Hourly and monthly distribution patterns
            - Magnitude distribution and extremes
            - Frequency trends over time
            - Comparative analysis with positive prices
            
            **Use Cases:**
            - Energy market research and analysis
            - Renewable integration impact studies
            - Grid planning and optimization
            - Energy arbitrage opportunity identification
            """,
            response_description="Comprehensive negative price analysis including frequencies, patterns, and market insights"
        )
        async def get_negative_price_analysis(
            service: NegativePriceAnalysisService = Depends(
                get_negative_price_service)
        ):
            """
            Get comprehensive analysis of negative electricity price occurrences.

            Returns detailed analysis of negative price events including frequency patterns,
            market insights, and temporal distributions with consumer impact assessments.
            """
            try:
                return service.get_negative_price_analysis()
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving negative price analysis")

        @self.router.get(
            "/negative-prices/summary",
            tags=["Negative Price Analysis"],
            summary="Get concise negative price summary statistics",
            description="""
            Get a concise summary of negative electricity price occurrences and key metrics.
            
            This endpoint provides a quick overview of negative price events without the
            detailed breakdowns available in the full analysis endpoint.
            
            **Summary Statistics:**
            - Total count of negative price occurrences
            - Average magnitude of negative prices
            - Most recent negative price events
            - Basic frequency metrics
            
            **Quick Insights:**
            - Overall trend indicators
            - Key statistical measures
            - Recent activity summary
            - Historical context metrics
            
            **Use Cases:**
            - Dashboard quick stats and KPI monitoring
            - Mobile app summary displays
            - Alert system threshold monitoring
            - Executive reporting summaries
            """,
            response_description="Concise summary of negative price statistics and key metrics"
        )
        async def get_negative_price_summary(
            service: NegativePriceAnalysisService = Depends(
                get_negative_price_service)
        ):
            """
            Get concise summary of negative electricity price occurrences.

            Returns key summary statistics and metrics about negative price events
            for quick assessment and monitoring purposes.
            """
            try:
                return service.get_negative_price_summary()
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving negative price summary")

        @self.router.get(
            "/negative-prices/trends",
            tags=["Negative Price Analysis"],
            summary="Get negative price trends and temporal patterns",
            description="""
            Analyze trends and temporal patterns in negative electricity price occurrences.
            
            This endpoint focuses on trend analysis and pattern identification in negative
            price events, providing insights into market evolution and behavioral patterns.
            
            **Trend Analysis:**
            - Historical frequency trends over time
            - Seasonal and cyclical pattern identification
            - Market evolution indicators
            - Predictive trend indicators
            
            **Temporal Patterns:**
            - Hourly occurrence patterns within days
            - Monthly and seasonal variations
            - Weekend vs weekday differences
            - Holiday and special period impacts
            
            **Pattern Insights:**
            - Peak negative price periods identification
            - Correlation with renewable energy production
            - Market condition pattern matching
            - Anomaly and outlier trend detection
            
            **Use Cases:**
            - Market trend forecasting and modeling
            - Renewable energy integration planning
            - Trading strategy development
            - Grid planning and optimization
            """,
            response_description="Negative price trend analysis with temporal patterns and market evolution insights"
        )
        async def get_negative_price_trends(
            service: NegativePriceAnalysisService = Depends(
                get_negative_price_service)
        ):
            """
            Get comprehensive trends and temporal patterns for negative prices.

            Returns detailed trend analysis and pattern insights for negative price events
            including seasonal variations and market evolution indicators.
            """
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
