"""
Controller for expensive price analysis endpoints.

This controller provides comprehensive analysis of expensive electricity prices
including consumer cost impact, trend analysis, and market monitoring capabilities.

Tags:
    - expensive-prices
    - consumer-analysis
    - cost-monitoring
    - price-alerts
    - market-trends

Features:
    - Consumer price impact analysis (c€/kWh)
    - Statistical threshold determination
    - Temporal trend analysis
    - Top expensive price identification
    - Market anomaly detection

Endpoints:
    - GET /expensive-prices: Comprehensive expensive price analysis
    - GET /expensive-prices/summary: Summary statistics and insights
    - GET /expensive-prices/top: Highest recorded prices
    - GET /expensive-prices/trends: Historical trend analysis
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

        @self.router.get(
            "/expensive-prices",
            tags=["Expensive Price Monitoring"],
            summary="Get comprehensive expensive price analysis",
            description="""
            Comprehensive analysis of expensive electricity prices with consumer impact.
            
            This endpoint provides detailed analysis of expensive price occurrences including:
            - Statistical summary and trends
            - Hourly and monthly distribution patterns
            - Consumer cost impact analysis in euro cents per kWh
            - Historical expensive price frequency
            
            **Threshold Configuration:**
            - Default: 15.0 c€/kWh (statistically determined as expensive)
            - Customizable threshold for different analysis needs
            - Based on consumer price formula: (0.10175 * wholesale) + 2.1316
            
            **Analysis Components:**
            - Overall statistics with total counts and averages
            - Hourly distribution showing peak expensive periods
            - Monthly trends revealing seasonal patterns
            - Price percentiles and threshold analysis
            
            **Use Cases:**
            - Consumer cost monitoring and alerts
            - Market analysis and reporting
            - Energy cost budgeting and planning
            - Historical trend identification
            """,
            response_description="Comprehensive expensive price analysis including statistics, trends, and distributions"
        )
        async def get_expensive_price_analysis(
            threshold: Optional[float] = Query(
                15.0,
                description="Minimum consumer price threshold in euro cents per kWh to consider expensive",
                example=15.0,
                ge=5.0,
                le=100.0
            ),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get comprehensive analysis of expensive consumer price occurrences."""
            try:
                return service.get_expensive_price_analysis(threshold)
            except Exception as e:
                self.handle_exception(
                    e, "Error retrieving expensive price analysis")

        @self.router.get(
            "/expensive-prices/summary",
            tags=["Expensive Price Monitoring"],
            summary="Get expensive price summary statistics",
            description="""
            Get a concise summary of expensive consumer price occurrences.
            
            This endpoint provides key statistics about expensive price periods
            in a simplified format perfect for dashboards and quick analysis.
            
            **Summary Includes:**
            - Total count of expensive price occurrences
            - Average expensive price level
            - Highest recorded expensive price
            - Date range of expensive price activity
            - Current threshold being applied
            
            **Performance Benefits:**
            - Faster response than full analysis
            - Reduced data payload
            - Ideal for real-time monitoring
            - Perfect for summary widgets
            """,
            response_description="Summary statistics of expensive price occurrences"
        )
        async def get_expensive_price_summary(
            threshold: Optional[float] = Query(
                15.0,
                description="Minimum consumer price threshold in euro cents per kWh",
                example=15.0,
                ge=5.0,
                le=100.0
            ),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get a summary of expensive consumer price occurrences."""
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
                1.500, description="Minimum consumer price threshold in €/kWh to consider expensive"),
            service: ExpensivePriceAnalysisService = Depends(
                get_expensive_price_service)
        ):
            """Get trends and insights about expensive consumer prices."""
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
