"""
Controller for API information and health endpoints.
"""

from fastapi import APIRouter

from .base_controller import BaseController
from ..models import APIInfo, HealthResponse


class InfoController(BaseController):
    """Controller for API information and health endpoints."""

    def _setup_routes(self):
        """Setup routes for API info and health."""

        @self.router.get("/", response_model=APIInfo)
        async def get_api_info():
            """API root endpoint with basic information."""
            return APIInfo(
                message="Electricity Price API",
                version="2.0.0",
                endpoints={
                    "prices": "/prices - Get price data",
                    "daily_stats": "/daily-stats - Get daily statistics",
                    "hourly_stats": "/hourly-stats - Get hourly patterns",
                    "extremes": "/extremes - Get highest/lowest prices",
                    "negative_prices": "/negative-prices - Get negative price analysis",
                    "current_prices": "/current-prices - Get most recent prices",
                    "database_info": "/database-info - Get database statistics",
                    "health": "/health - Health check"
                }
            )

        @self.router.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                service="electricity-price-api"
            )
