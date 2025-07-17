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

        @self.router.get(
            "/",
            response_model=APIInfo,
            tags=["System Information"],
            summary="Get API information and available endpoints",
            description="""
            Retrieve comprehensive information about the Electricity Price API.
            
            This root endpoint provides essential API information including version details,
            available endpoints, and service capabilities for API discovery and integration.
            
            **API Information:**
            - Current API version and service details
            - Complete endpoint catalog with descriptions
            - Service capabilities and features overview
            - Integration guidelines and best practices
            
            **Endpoint Directory:**
            - Price data endpoints for historical and current data
            - Statistical analysis endpoints for insights
            - Specialized analysis endpoints for market research
            - Health and monitoring endpoints for system status
            
            **Service Features:**
            - Real-time and historical electricity price data
            - Advanced statistical analysis and pattern recognition
            - Market insight generation and trend analysis
            - High-performance data retrieval with filtering
            
            **Use Cases:**
            - API discovery for new integrations
            - Service capability assessment
            - Documentation and integration planning
            - System architecture and endpoint mapping
            """,
            response_description="Comprehensive API information including version, endpoints, and service capabilities"
        )
        async def get_api_info():
            """
            Get comprehensive API information and endpoint directory.

            Returns detailed information about the Electricity Price API including
            version details, available endpoints, and service capabilities.
            """
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

        @self.router.get(
            "/health",
            response_model=HealthResponse,
            tags=["System Information"],
            summary="API health check and service status",
            description="""
            Check the health and operational status of the Electricity Price API service.
            
            This endpoint provides real-time health monitoring and service status information
            for system monitoring, alerting, and operational readiness verification.
            
            **Health Monitoring:**
            - Service operational status verification
            - API responsiveness and availability check
            - Basic system health indicators
            - Service version and build information
            
            **Status Information:**
            - Current service state and availability
            - Response time and performance indicators
            - System readiness for request processing
            - Operational health confirmation
            
            **Monitoring Integration:**
            - Compatible with health check monitoring systems
            - Standardized health response format
            - Suitable for load balancer health checks
            - Integration with observability platforms
            
            **Use Cases:**
            - Service health monitoring and alerting
            - Load balancer health verification
            - System status dashboard integration
            - Automated deployment health validation
            """,
            response_description="Service health status and operational readiness information"
        )
        async def health_check():
            """
            Perform API health check and return service status.

            Returns current operational status and health indicators for the
            Electricity Price API service monitoring and verification.
            """
            return HealthResponse(
                status="healthy",
                service="electricity-price-api"
            )
