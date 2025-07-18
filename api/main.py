"""
This module creates and configures the main FastAPI application for the
Electricity Price Analysis API. It implements a comprehensive REST API
for analyzing Belgian electricity market data with advanced analytics.

Tags:
    - fastapi
    - electricity-prices
    - data-analysis
    - rest-api
    - mvc-architecture

Features:
    - Real-time electricity price data access
    - Negative price analysis and alerts
    - Expensive price monitoring and trends
    - Statistical analysis and reporting
    - Comprehensive Swagger documentation
    - CORS-enabled for web applications

API Categories:
    - Information: System health and API metadata
    - Price Data: Raw price retrieval and filtering
    - Statistics: Aggregated data and analytics
    - Negative Prices: Sub-zero price analysis
    - Expensive Prices: High-cost period monitoring
    - Extreme Analysis: Market anomaly detection

Author: Ahmed Mahgoub (ahm282)
Version: 2.0.0
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .controllers import electricity_controller
from .config import app_config


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with comprehensive documentation.

    This function initializes the FastAPI application with:
    - Comprehensive API metadata and documentation
    - CORS middleware for cross-origin requests
    - Modular controller routing with proper tags
    - OpenAPI/Swagger documentation at /docs
    - ReDoc documentation at /redoc

    Returns:
        FastAPI: Configured FastAPI application instance ready for deployment.

    Configuration:
        - Title: From app_config.api.title
        - Description: From app_config.api.description  
        - Version: From app_config.api.version
        - CORS: Configurable origins, methods, and headers

    Routes:
        - /docs: Interactive Swagger UI documentation
        - /redoc: Alternative ReDoc documentation
        - /api/*: All electricity price analysis endpoints
    """

    # Initialize FastAPI app with comprehensive configuration
    app = FastAPI(
        title=app_config.api.title,
        description=app_config.api.description,
        version=app_config.api.version,
        docs_url="/docs",
        redoc_url="/redoc",
        contact={
            "name": "Electricity Price Analysis API",
            "url": "https://github.com/ahm282/engie-dynamic-price-tracker-api",
        },
        license_info={
            "name": "GNU General Public License v2.0",
            "url": "https://opensource.org/licenses/GPL-2.0"
        },
        openapi_tags=[
            {
                "name": "System Information",
                "description": "API health, version info, and system status endpoints"
            },
            {
                "name": "Price Data",
                "description": "Raw electricity price data retrieval with filtering and pagination"
            },
            {
                "name": "Statistics & Analytics",
                "description": "Aggregated statistics, trends, and analytical insights"
            },
            {
                "name": "Negative Price Analysis",
                "description": "Analysis of negative electricity prices and market anomalies"
            },
            {
                "name": "Expensive Price Monitoring",
                "description": "High-cost period detection, alerts, and consumer impact analysis"
            },
            {
                "name": "Extreme Market Analysis",
                "description": "Market extremes, volatility analysis, and anomaly detection"
            }, 
            {
                "name": "Forecasting Trends",
                "description": "Analysis of price trends, predictions, and forecasting accuracy for better consumer decision-making"
            }
        ]
    )

    # Add CORS middleware with comprehensive configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.api.allow_origins,
        allow_credentials=app_config.api.allow_credentials,
        allow_methods=app_config.api.allow_methods,
        allow_headers=app_config.api.allow_headers,
    )

    # Include electricity price analysis router with organized tags
    app.include_router(
        electricity_controller.router,
        prefix="/api",
    )

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=app_config.api.host,
        port=app_config.api.port,
        reload=app_config.api.reload
    )
