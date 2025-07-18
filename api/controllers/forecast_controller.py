from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.prophet_service import ProphetForecastService
from ..models.forecast_models import ForecastResponse
from ..repositories.prophet_repository import ProphetRepository


class ForecastController:
    """Handles endpoints related to forecasting."""

    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        @self.router.get("/forecast/prophet", response_model=ForecastResponse)
        def run_prophet_forecast(
            hours_ahead: int = Query(
                48,
                description="Number of hours to forecast into the future",
                ge=1,
                le=168  # Max 7 days
            )
        ):
            """
            Generate electricity price forecasts using Prophet model.

            This endpoint uses the Prophet forecasting model to predict electricity prices
            starting from the last available data point in the database.

            Features:
            - Uses advanced time series forecasting with Prophet
            - Incorporates seasonal patterns (hourly, daily, weekly)
            - Includes price volatility and demand proxy indicators
            - Provides confidence intervals for each prediction
            - Categorizes predictions (cheap, regular, expensive, extremely expensive)

            Args:
                hours_ahead: Number of hours to forecast (1-168 hours, default: 48)

            Returns:
                ForecastResponse: List of forecasted price points with timestamps,
                predictions, confidence intervals, and price categories
            """
            try:
                service = ProphetForecastService()
                forecast_data = service.forecast(hours_ahead)
                return ForecastResponse.from_list(forecast_data)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Forecasting error: {str(e)}")

        @self.router.get("/forecast/info")
        def get_forecast_info():
            """
            Get information about the available data for forecasting.

            Returns summary statistics about the database and forecasting capabilities.
            """
            try:
                repository = ProphetRepository()
                data_summary = repository.get_data_summary()

                return {
                    "data_summary": data_summary,
                    "forecast_capabilities": {
                        "model_type": "Prophet",
                        "max_hours_ahead": 168,
                        "default_hours_ahead": 48,
                        "features": [
                            "Hourly seasonality",
                            "Weekly seasonality",
                            "Price volatility indicators",
                            "Demand proxy patterns",
                            "Confidence intervals",
                            "Price categorization"
                        ]
                    },
                    "data_status": "ready" if data_summary.get('total_records', 0) > 0 else "no_data"
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error getting forecast info: {str(e)}")
