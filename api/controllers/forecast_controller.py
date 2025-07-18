from fastapi import APIRouter, HTTPException
from ..services.prophet_service import ProphetForecastService
from ..models.forecast_models import ForecastResponse


class ForecastController:
    """Handles endpoints related to forecasting."""

    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        @self.router.get("/forecast/prophet", response_model=ForecastResponse)
        def run_prophet_forecast(hours_ahead: int = 48):
            try:
                service = ProphetForecastService()
                forecast_data = service.forecast(hours_ahead)
                return ForecastResponse.from_list(forecast_data)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
