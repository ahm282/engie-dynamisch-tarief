import pandas as pd
from prophet import Prophet
from ..repositories.prophet_repository import ProphetRepository
from ..utils.cache_utils import cache_forecast


class ProphetForecastService:
    """Service layer for Prophet-based forecasting."""

    def __init__(self, repository: ProphetRepository = None):
        self.repository = repository or ProphetRepository()

    @cache_forecast(ttl_seconds=3600)  # Cache for 1 hour
    def forecast(self, hours_ahead: int = 48) -> list:
        df = self.repository.get_all_data()

        df = df.rename(columns={
            'timestamp': 'ds',
            'consumer_price_cents_kwh': 'y'
        })
        df['ds'] = pd.to_datetime(df['ds'])

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='additive'
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=hours_ahead, freq='h')
        forecast = model.predict(future)

        forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(
            hours_ahead).copy()

        # Round values
        forecast_results['yhat'] = forecast_results['yhat'].round(3)
        forecast_results['yhat_lower'] = forecast_results['yhat_lower'].round(
            3)
        forecast_results['yhat_upper'] = forecast_results['yhat_upper'].round(
            3)

        # Add 'date' and 'hour' columns
        forecast_results['date'] = forecast_results['ds'].dt.date.astype(str)
        forecast_results['hour'] = forecast_results['ds'].dt.hour

        # Rename columns as requested
        forecast_results = forecast_results.rename(columns={
            'ds': 'timestamp',
            'yhat': 'predicted_price_cents_kwh',
            'yhat_lower': 'lower_bound_cents_kwh',
            'yhat_upper': 'upper_bound_cents_kwh'
        })

        # Convert 'timestamp' to ISO string for JSON
        forecast_results['timestamp'] = forecast_results['timestamp'].astype(
            str)

        # Return list of dicts (JSON serializable)
        return forecast_results.to_dict(orient='records')
