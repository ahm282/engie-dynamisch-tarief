import pandas as pd
import numpy as np
from prophet import Prophet
import json
from pathlib import Path
from prophet.diagnostics import cross_validation, performance_metrics
from ..repositories.prophet_repository import ProphetRepository
from ..utils.cache_utils import cache_forecast
from ..services.price_service import PriceService


class ProphetForecastService:
    def __init__(self, repository: ProphetRepository = None):
        self.repository = repository or ProphetRepository()
        self.best_params = None

        params_file = Path(
            "../utils/prophet_models/prophet_best_params.json")

        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)

    def _prepare_data(self, df):
        df = df.rename(columns={'timestamp': 'ds',
                       'consumer_price_cents_kwh': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.drop_duplicates(subset=['ds']).sort_values(
            'ds').dropna(subset=['y'])

        # Add regressors
        df['hour'] = df['ds'].dt.hour
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        df['is_peak'] = ((df['ds'].dt.hour.between(7, 9)) | (
            df['ds'].dt.hour.between(17, 19))).astype(int)
        df['month'] = df['ds'].dt.month
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_workday'] = ((df['ds'].dt.dayofweek < 5) & (
            df['hour'].between(6, 18))).astype(int)

        # Advanced features
        df['price_lag1'] = df['y'].shift(1)
        df['price_lag24'] = df['y'].shift(24)
        df['price_ma7'] = df['y'].rolling(window=7, min_periods=1).mean()
        df['price_volatility'] = df['y'].rolling(
            window=24, min_periods=1).std()
        df['demand_proxy'] = np.sin(
            2 * np.pi * df['hour'] / 24) + 0.5 * np.sin(2 * np.pi * df['day_of_week'] / 7)

        # Remove outliers using dynamic threshold
        rolling_median = df['y'].rolling(
            window=168, min_periods=1).median()  # 7 days
        rolling_std = df['y'].rolling(window=168, min_periods=1).std()
        threshold = 2.5 * rolling_std
        outlier_mask = (df['y'] - rolling_median).abs() <= threshold
        df = df[outlier_mask | df['y'].isna()]

        return df.dropna(subset=['price_lag1', 'price_lag24'])

    def tune_hyperparameters(self):
        df = self._prepare_data(self.repository.get_all_data())
        param_grid = [
            {'changepoint_prior_scale': 0.001,
                'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1,
                'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0,
                'seasonality_mode': 'multiplicative'},
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.9}
        ]

        best_mape = float('inf')
        for params in param_grid:
            try:
                model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                                yearly_seasonality=False, **params)

                # Add regressors with different prior scales
                for reg in ['hour', 'is_weekend', 'is_peak', 'month', 'is_workday']:
                    model.add_regressor(reg, prior_scale=10.0)
                for reg in ['price_lag1', 'price_lag24', 'price_ma7', 'demand_proxy']:
                    model.add_regressor(reg, prior_scale=0.5)
                model.add_regressor('price_volatility', prior_scale=0.1)

                model.add_seasonality(
                    name='hourly', period=24, fourier_order=8)
                model.fit(df)

                cv_results = cross_validation(
                    model, initial='30 days', period='5 days', horizon='24 hours')
                mape = performance_metrics(cv_results)['mape'].mean()

                if mape < best_mape:
                    best_mape = mape
                    self.best_params = params

                PARAMS_FILE = Path(
                    "../utils/prophet_models/prophet_best_params.json")

                if self.best_params:
                    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(PARAMS_FILE, 'w') as f:
                        json.dump(self.best_params, f)
            except:
                continue
        return self.best_params

    @cache_forecast(ttl_seconds=3600)
    def forecast(self, hours_ahead: int = 48) -> list:
        """
        Generate forecasts starting from the last available data point in the database.

        Args:
            hours_ahead: Number of hours to predict into the future

        Returns:
            List of forecast dictionaries with predictions
        """
        # Get data and ensure we know the last timestamp
        df = self._prepare_data(self.repository.get_all_data())

        if df.empty:
            raise ValueError("No data available for forecasting")

        # Get the last timestamp from the actual data
        last_timestamp = df['ds'].max()
        data_summary = self.repository.get_data_summary()

        print(
            f"📊 Forecasting from last available data point: {last_timestamp}")
        print(
            f"📈 Database contains {data_summary.get('total_records', 0)} records")
        print(
            f"📅 Data range: {data_summary.get('first_timestamp', 'N/A')} to {data_summary.get('latest_timestamp', 'N/A')}")

        # Use tuned params or defaults
        params = self.best_params or {'changepoint_prior_scale': 0.05,
                                      'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}

        model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                        yearly_seasonality=False, **params)

        # Add regressors with optimized prior scales
        for reg in ['hour', 'is_weekend', 'is_peak', 'month', 'is_workday']:
            model.add_regressor(reg, prior_scale=10.0)
        for reg in ['price_lag1', 'price_lag24', 'price_ma7', 'demand_proxy']:
            model.add_regressor(reg, prior_scale=0.5)
        model.add_regressor('price_volatility', prior_scale=0.1)

        model.add_seasonality(name='hourly', period=24, fourier_order=8)
        model.fit(df)

        # Create future dataframe starting from the last data point
        future = model.make_future_dataframe(periods=hours_ahead, freq='h')

        # Add basic regressors
        future['hour'] = future['ds'].dt.hour
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        future['is_peak'] = ((future['ds'].dt.hour.between(7, 9)) |
                             (future['ds'].dt.hour.between(17, 19))).astype(int)
        future['month'] = future['ds'].dt.month
        future['day_of_week'] = future['ds'].dt.dayofweek
        future['is_workday'] = ((future['ds'].dt.dayofweek < 5) & (
            future['hour'].between(6, 18))).astype(int)

        # Add advanced regressors for future (use last known values for lags)
        last_price = df['y'].iloc[-1]
        last_price_24h = df['y'].iloc[-24] if len(df) > 24 else df['y'].mean()
        last_ma7 = df['price_ma7'].iloc[-1]
        last_volatility = df['price_volatility'].iloc[-1]

        # For future periods, we need to handle the lagged features carefully
        # For the training period, use actual values; for future, use approximations
        training_length = len(df)

        # Fill in historical values for regressors (for training period)
        future.loc[:training_length-1, 'price_lag1'] = df['price_lag1'].values
        future.loc[:training_length-1,
                   'price_lag24'] = df['price_lag24'].values
        future.loc[:training_length-1, 'price_ma7'] = df['price_ma7'].values
        future.loc[:training_length-1,
                   'price_volatility'] = df['price_volatility'].values

        # For future periods, use estimates
        future.loc[training_length:, 'price_lag1'] = last_price
        future.loc[training_length:, 'price_lag24'] = last_price_24h
        future.loc[training_length:, 'price_ma7'] = last_ma7
        future.loc[training_length:, 'price_volatility'] = last_volatility

        future['demand_proxy'] = np.sin(
            2 * np.pi * future['hour'] / 24) + 0.5 * np.sin(2 * np.pi * future['day_of_week'] / 7)

        forecast = model.predict(future)

        # Get only the future predictions (after the last training data point)
        results = forecast[['ds', 'yhat', 'yhat_lower',
                            'yhat_upper']].tail(hours_ahead).copy()

        # Ensure we're truly getting predictions from after the last data point
        future_mask = results['ds'] > last_timestamp
        results = results[future_mask].copy()

        print(
            f"🔮 Generated {len(results)} predictions starting from {results['ds'].min() if not results.empty else 'N/A'}")

        # Enhanced post-processing
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            results[col] = results[col].clip(lower=0).round(3)

        # Add confidence score based on volatility
        denom = results['yhat'].replace(0, np.nan)
        results['confidence'] = np.clip(
            1 - (results['yhat_upper'] - results['yhat_lower']) / denom, 0, 1
        ).fillna(0).round(3)

        results['date'] = results['ds'].dt.date.astype(str)
        results['hour'] = results['ds'].dt.hour
        results = results.rename(columns={
            'ds': 'timestamp', 'yhat': 'predicted_price_cents_kwh',
            'yhat_lower': 'lower_bound_cents_kwh', 'yhat_upper': 'upper_bound_cents_kwh'
        })
        results['price_category'] = results['predicted_price_cents_kwh'].apply(
            PriceService.categorize_price
        )
        results['timestamp'] = results['timestamp'].astype(str)

        return results.to_dict(orient='records')
