import pandas as pd
import numpy as np
from prophet import Prophet
import json
from pathlib import Path
from prophet.diagnostics import cross_validation, performance_metrics
from ..repositories.prophet_repository import ProphetRepository
from ..utils.cache_utils import cache_forecast
from ..services.price_service import PriceService
from ..weather.weather_collector import WeatherCollector


class ProphetForecastService:
    def __init__(self, repository: ProphetRepository = None):
        self.repository = repository or ProphetRepository()
        self.weather_collector = WeatherCollector()
        self.best_params = None

        params_file = Path("../utils/prophet_models/prophet_best_params.json")

        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)

    def _prepare_data(self, df, include_weather=True):
        """Prepare data with weather integration"""
        df = df.rename(columns={'timestamp': 'ds',
                       'consumer_price_cents_kwh': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.drop_duplicates(subset=['ds']).sort_values(
            'ds').dropna(subset=['y'])

        # Add time-based regressors
        df['hour'] = df['ds'].dt.hour
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)

        # Enhanced peak hour definitions
        df['is_peak'] = ((df['ds'].dt.hour.between(7, 9)) |
                         (df['ds'].dt.hour.between(19, 22))).astype(int)

        # New off-peak window (10-15h)
        df['is_offpeak'] = df['ds'].dt.hour.between(10, 15).astype(int)

        df['month'] = df['ds'].dt.month
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_workday'] = ((df['ds'].dt.dayofweek < 5) &
                            (df['hour'].between(6, 18))).astype(int)

        # Advanced time-based features
        df['price_lag1'] = df['y'].shift(1)
        df['price_lag24'] = df['y'].shift(24)
        df['price_ma7'] = df['y'].rolling(window=7, min_periods=1).mean()
        df['price_volatility'] = df['y'].rolling(
            window=24, min_periods=1).std()
        df['demand_proxy'] = (np.sin(2 * np.pi * df['hour'] / 24) +
                              0.5 * np.sin(2 * np.pi * df['day_of_week'] / 7))

        # Weather integration
        if include_weather:
            try:
                weather_data = self._get_weather_data(df['ds'].tolist())
                df = self._merge_weather_data(df, weather_data)
            except Exception as e:
                print(f"⚠️ Weather data unavailable, using proxy: {e}")
                df = self._add_weather_proxy(df)

        # Remove outliers using dynamic threshold
        rolling_median = df['y'].rolling(
            window=168, min_periods=1).median()  # 7 days
        rolling_std = df['y'].rolling(window=168, min_periods=1).std()
        threshold = 2.5 * rolling_std
        outlier_mask = (df['y'] - rolling_median).abs() <= threshold
        df = df[outlier_mask | df['y'].isna()]

        return df.dropna(subset=['price_lag1', 'price_lag24'])

    def _get_weather_data(self, timestamps):
        """Get weather data for given timestamps"""
        # For historical data, use proxy; for recent/future, try to get real data
        current_time = pd.Timestamp.now()
        historical_timestamps = [ts for ts in timestamps if pd.to_datetime(
            ts) < current_time - pd.Timedelta(hours=24)]

        weather_data = []

        # Get historical weather proxy
        if historical_timestamps:
            historical_weather = self.weather_collector.get_historical_weather_proxy(
                historical_timestamps)
            weather_data.append(historical_weather)

        # Get current weather for recent timestamps
        recent_timestamps = [ts for ts in timestamps if pd.to_datetime(
            ts) >= current_time - pd.Timedelta(hours=24)]
        if recent_timestamps:
            try:
                current_weather = self.weather_collector.get_current_weather()
                recent_weather = pd.DataFrame([{
                    'timestamp': ts,
                    'cloud_cover': current_weather['cloud_cover'],
                    'temperature': current_weather['temperature'],
                    'solar_factor': current_weather['solar_factor']
                } for ts in recent_timestamps])
                weather_data.append(recent_weather)
            except:
                # Fallback to proxy for recent data too
                recent_proxy = self.weather_collector.get_historical_weather_proxy(
                    recent_timestamps)
                weather_data.append(recent_proxy)

        if weather_data:
            return pd.concat(weather_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def _merge_weather_data(self, df, weather_data):
        """Merge weather data with price data"""
        if weather_data.empty:
            return self._add_weather_proxy(df)

        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
        df_with_weather = pd.merge_asof(
            df.sort_values('ds'),
            weather_data.sort_values('timestamp'),
            left_on='ds',
            right_on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(hours=1)
        )

        # Fill missing weather data with seasonal proxy
        missing_weather = df_with_weather[[
            'cloud_cover', 'temperature', 'solar_factor']].isna().any(axis=1)
        if missing_weather.any():
            proxy_data = self._add_weather_proxy(
                df_with_weather[missing_weather])
            df_with_weather.loc[missing_weather, ['cloud_cover', 'temperature', 'solar_factor']] = \
                proxy_data[['cloud_cover', 'temperature', 'solar_factor']].values

        return self._enhance_weather_features(df_with_weather)

    def _add_weather_proxy(self, df):
        """Add weather proxy based on seasonal patterns with proper rounding"""
        df = df.copy()
        hour = df['ds'].dt.hour
        month = df['ds'].dt.month
        day_of_year = df['ds'].dt.dayofyear

        # Seasonal cloud cover model
        # More clouds in winter
        base_clouds = 50 + 30 * np.sin(2 * np.pi * (month - 6) / 12)
        # Less clouds midday
        daily_variation = 20 * np.sin(2 * np.pi * hour / 24)
        cloud_cover = np.clip(
            base_clouds + daily_variation + np.random.normal(0, 5, len(df)), 0, 100)
        df['cloud_cover'] = np.round(cloud_cover, 2)

        # Seasonal temperature model
        temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + \
            5 * np.sin(2 * np.pi * hour / 24)
        df['temperature'] = np.round(temperature, 2)

        # Solar factor based on cloud cover
        solar_factor = np.clip((100 - df['cloud_cover']) / 100, 0, 1)
        df['solar_factor'] = np.round(solar_factor, 2)

        return self._enhance_weather_features(df)

    def _enhance_weather_features(self, df):
        """Create enhanced weather-based features with proper rounding"""
        # Solar production impact
        solar_production = df['solar_factor'] * np.maximum(
            # Solar production curve
            0, np.sin(2 * np.pi * (df['hour'] - 6) / 12)
        )
        df['solar_production_factor'] = np.round(solar_production, 2)

        # Temperature-based demand (heating/cooling)
        temp_demand = np.where(
            df['temperature'] < 15,  # Heating demand
            (15 - df['temperature']) / 10,
            np.where(df['temperature'] > 25,  # Cooling demand
                     (df['temperature'] - 25) / 10, 0)
        )
        df['temp_demand_factor'] = np.round(temp_demand, 2)

        # Weather volatility
        weather_vol = df['cloud_cover'].rolling(
            window=6, min_periods=1).std().fillna(0)
        df['weather_volatility'] = np.round(weather_vol, 2)

        # Combined weather impact on price
        weather_impact = (
            df['temp_demand_factor'] * 0.4 +  # Temperature drives demand
            # Less solar = higher price
            (1 - df['solar_production_factor']) * 0.3 +
            df['weather_volatility'] / 100 * 0.3  # Weather uncertainty
        )
        df['weather_price_impact'] = np.round(weather_impact, 2)

        return df

    def tune_hyperparameters(self):
        """Enhanced hyperparameter tuning with weather features"""
        df = self._prepare_data(self.repository.get_all_data())

        param_grid = [
            {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01,
             'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1,
             'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0,
             'seasonality_mode': 'multiplicative'},
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0,
             'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0,
             'seasonality_mode': 'additive', 'changepoint_range': 0.9},
            # New configurations optimized for weather integration
            {'changepoint_prior_scale': 0.02, 'seasonality_prior_scale': 2.0,
             'seasonality_mode': 'additive', 'changepoint_range': 0.8},
        ]

        best_mape = float('inf')
        for params in param_grid:
            try:
                model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                                yearly_seasonality=False, **params)

                # Add time-based regressors
                for reg in ['hour', 'is_weekend', 'is_peak', 'is_offpeak', 'month', 'is_workday']:
                    model.add_regressor(reg, prior_scale=10.0)

                # Price-based regressors
                for reg in ['price_lag1', 'price_lag24', 'price_ma7', 'demand_proxy']:
                    model.add_regressor(reg, prior_scale=0.5)

                # Weather regressors with optimized prior scales
                weather_regressors = ['cloud_cover', 'temperature', 'solar_factor',
                                      'solar_production_factor', 'temp_demand_factor',
                                      'weather_price_impact']
                for reg in weather_regressors:
                    if reg in df.columns:
                        prior_scale = 1.0 if 'impact' in reg else 0.3
                        model.add_regressor(reg, prior_scale=prior_scale)

                model.add_regressor('price_volatility', prior_scale=0.1)
                model.add_regressor('weather_volatility', prior_scale=0.2)

                # Enhanced seasonalities
                model.add_seasonality(
                    name='hourly', period=24, fourier_order=8)
                model.add_seasonality(
                    name='daily_weather', period=24, fourier_order=3)

                model.fit(df)

                cv_results = cross_validation(
                    model, initial='30 days', period='5 days', horizon='24 hours'
                )
                mape = performance_metrics(cv_results)['mape'].mean()

                if mape < best_mape:
                    best_mape = mape
                    self.best_params = params
                    print(f"🎯 New best MAPE: {mape:.4f} with params: {params}")

            except Exception as e:
                print(f"⚠️ Parameter combination failed: {e}")
                continue

        # Save best parameters
        if self.best_params:
            PARAMS_FILE = Path(
                "../utils/prophet_models/prophet_best_params.json")
            PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PARAMS_FILE, 'w') as f:
                json.dump(self.best_params, f)
            print(f"💾 Best parameters saved with MAPE: {best_mape:.4f}")

        return self.best_params

    @cache_forecast(ttl_seconds=3600)
    def forecast(self, hours_ahead: int = 48) -> list:
        """
        Generate weather-enhanced forecasts starting from the last available data point.

        Args:
            hours_ahead: Number of hours to predict into the future

        Returns:
            List of forecast dictionaries with predictions
        """
        # Get data with weather integration
        df = self._prepare_data(self.repository.get_all_data())

        if df.empty:
            raise ValueError("No data available for forecasting")

        last_timestamp = df['ds'].max()
        data_summary = self.repository.get_data_summary()

        print(f"📊 Weather-enhanced forecasting from: {last_timestamp}")
        print(
            f"📈 Database contains {data_summary.get('total_records', 0)} records")
        print(
            f"🌤️ Weather features integrated: {len([col for col in df.columns if 'weather' in col or col in ['cloud_cover', 'temperature', 'solar_factor']])}")

        # Use tuned params or enhanced defaults
        params = self.best_params or {
            'changepoint_prior_scale': 0.02,
            'seasonality_prior_scale': 2.0,
            'seasonality_mode': 'additive'
        }

        model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                        yearly_seasonality=False, **params)

        # Add all regressors
        time_regressors = ['hour', 'is_weekend',
                           'is_peak', 'is_offpeak', 'month', 'is_workday']
        for reg in time_regressors:
            model.add_regressor(reg, prior_scale=10.0)

        price_regressors = ['price_lag1',
                            'price_lag24', 'price_ma7', 'demand_proxy']
        for reg in price_regressors:
            model.add_regressor(reg, prior_scale=0.5)

        # Weather regressors
        weather_regressors = ['cloud_cover', 'temperature', 'solar_factor',
                              'solar_production_factor', 'temp_demand_factor',
                              'weather_price_impact']
        for reg in weather_regressors:
            if reg in df.columns:
                prior_scale = 1.0 if 'impact' in reg else 0.3
                model.add_regressor(reg, prior_scale=prior_scale)

        model.add_regressor('price_volatility', prior_scale=0.1)
        if 'weather_volatility' in df.columns:
            model.add_regressor('weather_volatility', prior_scale=0.2)

        # Enhanced seasonalities
        model.add_seasonality(name='hourly', period=24, fourier_order=8)
        model.add_seasonality(name='daily_weather', period=24, fourier_order=3)

        model.fit(df)

        # Create future dataframe with weather forecast
        future = model.make_future_dataframe(periods=hours_ahead, freq='h')
        future = self._add_future_regressors(future, df, hours_ahead)

        forecast = model.predict(future)

        # Get only future predictions
        training_length = len(df)
        results = forecast[['ds', 'yhat', 'yhat_lower',
                            'yhat_upper']].tail(hours_ahead).copy()

        # Ensure predictions are truly future
        future_mask = results['ds'] > last_timestamp
        results = results[future_mask].copy()

        print(f"🔮 Generated {len(results)} weather-enhanced predictions")

        # Enhanced post-processing
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            results[col] = results[col].clip(lower=0).round(3)

        # Enhanced confidence scoring
        results['confidence'] = self._calculate_confidence(results, df)
        results['date'] = results['ds'].dt.date.astype(str)
        results['hour'] = results['ds'].dt.hour

        # Add period categorization
        results['period_type'] = results['hour'].apply(self._categorize_period)

        results = results.rename(columns={
            'ds': 'timestamp', 'yhat': 'predicted_price_cents_kwh',
            'yhat_lower': 'lower_bound_cents_kwh', 'yhat_upper': 'upper_bound_cents_kwh'
        })

        results['price_category'] = results['predicted_price_cents_kwh'].apply(
            PriceService.categorize_price
        )
        results['timestamp'] = results['timestamp'].astype(str)

        return results.to_dict(orient='records')

    def _add_future_regressors(self, future, df, hours_ahead):
        """Add regressors for future predictions with weather forecast"""
        # Basic time regressors
        future['hour'] = future['ds'].dt.hour
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        future['is_peak'] = ((future['ds'].dt.hour.between(7, 9)) |
                             (future['ds'].dt.hour.between(19, 22))).astype(int)
        future['is_offpeak'] = future['ds'].dt.hour.between(10, 15).astype(int)
        future['month'] = future['ds'].dt.month
        future['day_of_week'] = future['ds'].dt.dayofweek
        future['is_workday'] = ((future['ds'].dt.dayofweek < 5) &
                                (future['hour'].between(6, 18))).astype(int)

        training_length = len(df)

        # Fill historical regressors
        for col in ['price_lag1', 'price_lag24', 'price_ma7', 'price_volatility']:
            if col in df.columns:
                future.loc[:training_length-1, col] = df[col].values

        # Estimate future price-based regressors
        last_price = df['y'].iloc[-1] if len(df) > 0 else 50
        last_price_24h = df['y'].iloc[-24] if len(df) > 24 else df['y'].mean()
        last_ma7 = df['price_ma7'].iloc[-1] if 'price_ma7' in df.columns else last_price
        last_volatility = df['price_volatility'].iloc[-1] if 'price_volatility' in df.columns else 5

        future.loc[training_length:, 'price_lag1'] = last_price
        future.loc[training_length:, 'price_lag24'] = last_price_24h
        future.loc[training_length:, 'price_ma7'] = last_ma7
        future.loc[training_length:, 'price_volatility'] = last_volatility

        future['demand_proxy'] = (np.sin(2 * np.pi * future['hour'] / 24) +
                                  0.5 * np.sin(2 * np.pi * future['day_of_week'] / 7))

        # Get weather forecast for future periods
        future_timestamps = future.loc[training_length:, 'ds'].tolist()
        if future_timestamps:
            try:
                # Use the standard weather forecast method
                weather_forecast = self.weather_collector.get_forecast(
                    hours=len(future_timestamps) * 3)
                if not weather_forecast.empty:
                    weather_forecast['timestamp'] = pd.to_datetime(
                        weather_forecast['timestamp'])
                    future_weather = pd.merge_asof(
                        future.loc[training_length:, ['ds']].reset_index(),
                        weather_forecast,
                        left_on='ds', right_on='timestamp',
                        direction='nearest'
                    )
                    if not future_weather.empty:
                        for col in ['cloud_cover', 'temperature', 'solar_factor']:
                            if col in future_weather.columns:
                                future.loc[training_length:,
                                           col] = future_weather[col].values
                else:
                    # Use weather proxy for future
                    future_proxy = self._add_weather_proxy(
                        future.loc[training_length:])
                    for col in ['cloud_cover', 'temperature', 'solar_factor']:
                        if col in future_proxy.columns:
                            future.loc[training_length:,
                                       col] = future_proxy[col].values
            except Exception as e:
                print(f"⚠️ Weather forecast failed, using proxy: {e}")
                # Fallback to proxy
                future_proxy = self._add_weather_proxy(
                    future.loc[training_length:])
                for col in ['cloud_cover', 'temperature', 'solar_factor']:
                    if col in future_proxy.columns:
                        future.loc[training_length:,
                                   col] = future_proxy[col].values

        # Fill historical weather data
        for col in ['cloud_cover', 'temperature', 'solar_factor']:
            if col in df.columns:
                future.loc[:training_length-1, col] = df[col].values

        # Calculate enhanced weather features for all periods
        future = self._enhance_weather_features(future)

        return future

    def _calculate_confidence(self, results, df):
        """Calculate enhanced confidence scores"""
        # Base confidence from prediction intervals
        denom = results['yhat'].replace(0, np.nan)
        interval_confidence = np.clip(
            1 - (results['yhat_upper'] - results['yhat_lower']) / denom, 0, 1
        ).fillna(0)

        # Weather-based confidence adjustment
        weather_confidence = 1.0
        if 'weather_volatility' in df.columns:
            recent_weather_vol = df['weather_volatility'].tail(24).mean()
            weather_confidence = np.clip(1 - recent_weather_vol / 50, 0.5, 1.0)

        # Time-based confidence (higher confidence for near-term predictions)
        time_confidence = np.exp(-0.05 * np.arange(len(results)))

        # Combined confidence
        combined_confidence = (
            interval_confidence * 0.5 +
            weather_confidence * 0.3 +
            time_confidence * 0.2
        )

        return combined_confidence.round(3)

    def _categorize_period(self, hour):
        """Categorize time periods including off-peak"""
        if 7 <= hour <= 9 or 19 <= hour <= 22:
            return 'peak'
        elif 10 <= hour <= 15:
            return 'off-peak'
        elif 22 <= hour <= 6:
            return 'night'
        else:
            return 'standard'

    def get_weather_impact_analysis(self):
        """Analyze weather impact on price predictions"""
        df = self._prepare_data(self.repository.get_all_data())

        if df.empty:
            return {}

        # Calculate correlations
        weather_cols = ['cloud_cover', 'temperature', 'solar_factor',
                        'solar_production_factor', 'temp_demand_factor',
                        'weather_price_impact']

        correlations = {}
        for col in weather_cols:
            if col in df.columns:
                correlations[col] = df[col].corr(df['y'])

        return {
            'weather_correlations': correlations,
            'avg_solar_factor': df['solar_factor'].mean() if 'solar_factor' in df.columns else None,
            'avg_temp_demand': df['temp_demand_factor'].mean() if 'temp_demand_factor' in df.columns else None,
            'weather_coverage': len([col for col in weather_cols if col in df.columns]) / len(weather_cols)
        }
