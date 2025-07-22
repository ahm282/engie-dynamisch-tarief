import pandas as pd
import numpy as np
from prophet import Prophet
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
from prophet.diagnostics import cross_validation, performance_metrics
from ..repositories.prophet_repository import ProphetRepository
from ..utils.cache_utils import cache_forecast
from ..services.price_service import PriceService
from ..weather.weather_collector import WeatherCollector

# Enhanced ML imports for ensemble modeling
try:
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost/sklearn not available - falling back to Prophet-only forecasting")
    ML_AVAILABLE = False


class ProphetForecastService:
    def __init__(self, repository: ProphetRepository = None, enable_enhancements=True):
        self.repository = repository or ProphetRepository()
        self.weather_collector = WeatherCollector()
        self.best_params = None

        # Load enhanced configuration
        self.enable_enhancements = enable_enhancements
        self.config = self._load_enhanced_config() if enable_enhancements else {}

        # Setup enhanced logging and monitoring
        if enable_enhancements:
            self.logger = self._setup_enhanced_logging()
            self.performance_metrics = {
                'prediction_count': 0,
                'total_execution_time': 0.0,
                'error_count': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            self._feature_cache = {}
            self._data_quality_cache = {}
        else:
            self.logger = logging.getLogger('prophet_service')

        params_file = Path("../utils/prophet_models/prophet_best_params.json")
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)

    def _load_enhanced_config(self) -> Dict[str, Any]:
        """Load enhanced configuration with intelligent defaults"""
        default_config = {
            "data_quality": {
                "enable_validation": True,
                "min_data_points": 100,
                "max_missing_percentage": 0.1,
                "outlier_threshold": 3.0,
                "quality_threshold": 0.7,
                "auto_fix_issues": True
            },
            "performance": {
                "enable_caching": True,
                "enable_parallel_training": True,
                "cache_expiry_minutes": 30,
                "max_workers": 4
            },
            "model_settings": {
                "default_forecast_hours": 48,
                "enable_ml_models": True,
                "enable_xgboost": True,
                "enable_lightgbm": True,
                "ensemble_weights": {
                    "prophet": 0.4,
                    "xgboost": 0.35,
                    "lightgbm": 0.25
                }
            },
            "monitoring": {
                "track_performance": True,
                "log_predictions": True,
                "log_level": "INFO"
            }
        }

        config_file = Path("config/prophet_enhanced_config.json")
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Deep merge configurations
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Could not load config: {e}")

        return default_config

    def _setup_enhanced_logging(self) -> logging.Logger:
        """Setup enhanced logging for the service"""
        logger = logging.getLogger('enhanced_prophet_service')

        if not logger.handlers:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Setup file handler
            handler = logging.FileHandler(log_dir / "prophet_predictions.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def _monitor_performance(self, func):
        """Performance monitoring decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_enhancements or not self.config.get("monitoring", {}).get("track_performance", False):
                return func(*args, **kwargs)

            start_time = time.time()
            self.performance_metrics['prediction_count'] += 1

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.performance_metrics['total_execution_time'] += execution_time

                self.logger.info(
                    f"{func.__name__} completed in {execution_time:.2f}s")
                return result

            except Exception as e:
                self.performance_metrics['error_count'] += 1
                self.logger.error(f"{func.__name__} failed: {e}")
                raise

        return wrapper

    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced data quality validation"""
        if not self.enable_enhancements or not self.config.get("data_quality", {}).get("enable_validation", False):
            return True, {"validation_skipped": True}

        quality_issues = []
        quality_score = 1.0

        # Check minimum data points
        min_points = self.config["data_quality"].get("min_data_points", 100)
        if len(df) < min_points:
            quality_issues.append(
                f"Insufficient data: {len(df)} < {min_points}")
            quality_score -= 0.3

        # Check missing values
        if 'y' in df.columns:
            missing_pct = df['y'].isna().sum() / len(df)
            max_missing = self.config["data_quality"].get(
                "max_missing_percentage", 0.1)
            if missing_pct > max_missing:
                quality_issues.append(
                    f"Too many missing values: {missing_pct:.2%}")
                quality_score -= 0.3

                # Auto-fix if enabled
                if self.config["data_quality"].get("auto_fix_issues", False):
                    df['y'] = df['y'].fillna(df['y'].median())
                    quality_issues.append("Auto-fixed missing values")

        # Check for outliers
        if 'y' in df.columns and len(df) > 10:
            z_threshold = self.config["data_quality"].get(
                "outlier_threshold", 3.0)
            z_scores = np.abs((df['y'] - df['y'].mean()) / df['y'].std())
            outlier_mask = z_scores > z_threshold
            outlier_pct = outlier_mask.sum() / len(df)

            if outlier_pct > 0.05:
                quality_issues.append(
                    f"High outlier percentage: {outlier_pct:.2%}")
                quality_score -= 0.2

        quality_score = max(0, quality_score)
        is_valid = quality_score >= self.config["data_quality"].get(
            "quality_threshold", 0.7)

        quality_report = {
            "overall_score": quality_score,
            "issues": quality_issues,
            "data_points": len(df),
            "missing_percentage": missing_pct if 'y' in df.columns else 0
        }

        if quality_issues and self.enable_enhancements:
            self.logger.warning(f"Data quality issues: {quality_issues}")

        return is_valid, quality_report

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.enable_enhancements:
            return {"enhancements_disabled": True}

        total_predictions = self.performance_metrics['prediction_count']
        avg_time = (
            self.performance_metrics['total_execution_time'] / max(1, total_predictions))

        return {
            "total_predictions": total_predictions,
            "average_execution_time_seconds": avg_time,
            "error_count": self.performance_metrics['error_count'],
            "error_rate": self.performance_metrics['error_count'] / max(1, total_predictions),
            "cache_hit_rate": self.performance_metrics['cache_hits'] / max(1,
                                                                           self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
        }

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary"""
        if not self.enable_enhancements:
            return {"enhancements_disabled": True}
        return self.config

    def _prepare_data(self, df, include_weather=True):
        """Prepare data with weather integration and off-peak optimization"""
        df = df.rename(columns={'timestamp': 'ds',
                       'consumer_price_cents_kwh': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.drop_duplicates(subset=['ds']).sort_values(
            'ds').dropna(subset=['y'])

        # Add time-based regressors
        df['hour'] = df['ds'].dt.hour
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        df['is_peak'] = ((df['ds'].dt.hour.between(7, 9)) | (
            df['ds'].dt.hour.between(19, 22))).astype(int)
        df['is_offpeak'] = df['ds'].dt.hour.between(10, 15).astype(int)
        df['recent_weight'] = np.exp(-(df['ds'].max() - df['ds']).dt.days / 14)
        df['month'] = df['ds'].dt.month
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_workday'] = ((df['ds'].dt.dayofweek < 5) & (
            df['hour'].between(6, 18))).astype(int)

        # Advanced time-based features with NaN handling
        df['price_lag1'] = df['y'].shift(1)
        df['price_lag24'] = df['y'].shift(24)
        df['price_ma7'] = df['y'].rolling(window=7, min_periods=1).mean()
        df['offpeak_ma3'] = df[df['is_offpeak'] == 1]['y'].rolling(
            window=3, min_periods=1).mean().ffill().bfill()
        df['price_volatility'] = df['y'].rolling(
            window=24, min_periods=1).std()
        df['demand_proxy'] = (np.sin(
            2 * np.pi * df['hour'] / 24) + 0.5 * np.sin(2 * np.pi * df['day_of_week'] / 7))

        # Fill NaN values for lag features
        df['price_lag1'] = df['price_lag1'].bfill()
        df['price_lag24'] = df['price_lag24'].fillna(df['y'].mean())
        df['price_ma7'] = df['price_ma7'].fillna(df['y'].mean())
        df['offpeak_ma3'] = df['offpeak_ma3'].fillna(df['y'].mean())
        df['price_volatility'] = df['price_volatility'].fillna(
            5.0)  # Default volatility

        # Off-peak specific features
        df['solar_peak_hours'] = (
            (df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        df['is_midday_solar'] = (
            (df['hour'] >= 12) & (df['hour'] <= 13)).astype(int)
        df['solar_seasonality_condition'] = df['solar_peak_hours'].copy()

        # Weather integration
        if include_weather:
            try:
                weather_data = self._get_weather_data(df['ds'].tolist())
                df = self._merge_weather_data(df, weather_data)
            except Exception as e:
                print(f"⚠️ Weather data unavailable, using proxy: {e}")
                df = self._add_weather_proxy(df)

        # Summer solar feature
        df['summer_extreme_solar'] = ((df['month'].between(6, 8)) & (
            df['hour'].between(11, 15)) & (df['solar_factor'] > 0.7)).astype(int)
        return df

    def _get_weather_data(self, timestamps):
        """Get actual weather data for given timestamps"""
        try:
            weather_df = self.weather_collector.get_historical_weather_proxy(
                timestamps)
            if weather_df.empty:
                print("⚠️ No weather data available from collector")
                return pd.DataFrame()

            weather_df['timestamp'] = pd.to_datetime(
                weather_df['timestamp']).dt.tz_localize(None)
            return weather_df
        except Exception as e:
            print(f"⚠️ Weather collector failed: {e}")
            return pd.DataFrame()

    def _merge_weather_data(self, df, weather_data):
        """Merge weather data with main dataframe"""
        if weather_data.empty:
            return self._add_weather_proxy(df)

        try:
            merged = pd.merge_asof(df.sort_values('ds'), weather_data.sort_values('timestamp'),
                                   left_on='ds', right_on='timestamp', direction='nearest')

            if len(merged) < len(df) * 0.8:
                print("⚠️ Poor weather data coverage, using proxy")
                return self._add_weather_proxy(df)

            return self._enhance_weather_features(merged)
        except Exception as e:
            print(f"⚠️ Weather merge failed: {e}")
            return self._add_weather_proxy(df)

    def _add_weather_proxy(self, df):
        """Create weather proxy when actual data unavailable"""
        df = df.copy()

        # Seasonal and daily patterns
        day_of_year = df['ds'].dt.dayofyear
        hour = df['hour']

        # Cloud cover model (higher in winter, lower in summer)
        cloud_cover = 60 + 20 * np.sin(2 * np.pi * (day_of_year - 172) / 365) + \
            10 * np.sin(2 * np.pi * hour / 24) + \
            np.random.normal(0, 10, len(df))
        df['cloud_cover'] = np.clip(cloud_cover, 0, 100).round(2)

        # Temperature model
        temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + \
            5 * np.sin(2 * np.pi * hour / 24)
        df['temperature'] = temperature.round(2)

        # Solar factor with stronger effect
        solar_factor = np.clip(((100 - df['cloud_cover']) / 100) ** 2.5, 0, 1)
        df['solar_factor'] = solar_factor.round(2)

        return self._enhance_weather_features(df)

    def _enhance_weather_features(self, df):
        """Create enhanced weather-based features with off-peak optimizations"""
        if 'hour' not in df.columns:
            df['hour'] = df['ds'].dt.hour
        if 'is_offpeak' not in df.columns:
            df['is_offpeak'] = df['ds'].dt.hour.between(10, 15).astype(int)

        # Solar production impact
        solar_production = df['solar_factor'] * \
            np.maximum(0, np.sin(2 * np.pi * (df['hour'] - 6) / 12) ** 2)
        df['solar_production_factor'] = solar_production.round(2)

        # Off-peak solar impact
        df['offpeak_solar_impact'] = np.where(
            df['is_offpeak'] == 1, df['solar_production_factor'] * -20.0, 0).round(2)

        # Sharp midday solar collapse
        df['midday_solar_collapse'] = np.where((df['hour'].between(13, 15)) & (df['solar_factor'] > 0.6),
                                               df['solar_factor'] * -25.0, 0).round(2)

        # Solar oversupply indicator
        df['solar_oversupply'] = np.where(
            (df['solar_factor'] > 0.8) & (df['hour'].between(12, 15)), 1, 0)

        # Temperature-based demand
        temp_demand = np.where(df['temperature'] < 15, (15 - df['temperature']) / 10,
                               np.where(df['temperature'] > 25, (df['temperature'] - 25) / 10, 0))
        df['temp_demand_factor'] = temp_demand.round(2)

        # Weather volatility
        weather_vol = df['cloud_cover'].rolling(
            window=6, min_periods=1).std().fillna(0)
        df['weather_volatility'] = weather_vol.round(2)
        df['offpeak_weather_vol'] = np.where(
            df['is_offpeak'] == 1, weather_vol * 1.5, weather_vol).round(2)

        # Combined weather impact on price
        base_impact = (df['temp_demand_factor'] * 0.4 + (1 - df['solar_production_factor']) * 1.2 +
                       df['weather_volatility'] / 100 * 0.05)
        df['weather_price_impact'] = np.where(
            df['is_offpeak'] == 1, base_impact + df['offpeak_solar_impact'] * 4.0, base_impact).round(2)

        return df

    def tune_hyperparameters(self):
        """Enhanced hyperparameter tuning with weather features"""
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
                'seasonality_mode': 'additive', 'changepoint_range': 0.9},
            {'changepoint_prior_scale': 0.02, 'seasonality_prior_scale': 2.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.8},
        ]

        best_mape = float('inf')
        for params in param_grid:
            try:
                model = Prophet(**params)
                self._add_prophet_regressors(model, df)
                model.fit(df)
                cv_df = cross_validation(
                    model, horizon='48 hours', parallel="processes")
                performance = performance_metrics(cv_df)
                mape = performance['mape'].mean()
                if mape < best_mape:
                    best_mape = mape
                    self.best_params = params
            except Exception as e:
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

    def _create_ml_features(self, df, prefix=""):
        """Create comprehensive feature set for ML models with enhanced price-aware features"""
        features = {}

        # Time-based features with stronger periodicity
        features[f'{prefix}hour'] = df['hour']
        features[f'{prefix}hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        features[f'{prefix}hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        features[f'{prefix}day_of_week'] = df['day_of_week']
        features[f'{prefix}day_sin'] = np.sin(
            2 * np.pi * df['day_of_week'] / 7)
        features[f'{prefix}day_cos'] = np.cos(
            2 * np.pi * df['day_of_week'] / 7)
        features[f'{prefix}month'] = df['month']
        features[f'{prefix}month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features[f'{prefix}month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Enhanced binary indicators
        features[f'{prefix}is_weekend'] = df['is_weekend']
        features[f'{prefix}is_peak'] = df['is_peak']
        features[f'{prefix}is_offpeak'] = df['is_offpeak']
        features[f'{prefix}is_workday'] = df['is_workday']
        features[f'{prefix}solar_peak_hours'] = df['solar_peak_hours']
        features[f'{prefix}is_midday_solar'] = df['is_midday_solar']

        # Weather features
        weather_cols = ['cloud_cover', 'temperature',
                        'solar_factor', 'solar_production_factor']
        for col in weather_cols:
            if col in df.columns:
                features[f'{prefix}{col}'] = df[col]

        # Price-based features (lagged values) with NaN handling
        if 'price_lag1' in df.columns:
            price_lag1_clean = df['price_lag1'].fillna(df['price_lag1'].mean())
            features[f'{prefix}price_lag1'] = price_lag1_clean
            # Calculate log transformation on clean data
            features[f'{prefix}price_lag1_log'] = np.log1p(
                price_lag1_clean.fillna(0))
        if 'price_lag24' in df.columns:
            price_lag24_clean = df['price_lag24'].fillna(
                df['price_lag24'].mean())
            features[f'{prefix}price_lag24'] = price_lag24_clean
            features[f'{prefix}price_change_24h'] = features.get(
                f'{prefix}price_lag1', 0) - price_lag24_clean
        if 'price_ma7' in df.columns:
            price_ma7_clean = df['price_ma7'].fillna(df['price_ma7'].mean())
            features[f'{prefix}price_ma7'] = price_ma7_clean
            features[f'{prefix}price_deviation_ma7'] = features.get(
                f'{prefix}price_lag1', 0) - price_ma7_clean

        # Volatility features
        if 'price_volatility' in df.columns:
            features[f'{prefix}price_volatility'] = df['price_volatility']
            features[f'{prefix}price_volatility_log'] = np.log1p(
                df['price_volatility'])
        if 'weather_volatility' in df.columns:
            features[f'{prefix}weather_volatility'] = df['weather_volatility']

        # Enhanced extreme condition indicators
        if 'summer_extreme_solar' in df.columns:
            features[f'{prefix}summer_extreme_solar'] = df['summer_extreme_solar']
        if 'midday_solar_collapse' in df.columns:
            features[f'{prefix}midday_solar_collapse'] = df['midday_solar_collapse']
        if 'solar_oversupply' in df.columns:
            features[f'{prefix}solar_oversupply'] = df['solar_oversupply']

        # Advanced engineered features for price-awareness
        features[f'{prefix}demand_proxy'] = df['demand_proxy']
        features[f'{prefix}recent_weight'] = df['recent_weight']

        # Create rolling features for better temporal patterns
        if 'y' in df.columns:
            features[f'{prefix}price_rolling_std_6h'] = df['y'].rolling(
                6, min_periods=1).std().fillna(0)
            features[f'{prefix}price_rolling_min_12h'] = df['y'].rolling(
                12, min_periods=1).min().fillna(df['y'])
            features[f'{prefix}price_rolling_max_12h'] = df['y'].rolling(
                12, min_periods=1).max().fillna(df['y'])
        else:
            # For future predictions, use proxy values
            num_rows = len(df)
            features[f'{prefix}price_rolling_std_6h'] = pd.Series(
                [2.0] * num_rows, index=df.index)
            features[f'{prefix}price_rolling_min_12h'] = pd.Series(
                [8.0] * num_rows, index=df.index)
            features[f'{prefix}price_rolling_max_12h'] = pd.Series(
                [15.0] * num_rows, index=df.index)

        df_features = pd.DataFrame(features)
        return self._validate_ml_features(df_features)

    def _validate_ml_features(self, features_df):
        """Validate and clean ML features to ensure no NaN values"""
        # Fill any remaining NaN values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        nan_fixes = 0

        for col in numeric_cols:
            if features_df[col].isna().any():
                nan_count = features_df[col].isna().sum()
                # Use median for better stability with outliers
                fill_value = features_df[col].median()
                if pd.isna(fill_value):  # If median is also NaN
                    fill_value = 0
                features_df[col] = features_df[col].fillna(fill_value)
                nan_fixes += 1

                # Only show warnings for unexpected NaN values (not log transforms)
                if nan_count > 0 and not col.endswith('_log'):
                    print(
                        f"⚠️ Filled {nan_count} NaN values in {col} with {fill_value}")

        if nan_fixes > 0:
            print(f"✅ ML features validated: {nan_fixes} columns cleaned")

        return features_df

    def _train_xgboost_residual_model(self, df, prophet_predictions):
        """Train XGBoost model on Prophet residuals for enhanced price-awareness"""
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, skipping XGBoost training")
            return None

        try:
            # Calculate Prophet residuals
            residuals = df['y'].values - prophet_predictions
            ml_features = self._create_ml_features(df, prefix="")

            # Remove any NaN values
            valid_mask = ~(ml_features.isna().any(axis=1) | pd.isna(residuals))
            ml_features_clean = ml_features[valid_mask]
            residuals_clean = residuals[valid_mask]

            if len(ml_features_clean) < 50:
                print("⚠️ Insufficient data for XGBoost training")
                return None

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)

            # XGBoost parameters optimized for residual modeling
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'auto'
            }

            # Train model with cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(ml_features_clean):
                X_train, X_val = ml_features_clean.iloc[train_idx], ml_features_clean.iloc[val_idx]
                y_train, y_val = residuals_clean[train_idx], residuals_clean[val_idx]

                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_train, y_train)

                val_pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_pred)
                cv_scores.append(val_mae)

            # Train final model on all data
            final_model = xgb.XGBRegressor(**xgb_params)
            final_model.fit(ml_features_clean, residuals_clean)

            avg_cv_score = np.mean(cv_scores)
            print("🎯 XGBoost residual model trained")

            # Feature importance analysis
            feature_importance = dict(
                zip(ml_features_clean.columns, final_model.feature_importances_))

            return {
                'model': final_model,
                'feature_columns': ml_features_clean.columns.tolist(),
                'cv_score': avg_cv_score,
                'feature_importance': feature_importance
            }

        except Exception as e:
            print(f"⚠️ XGBoost training failed: {str(e)}")
            return None

    def _train_lightgbm_residual_model(self, df, prophet_predictions):
        """Alternative LightGBM model for residual modeling"""
        if not ML_AVAILABLE:
            return None

        try:
            # Calculate residuals and create features
            residuals = df['y'].values - prophet_predictions
            ml_features = self._create_ml_features(df, prefix="")

            # Clean data
            valid_mask = ~(ml_features.isna().any(axis=1) | pd.isna(residuals))
            ml_features_clean = ml_features[valid_mask]
            residuals_clean = residuals[valid_mask]

            if len(ml_features_clean) < 50:
                return None

            # LightGBM parameters
            lgb_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }

            # Time series split
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []

            for train_idx, val_idx in tscv.split(ml_features_clean):
                X_train, X_val = ml_features_clean.iloc[train_idx], ml_features_clean.iloc[val_idx]
                y_train, y_val = residuals_clean[train_idx], residuals_clean[val_idx]

                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(
                    X_val, label=y_val, reference=train_data)

                model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=200,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                val_pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_pred)
                cv_scores.append(val_mae)

            # Final model
            train_data = lgb.Dataset(ml_features_clean, label=residuals_clean)
            final_model = lgb.train(
                lgb_params, train_data, num_boost_round=200)

            avg_cv_score = np.mean(cv_scores)
            print("🚀 LightGBM residual model trained")

            return {
                'model': final_model,
                'feature_columns': ml_features_clean.columns.tolist(),
                'cv_score': avg_cv_score,
                'model_type': 'lightgbm'
            }

        except Exception as e:
            print(f"⚠️ LightGBM training failed: {str(e)}")
            return None

    def ensemble_forecast(self, hours_ahead: int = 48, use_xgboost: bool = True, use_lightgbm: bool = False) -> list:
        """Enhanced ensemble forecasting with adaptive strategy based on data availability"""

        # Apply performance monitoring if enhancements are enabled
        if self.enable_enhancements:
            return self._monitor_performance(self._enhanced_ensemble_forecast)(hours_ahead, use_xgboost, use_lightgbm)
        else:
            return self._original_ensemble_forecast(hours_ahead, use_xgboost, use_lightgbm)

    def _enhanced_ensemble_forecast(self, hours_ahead: int = 48, use_xgboost: bool = True, use_lightgbm: bool = False) -> list:
        """Enhanced ensemble forecasting with data validation and monitoring"""
        if self.enable_enhancements:
            self.logger.info(
                f"Starting enhanced ensemble forecast for {hours_ahead} hours")

        print("🚀 Starting Enhanced Ensemble Forecasting (Prophet + ML)")

        # Get prepared data
        df = self._prepare_data(self.repository.get_all_data())
        if df.empty:
            raise ValueError("No data available for forecasting")

        # Validate data quality if enhancements are enabled
        if self.enable_enhancements:
            is_valid, quality_report = self.validate_data_quality(df)
            if not is_valid and quality_report["overall_score"] < 0.5:
                self.logger.error(
                    f"Critical data quality issues: {quality_report['issues']}")
                raise ValueError(
                    f"Data quality too poor for predictions: score={quality_report['overall_score']}")

        return self._original_ensemble_forecast(hours_ahead, use_xgboost, use_lightgbm)

    def _original_ensemble_forecast(self, hours_ahead: int = 48, use_xgboost: bool = True, use_lightgbm: bool = False) -> list:
        """Original ensemble forecasting logic"""
        # Get prepared data
        df = self._prepare_data(self.repository.get_all_data())
        if df.empty:
            raise ValueError("No data available for forecasting")

        last_timestamp = df['ds'].max()

        # 📊 Data availability assessment
        data_days = (df['ds'].max() - df['ds'].min()).days
        unique_days = df['ds'].dt.date.nunique()
        offpeak_samples = len(df[df['is_offpeak'] == 1])

        print(
            f"📈 Data assessment: {data_days} days span, {unique_days} unique days, {offpeak_samples} off-peak samples")

        # 🎯 Adaptive strategy based on data availability
        if unique_days < 7 or offpeak_samples < 50:
            print(
                "⚠️ Limited data detected - using ML-first approach with enhanced features")
            return self._ml_first_forecast(df, hours_ahead, use_xgboost, use_lightgbm)
        else:
            print("✅ Sufficient data for Prophet + ML ensemble")
            return self._full_ensemble_forecast(df, hours_ahead, use_xgboost, use_lightgbm)

    def _ml_first_forecast(self, df, hours_ahead: int, use_xgboost: bool, use_lightgbm: bool) -> list:
        """ML-first forecasting for limited data scenarios"""
        if not ML_AVAILABLE:
            print("⚠️ ML not available, falling back to simple Prophet")
            return self.forecast(hours_ahead)

        print("🤖 Using ML-first approach with enhanced solar features")

        # Create enhanced ML features with stronger solar interactions
        ml_features = self._create_enhanced_ml_features(df)

        # Target variable
        y = df['y'].values

        # Remove any NaN values
        valid_mask = ~(ml_features.isna().any(axis=1) | pd.isna(y))
        X_clean = ml_features[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < 24:  # Need at least one day
            print("⚠️ Insufficient clean data, falling back to Prophet")
            return self.forecast(hours_ahead)

        print(
            f"🔍 Training ML model with {len(X_clean)} samples and {X_clean.shape[1]} features")

        # Train primary ML model
        if use_xgboost:
            model = self._train_primary_xgboost(X_clean, y_clean)
        elif use_lightgbm:
            model = self._train_primary_lightgbm(X_clean, y_clean)
        else:
            print("⚠️ No ML model specified, falling back to Prophet")
            return self.forecast(hours_ahead)

        if not model:
            print("⚠️ ML training failed, falling back to Prophet")
            return self.forecast(hours_ahead)

        # Generate future features
        future_features = self._create_future_ml_features(df, hours_ahead)

        # Make predictions
        if model.get('model_type') == 'lightgbm':
            predictions = model['model'].predict(future_features)
        else:  # XGBoost
            predictions = model['model'].predict(future_features)

        # Create result dataframe
        last_timestamp = df['ds'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=hours_ahead,
            freq='h'
        )

        results = pd.DataFrame({
            'timestamp': future_timestamps,
            'predicted_price_cents_kwh': np.clip(predictions, 0, None).round(3),
            'hour': future_timestamps.hour,
            'date': future_timestamps.date.astype(str)
        })

        # Add confidence and categorization
        results['confidence'] = self._calculate_ml_confidence(results, model)
        results['period_type'] = results['hour'].apply(self._categorize_period)
        results['lower_bound_cents_kwh'] = (
            results['predicted_price_cents_kwh'] * 0.8).round(3)
        results['upper_bound_cents_kwh'] = (
            results['predicted_price_cents_kwh'] * 1.2).round(3)

        # Add weather data (temperature)
        weather_info = self._get_weather_for_timestamps(
            future_timestamps.tolist())
        results['temperature_celsius'] = weather_info.get(
            'temperature', [None] * len(results))

        # Add price categorization
        price_service = PriceService()
        results['price_category'] = results['predicted_price_cents_kwh'].apply(
            price_service.categorize_price)
        results['timestamp'] = results['timestamp'].astype(str)

        print(f"✅ ML-first forecast complete: {len(results)} predictions")
        return results.to_dict(orient='records')

    def _create_enhanced_ml_features(self, df):
        """Create enhanced ML features with stronger solar interactions for limited data"""
        features = {}

        # Enhanced time features with solar interactions
        features['hour'] = df['hour']
        features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        features['is_offpeak'] = df['is_offpeak']
        features['is_peak'] = df['is_peak']
        features['is_weekend'] = df['is_weekend']

        # Solar features
        if 'solar_factor' in df.columns:
            features['solar_factor'] = df['solar_factor']
            # 🌞 CRITICAL: Strong solar-hour interactions
            features['solar_hour_interaction'] = df['solar_factor'] * df['hour']
            features['solar_offpeak_interaction'] = df['solar_factor'] * \
                df['is_offpeak']
            features['solar_squared'] = df['solar_factor'] ** 2
            features['solar_midday_effect'] = df['solar_factor'] * \
                (df['hour'].between(12, 14).astype(int))

            # Exponential solar effect during peak solar hours
            features['solar_exponential'] = np.exp(
                df['solar_factor'] * df['is_offpeak'] * -2)

        # Temperature interactions
        if 'temperature' in df.columns:
            features['temperature'] = df['temperature']
            features['temp_squared'] = df['temperature'] ** 2
            features['temp_hour_interaction'] = df['temperature'] * df['hour']

        # Cloud cover effects
        if 'cloud_cover' in df.columns:
            features['cloud_cover'] = df['cloud_cover']
            features['clear_sky_indicator'] = (
                df['cloud_cover'] < 20).astype(int)
            features['cloudy_indicator'] = (df['cloud_cover'] > 70).astype(int)

        # Enhanced demand proxy
        features['demand_proxy'] = df['demand_proxy']
        features['demand_solar_interaction'] = df['demand_proxy'] * \
            features.get('solar_factor', 0)

        # Price-based features if available
        if 'price_lag1' in df.columns:
            price_lag1_clean = df['price_lag1'].fillna(df['price_lag1'].mean())
            features['price_lag1'] = price_lag1_clean
            features['price_lag1_log'] = np.log1p(price_lag1_clean.fillna(0))
        if 'price_lag24' in df.columns:
            features['price_lag24'] = df['price_lag24'].fillna(
                df['price_lag24'].mean())

        # Volatility features
        if 'price_volatility' in df.columns:
            features['price_volatility'] = df['price_volatility']

        # Recent weighting
        features['recent_weight'] = df['recent_weight']

        # Day of week effects
        features['day_of_week'] = df['day_of_week']
        features['is_workday'] = df['is_workday']

        # Month seasonality
        features['month'] = df['month']
        features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df_features = pd.DataFrame(features)
        return self._validate_ml_features(df_features)

    def _train_primary_xgboost(self, X, y):
        """Train XGBoost as primary model for limited data scenarios"""
        try:
            # Enhanced parameters for limited data
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 300,  # More trees for better learning
                'max_depth': 8,       # Deeper trees for complex interactions
                'learning_rate': 0.05,  # Slower learning for stability
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,     # L1 regularization
                'reg_lambda': 0.1,    # L2 regularization
                'random_state': 42,
                'tree_method': 'auto'
            }

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X, y)

            # Feature importance analysis
            feature_importance = dict(
                zip(X.columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(),
                                  key=lambda x: x[1], reverse=True)[:5]
            print(
                f"🔝 Top features: {[f'{name}: {imp:.3f}' for name, imp in top_features]}")

            return {
                'model': model,
                'feature_columns': X.columns.tolist(),
                'feature_importance': feature_importance,
                'model_type': 'xgboost'
            }

        except Exception as e:
            print(f"⚠️ XGBoost training failed: {str(e)}")
            return None

    def _train_primary_lightgbm(self, X, y):
        """Train LightGBM as primary model for limited data scenarios"""
        try:
            # Enhanced parameters for limited data
            lgb_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 50,     # More leaves for complex patterns
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1,
                'random_state': 42
            }

            train_data = lgb.Dataset(X, label=y)
            model = lgb.train(lgb_params, train_data, num_boost_round=300)

            print("🚀 LightGBM primary model trained")

            return {
                'model': model,
                'feature_columns': X.columns.tolist(),
                'model_type': 'lightgbm'
            }

        except Exception as e:
            print(f"⚠️ LightGBM training failed: {str(e)}")
            return None

    def _create_future_ml_features(self, df, hours_ahead):
        """Create ML features for future predictions"""
        last_timestamp = df['ds'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=hours_ahead,
            freq='h'
        )

        # Create future dataframe with basic time features
        future_df = pd.DataFrame({
            'ds': future_timestamps,
            'hour': future_timestamps.hour,
            'day_of_week': future_timestamps.dayofweek,
            'month': future_timestamps.month,
            'is_weekend': (future_timestamps.dayofweek >= 5).astype(int),
            'is_peak': ((future_timestamps.hour.between(7, 9)) |
                        (future_timestamps.hour.between(19, 22))).astype(int),
            'is_offpeak': future_timestamps.hour.between(10, 15).astype(int),
            'is_workday': ((future_timestamps.dayofweek < 5) &
                           (future_timestamps.hour.between(6, 18))).astype(int),
        })

        # Add recent data weighting
        future_df['recent_weight'] = np.exp(
            -(future_df['ds'] - df['ds'].max()).dt.days / 14)

        # Add demand proxy
        future_df['demand_proxy'] = (np.sin(2 * np.pi * future_df['hour'] / 24) +
                                     0.5 * np.sin(2 * np.pi * future_df['day_of_week'] / 7))

        # Add weather features (proxy for future)
        future_df = self._add_weather_proxy(future_df)

        # Create enhanced ML features
        enhanced_features = self._create_enhanced_ml_features(future_df)

        # Fill price-based features with last known values
        if 'price_lag1' in df.columns:
            last_price = df['y'].iloc[-1]
            enhanced_features['price_lag1'] = last_price
            enhanced_features['price_lag1_log'] = np.log1p(last_price)
        if 'price_lag24' in df.columns:
            last_price_24h = df['y'].iloc[-24] if len(
                df) > 24 else df['y'].mean()
            enhanced_features['price_lag24'] = last_price_24h
        if 'price_volatility' in df.columns:
            last_volatility = df['price_volatility'].iloc[-1] if 'price_volatility' in df.columns else 5
            enhanced_features['price_volatility'] = last_volatility

        # Final validation to ensure no NaN values
        return self._validate_ml_features(enhanced_features)

    def _calculate_ml_confidence(self, results, model_info):
        """Calculate confidence scores for ML predictions"""
        # Base confidence (higher for shorter horizons)
        time_confidence = np.exp(-0.02 * np.arange(len(results)))

        # Off-peak confidence (ML models handle solar better)
        offpeak_confidence = np.where(
            results['hour'].between(10, 15), 0.95, 0.9)

        # Model-based confidence
        model_confidence = 0.85  # ML models generally more confident with interactions

        combined_confidence = (time_confidence * 0.4 +
                               offpeak_confidence * 0.3 +
                               model_confidence * 0.3)

        return combined_confidence.round(3)

    def _full_ensemble_forecast(self, df, hours_ahead: int, use_xgboost: bool, use_lightgbm: bool) -> list:
        """Full Prophet + ML ensemble for sufficient data scenarios"""
        last_timestamp = df['ds'].max()

        # Step 1: Train Prophet baseline model
        params = self.best_params or {
            'changepoint_prior_scale': 0.02,
            'seasonality_prior_scale': 2.0,
            'seasonality_mode': 'multiplicative'
        }

        prophet_model = Prophet(
            daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False, **params)

        # Add all the enhanced regressors
        self._add_prophet_regressors(prophet_model, df)

        # Fit Prophet
        prophet_model.fit(df)

        # Get Prophet predictions on training data
        prophet_train_forecast = prophet_model.predict(df)
        prophet_predictions = prophet_train_forecast['yhat'].values

        # Step 2: Train residual models
        residual_models = {}

        if use_xgboost and ML_AVAILABLE:
            xgb_model = self._train_xgboost_residual_model(
                df, prophet_predictions)
            if xgb_model:
                residual_models['xgboost'] = xgb_model

        if use_lightgbm and ML_AVAILABLE:
            lgb_model = self._train_lightgbm_residual_model(
                df, prophet_predictions)
            if lgb_model:
                residual_models['lightgbm'] = lgb_model

        if not residual_models:
            print("⚠️ No residual models trained, falling back to Prophet-only")
            return self.forecast(hours_ahead)

        # Step 3: Generate future predictions
        future = prophet_model.make_future_dataframe(
            periods=hours_ahead, freq='h')
        future = self._add_future_regressors(future, df, hours_ahead)

        # Prophet baseline forecast
        prophet_forecast = prophet_model.predict(future)

        # Get future period predictions
        training_length = len(df)
        future_prophet = prophet_forecast[[
            'ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(hours_ahead).copy()

        # Ensure we only get truly future predictions
        future_mask = future_prophet['ds'] > last_timestamp
        future_prophet = future_prophet[future_mask].copy()

        if future_prophet.empty:
            print("⚠️ No future predictions generated")
            return []

        # Create ML features for future predictions
        future_df = future.tail(len(future_prophet)).copy()
        future_ml_features = self._create_ml_features(future_df, prefix="")

        # Generate residual predictions
        residual_predictions = {}
        for model_name, model_info in residual_models.items():
            try:
                # Ensure feature alignment
                available_features = [
                    col for col in model_info['feature_columns'] if col in future_ml_features.columns]
                if len(available_features) < len(model_info['feature_columns']) * 0.8:
                    continue

                future_features_aligned = future_ml_features[available_features]

                if model_info.get('model_type') == 'lightgbm':
                    residual_pred = model_info['model'].predict(
                        future_features_aligned)
                else:  # XGBoost
                    residual_pred = model_info['model'].predict(
                        future_features_aligned)

                residual_predictions[model_name] = residual_pred

            except Exception as e:
                print(f"⚠️ {model_name} prediction failed: {str(e)}")

        # Step 4: Combine predictions
        if residual_predictions:
            # Ensemble residuals (simple average if multiple models)
            ensemble_residuals = np.mean(
                list(residual_predictions.values()), axis=0)

            # Final ensemble prediction
            future_prophet['yhat_ensemble'] = future_prophet['yhat'] + \
                ensemble_residuals

            # Adjust confidence intervals
            residual_std = np.std(ensemble_residuals)
            future_prophet['yhat_lower_ensemble'] = future_prophet['yhat_ensemble'] - \
                1.96 * residual_std
            future_prophet['yhat_upper_ensemble'] = future_prophet['yhat_ensemble'] + \
                1.96 * residual_std

            # Use ensemble predictions as primary
            future_prophet['yhat'] = future_prophet['yhat_ensemble']
            future_prophet['yhat_lower'] = future_prophet['yhat_lower_ensemble']
            future_prophet['yhat_upper'] = future_prophet['yhat_upper_ensemble']
        else:
            print("⚠️ No residual predictions available, using Prophet baseline")

        # Post-processing and formatting
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            future_prophet[col] = future_prophet[col].clip(lower=0).round(3)

        future_prophet['hour'] = future_prophet['ds'].dt.hour
        future_prophet['date'] = future_prophet['ds'].dt.date.astype(str)
        future_prophet['confidence'] = self._calculate_confidence(
            future_prophet, df)
        future_prophet['period_type'] = future_prophet['hour'].apply(
            self._categorize_period)

        # Get weather data for the corresponding future timestamps
        future_timestamps = future_prophet['ds'].tolist()
        weather_info = self._get_weather_for_timestamps(future_timestamps)

        # Add temperature to results
        future_prophet['temperature_celsius'] = weather_info.get(
            'temperature', [None] * len(future_prophet))

        # Select only user-friendly columns for final output
        user_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'hour', 'date',
                        'confidence', 'period_type', 'temperature_celsius']
        future_prophet = future_prophet[user_columns].copy()

        future_prophet = future_prophet.rename(columns={
            'ds': 'timestamp',
            'yhat': 'predicted_price_cents_kwh',
            'yhat_lower': 'lower_bound_cents_kwh',
            'yhat_upper': 'upper_bound_cents_kwh'
        })

        # Add price categorization
        price_service = PriceService()
        future_prophet['price_category'] = future_prophet['predicted_price_cents_kwh'].apply(
            price_service.categorize_price)
        future_prophet['timestamp'] = future_prophet['timestamp'].astype(str)

        print("✅ Enhanced ensemble forecast complete")
        return future_prophet.to_dict(orient='records')

    def _add_prophet_regressors(self, model, df):
        """Add all Prophet regressors (centralized method to avoid duplication)"""
        # Time regressors
        time_regressors = ['hour', 'is_weekend',
                           'is_peak', 'is_offpeak', 'month', 'is_workday']
        for reg in time_regressors:
            prior_scale = 50.0 if reg == 'is_offpeak' else 10.0
            model.add_regressor(reg, prior_scale=prior_scale)

        # Off-peak specific regressors
        offpeak_regressors = ['solar_peak_hours',
                              'is_midday_solar', 'summer_extreme_solar']
        for reg in offpeak_regressors:
            if reg in df.columns:
                prior_scale = 50.0 if reg == 'summer_extreme_solar' else 20.0
                model.add_regressor(reg, prior_scale=prior_scale)

        # Price-based regressors (minimal influence)
        for reg in ['price_lag1', 'price_lag24', 'price_ma7', 'demand_proxy']:
            if reg in df.columns:
                model.add_regressor(reg, prior_scale=0.1)

        if 'offpeak_ma3' in df.columns:
            model.add_regressor('offpeak_ma3', prior_scale=0.2)

        # Weather regressors
        weather_regressors = ['cloud_cover', 'temperature', 'solar_factor',
                              'solar_production_factor', 'temp_demand_factor', 'weather_price_impact']
        for reg in weather_regressors:
            if reg in df.columns:
                prior_scale = 8.0 if 'solar' in reg or 'impact' in reg else 3.0
                model.add_regressor(reg, prior_scale=prior_scale)

        # Critical solar collapse regressors
        if 'midday_solar_collapse' in df.columns:
            model.add_regressor('midday_solar_collapse', prior_scale=25.0)
        if 'solar_oversupply' in df.columns:
            model.add_regressor('solar_oversupply', prior_scale=15.0)

        # Off-peak weather regressors
        for reg in ['offpeak_solar_impact', 'offpeak_weather_vol']:
            if reg in df.columns:
                model.add_regressor(reg, prior_scale=20.0)

        # Volatility and other regressors
        if 'price_volatility' in df.columns:
            model.add_regressor('price_volatility', prior_scale=0.1)
        if 'weather_volatility' in df.columns:
            model.add_regressor('weather_volatility', prior_scale=0.2)
        if 'recent_weight' in df.columns:
            model.add_regressor('recent_weight', prior_scale=0.8)

        # Enhanced seasonalities
        model.add_seasonality(name='hourly', period=24, fourier_order=12)
        model.add_seasonality(name='daily_weather', period=24, fourier_order=5)

        if 'solar_seasonality_condition' in df.columns:
            model.add_seasonality(name='offpeak_solar', period=24,
                                  fourier_order=6, condition_name='solar_seasonality_condition')

        model.add_seasonality(name='midday_sharp', period=24,
                              fourier_order=8, prior_scale=15.0)

    @cache_forecast(ttl_seconds=3600)
    def forecast(self, hours_ahead: int = 48) -> list:
        """Generate weather-enhanced forecasts starting from the last available data point"""
        # Get data with weather integration
        df = self._prepare_data(self.repository.get_all_data())

        if df.empty:
            raise ValueError("No data available for forecasting")

        last_timestamp = df['ds'].max()
        data_summary = self.repository.get_data_summary()

        # Use tuned params or enhanced defaults
        params = self.best_params or {
            'changepoint_prior_scale': 0.02,
            'seasonality_prior_scale': 2.0,
            'seasonality_mode': 'multiplicative'
        }

        model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                        yearly_seasonality=False, **params)

        # Add all regressors
        self._add_prophet_regressors(model, df)
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

        # Enhanced post-processing
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            results[col] = results[col].clip(lower=0).round(3)

        # Add hour column for confidence calculation and categorization
        results['hour'] = results['ds'].dt.hour
        results['date'] = results['ds'].dt.date.astype(str)
        results['confidence'] = self._calculate_confidence(results, df)
        results['period_type'] = results['hour'].apply(self._categorize_period)

        # Add weather data (temperature)
        weather_info = self._get_weather_for_timestamps(results['ds'].tolist())
        results['temperature_celsius'] = weather_info.get(
            'temperature', [None] * len(results))

        results = results.rename(columns={
            'ds': 'timestamp', 'yhat': 'predicted_price_cents_kwh',
            'yhat_lower': 'lower_bound_cents_kwh', 'yhat_upper': 'upper_bound_cents_kwh'
        })

        # Create PriceService instance for categorization
        price_service = PriceService()
        results['price_category'] = results['predicted_price_cents_kwh'].apply(
            price_service.categorize_price)
        results['timestamp'] = results['timestamp'].astype(str)

        return results.to_dict(orient='records')

    def _add_future_regressors(self, future, df, hours_ahead):
        """Add regressors for future predictions with off-peak weather optimization"""
        # Basic time regressors
        future['hour'] = future['ds'].dt.hour
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        future['is_peak'] = ((future['ds'].dt.hour.between(7, 9)) | (
            future['ds'].dt.hour.between(19, 22))).astype(int)
        future['is_offpeak'] = future['ds'].dt.hour.between(10, 15).astype(int)
        future['month'] = future['ds'].dt.month
        future['day_of_week'] = future['ds'].dt.dayofweek
        future['is_workday'] = ((future['ds'].dt.dayofweek < 5) & (
            future['hour'].between(6, 18))).astype(int)

        # Off-peak specific features
        future['solar_peak_hours'] = (
            (future['hour'] >= 11) & (future['hour'] <= 14)).astype(int)
        future['is_midday_solar'] = (
            (future['hour'] >= 12) & (future['hour'] <= 13)).astype(int)
        future['solar_seasonality_condition'] = future['solar_peak_hours'].copy()

        # Add recent data weighting
        future['data_age_days'] = (future['ds'].max() - future['ds']).dt.days
        future['recent_weight'] = np.exp(-future['data_age_days'] / 14)

        training_length = len(df)

        # Fill historical regressors
        for col in ['price_lag1', 'price_lag24', 'price_ma7', 'price_volatility', 'offpeak_ma3']:
            if col in df.columns:
                future.loc[:training_length-1, col] = df[col].values

        # Estimate future price-based regressors
        last_price = df['y'].iloc[-1] if len(df) > 0 else 50
        last_price_24h = df['y'].iloc[-24] if len(df) > 24 else df['y'].mean()
        last_ma7 = df['price_ma7'].iloc[-1] if 'price_ma7' in df.columns else last_price
        last_volatility = df['price_volatility'].iloc[-1] if 'price_volatility' in df.columns else 5
        recent_offpeak = df[df['is_offpeak'] == 1]['y'].tail(
            7).mean() if len(df[df['is_offpeak'] == 1]) > 0 else last_price

        future.loc[training_length:, 'price_lag1'] = last_price
        future.loc[training_length:, 'price_lag24'] = last_price_24h
        future.loc[training_length:, 'price_ma7'] = last_ma7
        future.loc[training_length:, 'price_volatility'] = last_volatility
        future.loc[training_length:, 'offpeak_ma3'] = recent_offpeak

        future['demand_proxy'] = (np.sin(
            2 * np.pi * future['hour'] / 24) + 0.5 * np.sin(2 * np.pi * future['day_of_week'] / 7))

        # Get weather forecast for future periods
        future_timestamps = future.loc[training_length:, 'ds'].tolist()
        if future_timestamps:
            try:
                weather_forecast = self.weather_collector.get_forecast(
                    hours=len(future_timestamps) * 3)
                if not weather_forecast.empty:
                    weather_forecast['timestamp'] = pd.to_datetime(
                        weather_forecast['timestamp']).dt.tz_localize(None)
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
                    future_proxy = self._add_weather_proxy(
                        future.loc[training_length:])
                    for col in ['cloud_cover', 'temperature', 'solar_factor']:
                        if col in future_proxy.columns:
                            future.loc[training_length:,
                                       col] = future_proxy[col].values
            except Exception as e:
                print(f"⚠️ Weather forecast failed, using proxy: {e}")
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

        # Add summer solar feature AFTER weather features are calculated
        future['summer_extreme_solar'] = ((future['month'].between(6, 8)) & (
            future['hour'].between(11, 15)) & (future['solar_factor'] > 0.7)).astype(int)

        # Add sharp solar collapse features for future predictions
        future['midday_solar_collapse'] = np.where((future['hour'].between(13, 15)) & (future['solar_factor'] > 0.6),
                                                   future['solar_factor'] * -25.0, 0).round(2)

        future['solar_oversupply'] = np.where(
            (future['solar_factor'] > 0.8) & (future['hour'].between(12, 15)), 1, 0)

        return future

    def _get_weather_for_timestamps(self, timestamps):
        """Get weather data (specifically temperature) for given timestamps"""
        try:
            # Try to get actual weather forecast
            weather_df = self.weather_collector.get_forecast(
                hours=len(timestamps) * 2)

            if not weather_df.empty:
                weather_df['timestamp'] = pd.to_datetime(
                    weather_df['timestamp']).dt.tz_localize(None)

                # Create DataFrame for timestamps to merge
                ts_df = pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})

                # Merge weather data
                merged = pd.merge_asof(
                    ts_df.sort_values('timestamp'),
                    weather_df.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'
                )

                return {
                    'temperature': merged['temperature'].round(1).tolist() if 'temperature' in merged.columns else None
                }
            else:
                # Use weather proxy if no actual data available
                proxy_temps = self._generate_temperature_proxy(timestamps)
                return {'temperature': proxy_temps}

        except Exception as e:
            print(f"⚠️ Weather data retrieval failed: {e}")
            # Generate proxy temperatures as fallback
            proxy_temps = self._generate_temperature_proxy(timestamps)
            return {'temperature': proxy_temps}

    def _generate_temperature_proxy(self, timestamps):
        """Generate realistic temperature proxy for given timestamps"""
        import numpy as np

        temps = []
        for ts in timestamps:
            ts_dt = pd.to_datetime(ts)
            day_of_year = ts_dt.dayofyear
            hour = ts_dt.hour

            # Base seasonal temperature (15°C average, +/-10°C seasonal variation)
            seasonal_temp = 15 + 10 * \
                np.sin(2 * np.pi * (day_of_year - 80) / 365)

            # Daily variation (+/-5°C)
            daily_variation = 5 * np.sin(2 * np.pi * hour / 24)

            # Small random variation
            random_variation = np.random.normal(0, 1)

            temperature = round(
                seasonal_temp + daily_variation + random_variation, 1)
            temps.append(temperature)

        return temps

    def _calculate_confidence(self, results, df):
        """Calculate enhanced confidence scores with off-peak considerations"""
        # Base confidence from prediction intervals
        denom = results['yhat'].replace(0, np.nan)
        interval_confidence = np.clip(
            1 - (results['yhat_upper'] - results['yhat_lower']) / denom, 0, 1).fillna(0)

        # Weather-based confidence adjustment
        weather_confidence = 1.0
        if 'weather_volatility' in df.columns:
            recent_weather_vol = df['weather_volatility'].tail(24).mean()
            weather_confidence = np.clip(1 - recent_weather_vol / 50, 0.5, 1.0)

        # Time-based confidence (higher confidence for near-term predictions)
        time_confidence = np.exp(-0.05 * np.arange(len(results)))

        # Off-peak confidence adjustment
        offpeak_confidence = np.where(
            results['hour'].between(10, 15), 0.9, 1.0)

        # Combined confidence with off-peak considerations
        combined_confidence = (interval_confidence * 0.4 + weather_confidence *
                               0.25 + time_confidence * 0.2 + offpeak_confidence * 0.15)

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

    def get_offpeak_accuracy_analysis(self, days_back: int = 7):
        """Simple off-peak analysis for recent period"""
        try:
            df = self._prepare_data(self.repository.get_all_data())
            if df.empty:
                return {"error": "No data available"}

            # Filter to recent period
            cutoff_date = df['ds'].max() - pd.Timedelta(days=days_back)
            recent_df = df[df['ds'] >= cutoff_date].copy()

            if recent_df.empty:
                return {"error": f"No data available for last {days_back} days"}

            # Focus on off-peak period (10-15h)
            offpeak_data = recent_df[recent_df['is_offpeak'] == 1]

            if offpeak_data.empty:
                return {"error": "No off-peak data found"}

            # Basic statistics
            offpeak_mean = offpeak_data['y'].mean()
            offpeak_std = offpeak_data['y'].std()
            midday_data = recent_df[recent_df['hour'].between(12, 14)]
            midday_mean = midday_data['y'].mean(
            ) if not midday_data.empty else offpeak_mean
            offpeak_volatility = offpeak_data['y'].rolling(
                window=6, min_periods=1).std().mean()

            # Solar correlation (if available)
            solar_correlation = 0
            if 'solar_factor' in offpeak_data.columns:
                solar_correlation = offpeak_data['solar_factor'].corr(
                    offpeak_data['y'])
                if pd.isna(solar_correlation):
                    solar_correlation = 0

            # Count extreme low prices
            extreme_low_count = len(offpeak_data[offpeak_data['y'] < 3.0])

            return {
                'analysis_period': f"Last {days_back} days",
                'offpeak_hours_analyzed': len(offpeak_data),
                'avg_offpeak_price': round(offpeak_mean, 2),
                'avg_midday_price': round(midday_mean, 2),
                'offpeak_price_std': round(offpeak_std, 2),
                'offpeak_volatility': round(offpeak_volatility, 2),
                'extreme_low_prices_count': extreme_low_count,
                'extreme_low_percentage': round(extreme_low_count / len(offpeak_data) * 100, 1),
                'solar_price_correlation': round(solar_correlation, 3),
                'recommendation': self._get_offpeak_recommendation(offpeak_volatility, solar_correlation, extreme_low_count)
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _get_offpeak_recommendation(self, volatility, solar_corr, extreme_low_count=0):
        """Simple recommendations based on off-peak analysis"""
        recommendations = []

        if volatility > 8:
            recommendations.append(
                "High off-peak volatility detected - consider shorter forecast horizons")
        if abs(solar_corr) > 0.3:
            recommendations.append(
                f"Strong solar correlation ({solar_corr:.2f}) - weather forecasts critical")
        elif abs(solar_corr) < 0.1:
            recommendations.append(
                "Weak solar correlation - check weather proxy accuracy")
        if extreme_low_count > 0:
            recommendations.append(
                f"Detected {extreme_low_count} extreme low prices - enhanced solar model active")
        if not recommendations:
            recommendations.append(
                "Off-peak patterns stable - current model performing well")

        return recommendations

    # Enhanced convenience methods
    def enhanced_ensemble_forecast(self, hours_ahead: int = None, use_xgboost: bool = None, use_lightgbm: bool = None) -> list:
        """Convenience method for enhanced ensemble forecasting"""
        # Use configuration defaults if available
        if self.enable_enhancements and hasattr(self, 'config'):
            hours_ahead = hours_ahead or self.config.get(
                "model_settings", {}).get("default_forecast_hours", 48)
            use_xgboost = use_xgboost if use_xgboost is not None else self.config.get(
                "model_settings", {}).get("enable_ml_models", True)
            use_lightgbm = use_lightgbm if use_lightgbm is not None else self.config.get(
                "model_settings", {}).get("enable_ml_models", True)
        else:
            hours_ahead = hours_ahead or 48
            use_xgboost = use_xgboost if use_xgboost is not None else True
            # Enable LightGBM by default too
            use_lightgbm = use_lightgbm if use_lightgbm is not None else True

        return self.ensemble_forecast(hours_ahead, use_xgboost, use_lightgbm)

    def update_configuration(self, section: str, key: str, value: Any):
        """Update configuration value"""
        if not self.enable_enhancements:
            raise ValueError("Enhancements not enabled")

        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            if hasattr(self, 'logger'):
                self.logger.info(
                    f"Configuration updated: {section}.{key} = {value}")
        else:
            raise ValueError(f"Invalid configuration path: {section}.{key}")

# Factory function for easy enhanced service creation


def create_enhanced_prophet_service(repository=None, enable_all_features=True):
    """Create an enhanced Prophet service with all improvements"""
    return ProphetForecastService(repository, enable_enhancements=enable_all_features)
