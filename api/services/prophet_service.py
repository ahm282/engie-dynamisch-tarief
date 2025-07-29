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

        # CRITICAL: Add solar suppression indicator for cloud cover < 30%
        df['solar_suppression_active'] = ((df['cloud_cover'] < 30) &
                                          (df['hour'].between(11, 16))).astype(int)

        return self._enhance_weather_features(df)

    def _enhance_weather_features(self, df):
        """Create enhanced weather-based features with aggressive off-peak solar optimizations"""
        if 'hour' not in df.columns:
            df['hour'] = df['ds'].dt.hour
        if 'is_offpeak' not in df.columns:
            df['is_offpeak'] = df['ds'].dt.hour.between(10, 15).astype(int)

        # Solar production impact (enhanced)
        solar_production = df['solar_factor'] * \
            np.maximum(0, np.sin(2 * np.pi * (df['hour'] - 6) / 12) ** 2)
        df['solar_production_factor'] = solar_production.round(2)

        # Phase 1.1: STRONGER SOLAR INTERACTION TERMS

        # Non-linear solar interactions for extreme price depression
        df['solar_factor_squared_offpeak'] = np.where(
            df['is_offpeak'] == 1, df['solar_factor'] ** 2, 0).round(3)

        df['solar_factor_cubed_offpeak'] = np.where(
            df['is_offpeak'] == 1, df['solar_factor'] ** 3, 0).round(3)

        # Solar-hour interaction with exponential emphasis on midday
        df['solar_hour_squared_offpeak'] = np.where(
            df['is_offpeak'] == 1,
            df['solar_factor'] * (df['hour'] - 12) ** 2 * -0.5, 0).round(3)

        # Exponential solar impact during peak solar + off-peak hours
        df['solar_exponential_offpeak'] = np.where(
            df['is_offpeak'] == 1,
            np.exp(df['solar_factor'] * 3) * -2.0, 0).round(3)

        # Solar ramp indicators for sudden price shifts
        if len(df) > 1:
            df['solar_factor_change_1h'] = df['solar_factor'].diff().fillna(0).round(3)
            df['solar_rapid_increase'] = np.where(
                (df['solar_factor_change_1h'] > 0.2) & (df['is_offpeak'] == 1),
                df['solar_factor_change_1h'] * -10.0, 0).round(3)
        else:
            df['solar_factor_change_1h'] = 0
            df['solar_rapid_increase'] = 0

        # Enhanced off-peak solar impact (much more aggressive)
        df['offpeak_solar_impact'] = np.where(
            df['is_offpeak'] == 1, df['solar_production_factor'] * -35.0, 0).round(2)

        # Multiple levels of solar oversupply
        df['solar_oversupply_high'] = np.where(
            (df['solar_factor'] > 0.85) & (df['hour'].between(11, 15)), 1, 0)

        df['solar_oversupply_extreme'] = np.where(
            (df['solar_factor'] > 0.9) & (df['hour'].between(12, 14)), 1, 0)

        # Enhanced midday solar collapse with non-linear scaling
        df['midday_solar_collapse'] = np.where(
            (df['hour'].between(13, 15)) & (df['solar_factor'] > 0.6),
            df['solar_factor'] ** 1.5 * -40.0, 0).round(2)

        # Extreme solar collapse for very high solar during optimal hours
        df['extreme_solar_collapse'] = np.where(
            (df['hour'].between(12, 14)) & (df['solar_factor'] > 0.85),
            df['solar_factor'] ** 2 * -60.0, 0).round(2)

        # Standard solar oversupply indicator
        df['solar_oversupply'] = np.where(
            (df['solar_factor'] > 0.8) & (df['hour'].between(12, 15)), 1, 0)

        # CRITICAL: Solar Suppression Effect (cloud cover < 30% during hours 11-16)
        # This is the core issue - when solar floods the grid, prices crash to 20-40% of baseline
        if 'solar_suppression_active' in df.columns:
            # Extreme price suppression during high solar production
            df['solar_flood_effect'] = np.where(
                df['solar_suppression_active'] == 1,
                # Much more aggressive than current -60
                (df['solar_factor'] ** 3) * -80.0,
                0).round(2)

            # Additional targeted suppression for the 11-16 hour window
            df['midday_flood_multiplier'] = np.where(
                (df['hour'].between(11, 16)) & (df['cloud_cover'] < 30),
                df['solar_factor'] * -120.0,  # Extreme suppression
                0).round(2)

            # Cloud-specific suppression (lower cloud cover = more price crash)
            df['clear_sky_suppression'] = np.where(
                (df['hour'].between(11, 16)) & (df['cloud_cover'] < 20),
                (1 - df['cloud_cover']/100) * df['solar_factor'] * -150.0,
                0).round(2)
        else:
            # Fallback if solar_suppression_active not available
            df['solar_flood_effect'] = 0
            df['midday_flood_multiplier'] = 0
            df['clear_sky_suppression'] = 0

        # Phase 1.2: ENHANCED TEMPERATURE-SOLAR INTERACTIONS

        # Temperature-based demand
        temp_demand = np.where(df['temperature'] < 15, (15 - df['temperature']) / 10,
                               np.where(df['temperature'] > 25, (df['temperature'] - 25) / 10, 0))
        df['temp_demand_factor'] = temp_demand.round(2)

        # CRITICAL: Evening Demand Spikes (hot evenings + no solar = price spikes)
        df['is_evening_peak'] = ((df['hour'].between(19, 22))).astype(int)
        df['is_hot_evening'] = ((df['temperature'] > 25)
                                & df['is_evening_peak']).astype(int)

        # Evening spike when high demand meets no solar production (50-100% increase)
        # Base cooling demand effect
        cooling_demand = np.where(df['temperature'] > 25,
                                  ((df['temperature'] - 25) / 5) ** 1.5, 0)  # Non-linear scaling

        df['evening_demand_spike'] = np.where(
            df['is_hot_evening'] == 1,
            cooling_demand * 80.0,  # Increased from 40 to 80 for 50-100% effect
            0).round(2)

        # Extreme heat effect (temp > 30°C) during evening peak
        df['extreme_heat_evening'] = np.where(
            (df['temperature'] > 30) & df['is_evening_peak'],
            ((df['temperature'] - 30) / 2) * 120.0,  # Very aggressive scaling
            0).round(2)

        # No solar available during evening peak hours - amplifies price
        df['no_solar_evening_effect'] = np.where(
            df['is_evening_peak'] == 1,
            df['temp_demand_factor'] * 50.0,  # Increased from 25 to 50
            0).round(2)

        # Compound effect: hot evening + high baseline demand
        df['compound_evening_demand'] = np.where(
            (df['is_hot_evening'] == 1) & (df['temp_demand_factor'] > 0.5),
            df['temp_demand_factor'] * df['evening_demand_spike'] * 0.3,
            0).round(2)

        # Solar-temperature mild weather interaction (high solar + mild temp = very low prices)
        mild_temp_indicator = ((df['temperature'] >= 15) & (
            df['temperature'] <= 25)).astype(int)
        df['solar_temp_mild_interaction'] = np.where(
            df['is_offpeak'] == 1,
            df['solar_factor'] * mild_temp_indicator * -15.0, 0).round(2)

        # Phase 1.3: PRICE-TARGETED FEATURES

        # Off-peak specific price statistics (rolling minimums)
        if 'y' in df.columns:
            offpeak_mask = df['is_offpeak'] == 1
            df['offpeak_min_6h'] = df['y'].where(offpeak_mask).rolling(
                window=6, min_periods=1).min().fillna(df['y']).round(2)
            df['offpeak_median_24h'] = df['y'].where(offpeak_mask).rolling(
                window=24, min_periods=1).median().fillna(df['y']).round(2)
        else:
            # For future predictions, use proxy values
            df['offpeak_min_6h'] = 2.0  # Very low expected minimum
            df['offpeak_median_24h'] = 8.0  # Expected off-peak median

        # Weather volatility
        weather_vol = df['cloud_cover'].rolling(
            window=6, min_periods=1).std().fillna(0)
        df['weather_volatility'] = weather_vol.round(2)
        df['offpeak_weather_vol'] = np.where(
            df['is_offpeak'] == 1, weather_vol * 1.5, weather_vol).round(2)

        # Enhanced combined weather impact with aggressive solar terms
        base_impact = (df['temp_demand_factor'] * 0.3 +
                       (1 - df['solar_production_factor']) * 1.0 +
                       df['weather_volatility'] / 100 * 0.05)

        # Much more aggressive off-peak solar impact
        aggressive_solar_impact = (df['offpeak_solar_impact'] * 6.0 +
                                   df['solar_exponential_offpeak'] * 2.0 +
                                   df['solar_temp_mild_interaction'] * 1.5)

        df['weather_price_impact'] = np.where(
            df['is_offpeak'] == 1,
            base_impact + aggressive_solar_impact,
            base_impact).round(2)

        # CRITICAL: Weather Pattern Matching for Historical Baselines
        df = self._add_weather_pattern_baselines(df)

        return df

    def _add_weather_pattern_baselines(self, df):
        """Add weather-conditional baselines based on historical pattern matching"""
        if 'y' not in df.columns or len(df) < 168:  # Need at least 1 week of data
            # Set default baselines for future predictions
            df['weather_baseline_adjustment'] = 0
            df['similar_day_baseline'] = df.get(
                'y', pd.Series([10.0] * len(df)))
            return df

        weather_baseline_adjustments = []
        similar_day_baselines = []

        for idx, row in df.iterrows():
            # Find similar weather days in historical data (at least 7 days ago to avoid overfitting)
            historical_mask = (
                # At least 1 week ago
                (df['ds'] < row['ds'] - pd.Timedelta(days=7)) &
                # ±15% cloud cover
                (abs(df['cloud_cover'] - row['cloud_cover']) <= 15) &
                # ±3°C temperature
                (abs(df['temperature'] - row['temperature']) <= 3) &
                (df['hour'] == row['hour'])  # Same hour of day
            )

            similar_days = df[historical_mask]

            if len(similar_days) >= 3:  # Need at least 3 similar days
                # Calculate weather-adjusted baseline from similar days
                similar_prices = similar_days['y'].dropna()
                if len(similar_prices) > 0:
                    weather_baseline = similar_prices.median()

                    # Seasonal adjustment (account for month differences)
                    month_diff = abs(
                        row['month'] - similar_days['month'].median())
                    # Max 10% seasonal adjustment
                    seasonal_factor = 1.0 - (month_diff / 12) * 0.1

                    adjusted_baseline = weather_baseline * seasonal_factor

                    # Calculate adjustment relative to overall hour baseline
                    hour_baseline = df[df['hour'] == row['hour']]['y'].median()
                    baseline_adjustment = adjusted_baseline - hour_baseline

                    weather_baseline_adjustments.append(baseline_adjustment)
                    similar_day_baselines.append(adjusted_baseline)
                else:
                    weather_baseline_adjustments.append(0)
                    similar_day_baselines.append(row.get('y', 10.0))
            else:
                # Not enough similar days, use default
                weather_baseline_adjustments.append(0)
                similar_day_baselines.append(row.get('y', 10.0))

        df['weather_baseline_adjustment'] = weather_baseline_adjustments
        df['similar_day_baseline'] = similar_day_baselines

        return df

    def tune_hyperparameters(self):
        """Enhanced hyperparameter tuning optimized for extreme solar price scenarios"""
        df = self._prepare_data(self.repository.get_all_data())

        # Enhanced parameter grid with focus on extreme solar scenarios
        param_grid = [
            # Original parameters for baseline
            {'changepoint_prior_scale': 0.001,
                'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1,
                'seasonality_mode': 'additive'},

            # Enhanced parameters for solar extreme scenarios
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.9},
            {'changepoint_prior_scale': 0.02, 'seasonality_prior_scale': 2.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.8},

            # NEW: Additive mode with high seasonality for strong solar regressor effects
            {'changepoint_prior_scale': 0.08, 'seasonality_prior_scale': 8.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.85},
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 12.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.9},

            # NEW: Very flexible models for capturing extreme solar effects
            {'changepoint_prior_scale': 0.15, 'seasonality_prior_scale': 15.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.95},
            {'changepoint_prior_scale': 0.2, 'seasonality_prior_scale': 20.0,
                'seasonality_mode': 'additive', 'changepoint_range': 0.9},

            # Conservative multiplicative options (in case additive regressors work better with multiplicative base)
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0,
                'seasonality_mode': 'multiplicative'},
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 3.0,
                'seasonality_mode': 'multiplicative', 'changepoint_range': 0.85},
        ]

        best_mape = float('inf')
        best_offpeak_mape = float('inf')  # Track off-peak specific performance

        for params in param_grid:
            try:
                model = Prophet(**params)
                self._add_prophet_regressors(model, df)
                model.fit(df)

                # Cross-validation with focus on recent data (more relevant for solar patterns)
                cv_df = cross_validation(
                    model, horizon='48 hours', period='24 hours',
                    initial='168 hours', parallel="processes")
                performance = performance_metrics(cv_df)
                mape = performance['mape'].mean()

                # Calculate off-peak specific MAPE for better solar model selection
                cv_predictions = model.predict(cv_df)
                cv_with_pred = cv_df.merge(
                    cv_predictions[['ds', 'yhat']], on='ds', how='left')
                cv_with_pred['hour'] = cv_with_pred['ds'].dt.hour
                offpeak_data = cv_with_pred[cv_with_pred['hour'].between(
                    10, 15)]

                if len(offpeak_data) > 0:
                    offpeak_mape = np.mean(
                        np.abs((offpeak_data['y'] - offpeak_data['yhat']) / offpeak_data['y'])) * 100
                else:
                    offpeak_mape = mape

                # Weighted scoring: 60% overall MAPE + 40% off-peak MAPE
                combined_score = 0.6 * mape + 0.4 * offpeak_mape

                if combined_score < best_mape:
                    best_mape = combined_score
                    best_offpeak_mape = offpeak_mape
                    self.best_params = params
                    print(
                        f"🎯 New best params: MAPE={mape:.2f}, Off-peak MAPE={offpeak_mape:.2f}, Combined={combined_score:.2f}")

            except Exception as e:
                print(f"⚠️ Parameter set failed: {e}")
                continue

        # Save best parameters with enhanced metadata
        if self.best_params:
            PARAMS_FILE = Path(
                "../utils/prophet_models/prophet_best_params.json")
            PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)

            enhanced_params = {
                'parameters': self.best_params,
                'performance': {
                    'combined_score': best_mape,
                    'offpeak_mape': best_offpeak_mape,
                    'tuning_date': pd.Timestamp.now().isoformat(),
                    'optimization_focus': 'extreme_solar_scenarios'
                }
            }

            with open(PARAMS_FILE, 'w') as f:
                json.dump(enhanced_params, f, indent=2)

            print(
                f"✅ Best parameters saved with combined score: {best_mape:.2f}")
        else:
            print("⚠️ No valid parameters found, using defaults")
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

        # CRITICAL: New solar suppression features for cloud cover < 30% effect
        if 'solar_flood_effect' in df.columns:
            features[f'{prefix}solar_flood_effect'] = df['solar_flood_effect']
        if 'midday_flood_multiplier' in df.columns:
            features[f'{prefix}midday_flood_multiplier'] = df['midday_flood_multiplier']
        if 'clear_sky_suppression' in df.columns:
            features[f'{prefix}clear_sky_suppression'] = df['clear_sky_suppression']
        if 'solar_suppression_active' in df.columns:
            features[f'{prefix}solar_suppression_active'] = df['solar_suppression_active']

        # CRITICAL: New evening demand spike features
        if 'evening_demand_spike' in df.columns:
            features[f'{prefix}evening_demand_spike'] = df['evening_demand_spike']
        if 'no_solar_evening_effect' in df.columns:
            features[f'{prefix}no_solar_evening_effect'] = df['no_solar_evening_effect']
        if 'is_hot_evening' in df.columns:
            features[f'{prefix}is_hot_evening'] = df['is_hot_evening']
        if 'is_evening_peak' in df.columns:
            features[f'{prefix}is_evening_peak'] = df['is_evening_peak']
        if 'extreme_heat_evening' in df.columns:
            features[f'{prefix}extreme_heat_evening'] = df['extreme_heat_evening']
        if 'compound_evening_demand' in df.columns:
            features[f'{prefix}compound_evening_demand'] = df['compound_evening_demand']

        # NEW: Weather pattern matching features
        if 'weather_baseline_adjustment' in df.columns:
            features[f'{prefix}weather_baseline_adjustment'] = df['weather_baseline_adjustment']
        if 'similar_day_baseline' in df.columns:
            features[f'{prefix}similar_day_baseline'] = df['similar_day_baseline']

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
        """Train XGBoost model on Prophet residuals with focus on solar overestimation correction"""
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

            # Phase 2.2: Weight samples where Prophet overestimates (positive residuals)
            sample_weights = np.ones(len(residuals_clean))

            # Heavy weight for cases where Prophet significantly overestimates
            # Actual price much lower than predicted
            overestimate_mask = residuals_clean < -5.0
            sample_weights[overestimate_mask] *= 6.0

            # Moderate weight for moderate overestimation
            moderate_overest_mask = (
                residuals_clean >= -5.0) & (residuals_clean < -2.0)
            sample_weights[moderate_overest_mask] *= 3.0

            # Extra weight for off-peak overestimation (solar scenarios)
            if 'is_offpeak' in ml_features_clean.columns:
                offpeak_overest_mask = (ml_features_clean['is_offpeak'] == 1) & (
                    residuals_clean < -3.0)
                sample_weights[offpeak_overest_mask] *= 4.0

            # CRITICAL: Extra weights for solar suppression scenarios (cloud < 30%, hours 11-16)
            if 'solar_suppression_active' in ml_features_clean.columns:
                solar_suppression_mask = (ml_features_clean['solar_suppression_active'] == 1) & (
                    residuals_clean < -5.0)  # Prophet overestimated during solar flooding
                # Very high weight
                sample_weights[solar_suppression_mask] *= 8.0
                print(
                    f"🌞 Solar suppression cases weighted 8x: {np.sum(solar_suppression_mask)}")

            # CRITICAL: Extra weights for evening demand spike underestimation
            if 'is_evening_peak' in ml_features_clean.columns:
                evening_underest_mask = (ml_features_clean['is_evening_peak'] == 1) & (
                    residuals_clean > 3.0)  # Prophet underestimated evening spikes
                # High weight for spikes
                sample_weights[evening_underest_mask] *= 5.0
                print(
                    f"🌆 Evening spike cases weighted 5x: {np.sum(evening_underest_mask)}")

            # CRITICAL: Extra weights for extreme heat evening underestimation
            if 'extreme_heat_evening' in ml_features_clean.columns:
                extreme_heat_mask = (ml_features_clean['extreme_heat_evening'] > 0) & (
                    residuals_clean > 5.0)  # Severe underestimation during extreme heat
                sample_weights[extreme_heat_mask] *= 7.0  # Very high weight
                print(
                    f"🔥 Extreme heat evening cases weighted 7x: {np.sum(extreme_heat_mask)}")

            # Extra weight for clear sky (cloud < 20%) overestimation during midday
            if 'clear_sky_suppression' in ml_features_clean.columns:
                clear_sky_mask = (ml_features_clean['clear_sky_suppression'] != 0) & (
                    residuals_clean < -8.0)  # Extreme overestimation on clear days
                sample_weights[clear_sky_mask] *= 10.0  # Maximum weight
                print(
                    f"☀️ Clear sky cases weighted 10x: {np.sum(clear_sky_mask)}")

            print(
                f"🎯 Residual XGBoost: {np.sum(overestimate_mask)} severe overestimation cases weighted 6x")

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)

            # Enhanced XGBoost parameters for correcting overestimation
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 300,  # More trees for better correction
                'max_depth': 8,       # Deeper for complex solar interactions
                'learning_rate': 0.08,  # Moderate learning rate
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.05,    # Reduced regularization for more flexibility
                'reg_lambda': 0.05,
                'gamma': 0.1,
                'random_state': 42,
                'tree_method': 'auto'
            }

            # Train model with cross-validation and weighting
            cv_scores = []
            for train_idx, val_idx in tscv.split(ml_features_clean):
                X_train, X_val = ml_features_clean.iloc[train_idx], ml_features_clean.iloc[val_idx]
                y_train, y_val = residuals_clean[train_idx], residuals_clean[val_idx]
                w_train = sample_weights[train_idx]

                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_train, y_train, sample_weight=w_train)

                val_pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_pred)
                cv_scores.append(val_mae)

            # Train final model on all data with sample weighting
            final_model = xgb.XGBRegressor(**xgb_params)
            final_model.fit(ml_features_clean, residuals_clean,
                            sample_weight=sample_weights)

            avg_cv_score = np.mean(cv_scores)
            print("🎯 XGBoost residual model trained with overestimation focus")

            # Feature importance analysis with focus on solar features
            feature_importance = dict(
                zip(ml_features_clean.columns, final_model.feature_importances_))
            solar_features = {
                k: v for k, v in feature_importance.items() if 'solar' in k.lower()}

            print(
                f"☀️ Solar feature importance in residual model: {len(solar_features)} features")

            return {
                'model': final_model,
                'feature_columns': ml_features_clean.columns.tolist(),
                'cv_score': avg_cv_score,
                'feature_importance': feature_importance,
                'sample_weighting': {
                    'severe_overestimation': int(np.sum(overestimate_mask)),
                    'moderate_overestimation': int(np.sum(moderate_overest_mask)),
                    'total_samples': len(residuals_clean)
                }
            }

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

        # Convert to dict and add weather-aware categories
        result_records = results.to_dict(orient='records')
        result_records = self.add_weather_aware_price_categories(
            result_records)

        print(
            f"✅ ML-first forecast complete: {len(result_records)} predictions")
        return result_records

    def _create_enhanced_ml_features(self, df):
        """Create enhanced ML features with AGGRESSIVE solar interactions for extreme price prediction"""
        features = {}

        # Enhanced time features with solar interactions
        features['hour'] = df['hour']
        features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        features['is_offpeak'] = df['is_offpeak']
        features['is_peak'] = df['is_peak']
        features['is_weekend'] = df['is_weekend']

        # ENHANCED SOLAR FEATURES - Phase 1.1 Implementation
        if 'solar_factor' in df.columns:
            features['solar_factor'] = df['solar_factor']

            # Original strong interactions
            features['solar_hour_interaction'] = df['solar_factor'] * df['hour']
            features['solar_offpeak_interaction'] = df['solar_factor'] * \
                df['is_offpeak']
            features['solar_squared'] = df['solar_factor'] ** 2
            features['solar_midday_effect'] = df['solar_factor'] * \
                (df['hour'].between(12, 14).astype(int))

            # NEW: Non-linear solar terms for extreme price depression
            features['solar_factor_squared_offpeak'] = df.get(
                'solar_factor_squared_offpeak', 0)
            features['solar_factor_cubed_offpeak'] = df.get(
                'solar_factor_cubed_offpeak', 0)
            features['solar_hour_squared_offpeak'] = df.get(
                'solar_hour_squared_offpeak', 0)
            features['solar_exponential_offpeak'] = df.get(
                'solar_exponential_offpeak', 0)

            # Solar change indicators
            features['solar_factor_change_1h'] = df.get(
                'solar_factor_change_1h', 0)
            features['solar_rapid_increase'] = df.get(
                'solar_rapid_increase', 0)

            # Multiple levels of oversupply
            features['solar_oversupply_high'] = df.get(
                'solar_oversupply_high', 0)
            features['solar_oversupply_extreme'] = df.get(
                'solar_oversupply_extreme', 0)

            # Enhanced collapse features
            features['extreme_solar_collapse'] = df.get(
                'extreme_solar_collapse', 0)

            # CRITICAL: New solar suppression features for cloud cover < 30% effect
            features['solar_flood_effect'] = df.get('solar_flood_effect', 0)
            features['midday_flood_multiplier'] = df.get(
                'midday_flood_multiplier', 0)
            features['clear_sky_suppression'] = df.get(
                'clear_sky_suppression', 0)
            features['solar_suppression_active'] = df.get(
                'solar_suppression_active', 0)

            # Exponential solar effect during peak solar hours (enhanced)
            features['solar_exponential'] = np.exp(
                df['solar_factor'] * df['is_offpeak'] * -3)

        # ENHANCED TEMPERATURE-SOLAR INTERACTIONS
        if 'temperature' in df.columns:
            features['temperature'] = df['temperature']
            features['temp_squared'] = df['temperature'] ** 2
            features['temp_hour_interaction'] = df['temperature'] * df['hour']

            # NEW: Solar-temperature mild weather interaction
            features['solar_temp_mild_interaction'] = df.get(
                'solar_temp_mild_interaction', 0)

            # CRITICAL: Evening demand spike features for hot evenings + no solar
            features['is_evening_peak'] = df.get('is_evening_peak', 0)
            features['is_hot_evening'] = df.get('is_hot_evening', 0)
            features['evening_demand_spike'] = df.get(
                'evening_demand_spike', 0)
            features['no_solar_evening_effect'] = df.get(
                'no_solar_evening_effect', 0)
            features['extreme_heat_evening'] = df.get(
                'extreme_heat_evening', 0)
            features['compound_evening_demand'] = df.get(
                'compound_evening_demand', 0)

        # NEW: Weather pattern matching features
        features['weather_baseline_adjustment'] = df.get(
            'weather_baseline_adjustment', 0)
        features['similar_day_baseline'] = df.get('similar_day_baseline', 10.0)

        # Cloud cover effects with solar interactions
        if 'cloud_cover' in df.columns:
            features['cloud_cover'] = df['cloud_cover']
            features['clear_sky_indicator'] = (
                df['cloud_cover'] < 20).astype(int)
            features['cloudy_indicator'] = (df['cloud_cover'] > 70).astype(int)

            # NEW: Clear sky + high solar interaction
            if 'solar_factor' in df.columns:
                features['clear_sky_solar_interaction'] = (
                    features['clear_sky_indicator'] * df['solar_factor'] * df['is_offpeak'])

        # Enhanced demand proxy with solar interactions
        features['demand_proxy'] = df['demand_proxy']
        if 'solar_factor' in df.columns:
            features['demand_solar_interaction'] = df['demand_proxy'] * \
                df['solar_factor']
            features['demand_solar_offpeak_interaction'] = (
                df['demand_proxy'] * df['solar_factor'] * df['is_offpeak'])

        # PRICE-TARGETED FEATURES - Phase 1.2 Implementation

        # Enhanced off-peak price statistics
        features['offpeak_min_6h'] = df.get('offpeak_min_6h', 2.0)
        features['offpeak_median_24h'] = df.get('offpeak_median_24h', 8.0)

        # Traditional price-based features
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

        # Day of week effects with solar interactions
        features['day_of_week'] = df['day_of_week']
        features['is_workday'] = df['is_workday']

        # Weekend + high solar interaction
        if 'solar_factor' in df.columns:
            features['weekend_solar_interaction'] = df['is_weekend'] * \
                df['solar_factor']

        # Month seasonality with enhanced solar seasonal effects
        features['month'] = df['month']
        features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Summer + high solar interaction
        if 'solar_factor' in df.columns:
            summer_indicator = df['month'].between(6, 8).astype(int)
            features['summer_solar_interaction'] = summer_indicator * \
                df['solar_factor']

        # EXTREME CONDITION INDICATORS - Phase 1.3 Implementation

        # Perfect storm conditions (high solar + off-peak + mild weather + weekend)
        if 'solar_factor' in df.columns and 'temperature' in df.columns:
            mild_weather = ((df['temperature'] >= 15) & (
                df['temperature'] <= 25)).astype(int)
            features['perfect_storm_low_price'] = (
                df['is_offpeak'] * (df['solar_factor'] > 0.8).astype(int) *
                mild_weather * df['is_weekend'])

        # High solar + low demand proxy interaction
        if 'solar_factor' in df.columns:
            low_demand_indicator = (df['demand_proxy'] < -0.5).astype(int)
            features['solar_low_demand_interaction'] = (
                df['solar_factor'] * low_demand_indicator * df['is_offpeak'])

        df_features = pd.DataFrame(features)
        return self._validate_ml_features(df_features)

    def _train_primary_xgboost(self, X, y):
        """Train XGBoost as primary model with focus on extreme low-price scenarios"""
        try:
            # Enhanced parameters for extreme solar scenarios
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 400,  # More trees for better complex pattern learning
                'max_depth': 10,      # Deeper trees for complex solar interactions
                'learning_rate': 0.03,  # Slower learning for better extreme case capture
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.05,    # Reduced L1 regularization for more flexibility
                'reg_lambda': 0.05,   # Reduced L2 regularization for more flexibility
                'gamma': 0.1,         # Minimum split loss for pruning
                'random_state': 42,
                'tree_method': 'auto'
            }

            # Phase 2.2: Focus on low-price regions with sample weighting
            sample_weights = np.ones(len(y))

            # Give much higher weight to extreme low prices (< 3.0 cents/kWh)
            extreme_low_mask = y < 3.0
            sample_weights[extreme_low_mask] *= 5.0

            # Give higher weight to low prices (< 6.0 cents/kWh)
            low_price_mask = (y >= 3.0) & (y < 6.0)
            sample_weights[low_price_mask] *= 3.0

            # Give moderate weight to off-peak hours
            if 'is_offpeak' in X.columns:
                offpeak_mask = X['is_offpeak'] == 1
                sample_weights[offpeak_mask] *= 2.0

            print(
                f"🎯 Training XGBoost with sample weighting: {np.sum(extreme_low_mask)} extreme low samples weighted 5x")

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X, y, sample_weight=sample_weights)

            # Feature importance analysis with solar focus
            feature_importance = dict(
                zip(X.columns, model.feature_importances_))

            # Highlight solar-related features
            solar_features = {
                k: v for k, v in feature_importance.items() if 'solar' in k.lower()}
            top_features = sorted(feature_importance.items(),
                                  key=lambda x: x[1], reverse=True)[:8]
            top_solar_features = sorted(
                solar_features.items(), key=lambda x: x[1], reverse=True)[:5]

            print(
                f"🔝 Top features: {[f'{name}: {imp:.3f}' for name, imp in top_features]}")
            print(
                f"☀️ Top solar features: {[f'{name}: {imp:.3f}' for name, imp in top_solar_features]}")

            return {
                'model': model,
                'feature_columns': X.columns.tolist(),
                'feature_importance': feature_importance,
                'model_type': 'xgboost',
                'sample_weighting': {
                    'extreme_low_samples': int(np.sum(extreme_low_mask)),
                    'low_price_samples': int(np.sum(low_price_mask)),
                    'total_samples': len(y)
                }
            }

        except Exception as e:
            print(f"⚠️ XGBoost training failed: {str(e)}")
            return None

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

        # Convert to dict and add weather-aware categories
        result_records = future_prophet.to_dict(orient='records')
        result_records = self.add_weather_aware_price_categories(
            result_records)

        print("✅ Enhanced ensemble forecast complete")
        return result_records

    def get_weather_aware_price_category(self, price: float, hour: int, cloud_cover: float, temperature: float) -> str:
        """Get weather-aware price category instead of static thresholds"""

        # Define base thresholds
        base_cheap = 8.0
        base_moderate = 12.0
        base_expensive = 18.0

        # Solar suppression adjustment (cloud cover < 30%, hours 11-16)
        if 11 <= hour <= 16 and cloud_cover < 30:
            solar_factor = (100 - cloud_cover) / 100
            solar_adjustment = solar_factor * 0.6  # Up to 60% threshold reduction

            cheap_threshold = base_cheap * (1 - solar_adjustment)
            moderate_threshold = base_moderate * (1 - solar_adjustment * 0.4)
            expensive_threshold = base_expensive * (1 - solar_adjustment * 0.2)

        # Evening demand spike adjustment (temp > 25°C, hours 19-22)
        elif 19 <= hour <= 22 and temperature > 25:
            # Cap at 10°C above 25°C
            heat_factor = min((temperature - 25) / 10, 1.0)
            heat_adjustment = heat_factor * 0.4  # Up to 40% threshold increase

            cheap_threshold = base_cheap * (1 + heat_adjustment * 0.5)
            moderate_threshold = base_moderate * (1 + heat_adjustment)
            expensive_threshold = base_expensive * (1 + heat_adjustment * 1.5)

        # Standard thresholds for other conditions
        else:
            cheap_threshold = base_cheap
            moderate_threshold = base_moderate
            expensive_threshold = base_expensive

        # Categorize with dynamic thresholds
        if price <= cheap_threshold:
            return "cheap"
        elif price <= moderate_threshold:
            return "moderate"
        elif price <= expensive_threshold:
            return "expensive"
        else:
            return "very_expensive"

    def add_weather_aware_price_categories(self, results: list) -> list:
        """Add weather-aware price categories to forecast results"""

        for result in results:
            price = result.get('predicted_price_cents_kwh', 0)
            hour = result.get('hour', 12)

            # Get weather data from result or use defaults
            temperature = result.get('temperature_celsius', 20)

            # Estimate cloud cover based on time and season (if not available)
            # This is a fallback - ideally should come from weather forecast
            if 'cloud_cover' not in result:
                # Simple seasonal model: less clouds in summer, more in winter
                # More clouds in morning/evening, less at midday
                month = pd.Timestamp(result.get(
                    'timestamp', '2025-07-30')).month
                seasonal_clouds = 60 + 20 * \
                    np.sin(2 * np.pi * (month - 6) / 12)
                daily_variation = -10 * np.sin(2 * np.pi * (hour - 6) / 12)
                cloud_cover = max(
                    0, min(100, seasonal_clouds + daily_variation))
            else:
                cloud_cover = result['cloud_cover']

            # Get weather-aware category
            weather_category = self.get_weather_aware_price_category(
                price, hour, cloud_cover, temperature)

            # Add both standard and weather-aware categories for comparison
            result['weather_aware_category'] = weather_category

            # Also add context information
            result['category_context'] = {
                'cloud_cover': cloud_cover,
                'is_solar_suppression_period': (11 <= hour <= 16 and cloud_cover < 30),
                'is_evening_spike_period': (19 <= hour <= 22 and temperature > 25),
                'weather_threshold_adjustment': 'solar_suppression' if (11 <= hour <= 16 and cloud_cover < 30)
                else 'evening_spike' if (19 <= hour <= 22 and temperature > 25)
                else 'standard'
            }

        return results

    def analyze_weather_price_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze historical weather-price correlations for model validation"""
        try:
            df = self._prepare_data(self.repository.get_all_data())
            if df.empty or len(df) < 168:  # Need at least 1 week
                return {"error": "Insufficient historical data"}

            # Filter to recent data
            cutoff_date = df['ds'].max() - pd.Timedelta(days=days_back)
            recent_df = df[df['ds'] >= cutoff_date].copy()

            if len(recent_df) < 24:
                return {"error": "Insufficient recent data"}

            analysis = {
                "analysis_period": {
                    "start_date": recent_df['ds'].min().strftime('%Y-%m-%d'),
                    "end_date": recent_df['ds'].max().strftime('%Y-%m-%d'),
                    "total_hours": len(recent_df)
                }
            }

            # 1. Solar Suppression Analysis (cloud cover < 30%, hours 11-16)
            solar_suppression_mask = (
                (recent_df['hour'].between(11, 16)) &
                (recent_df['cloud_cover'] < 30)
            )
            solar_suppression_data = recent_df[solar_suppression_mask]

            if len(solar_suppression_data) > 0:
                baseline_midday = recent_df[recent_df['hour'].between(
                    11, 16)]['y'].median()
                suppression_median = solar_suppression_data['y'].median()
                suppression_effect = (
                    suppression_median / baseline_midday - 1) * 100

                analysis["solar_suppression"] = {
                    "periods_identified": len(solar_suppression_data),
                    "baseline_midday_price": baseline_midday,
                    "suppression_median_price": suppression_median,
                    "price_reduction_percent": suppression_effect,
                    "cheap_category_rate": (solar_suppression_data['y'] <= 8.0).mean() * 100,
                    "avg_cloud_cover": solar_suppression_data['cloud_cover'].mean(),
                    "avg_solar_factor": solar_suppression_data['solar_factor'].mean()
                }

            # 2. Evening Demand Spike Analysis (temp > 25°C, hours 19-22)
            evening_spike_mask = (
                (recent_df['hour'].between(19, 22)) &
                (recent_df['temperature'] > 25)
            )
            evening_spike_data = recent_df[evening_spike_mask]

            if len(evening_spike_data) > 0:
                baseline_evening = recent_df[recent_df['hour'].between(
                    19, 22)]['y'].median()
                spike_median = evening_spike_data['y'].median()
                spike_effect = (spike_median / baseline_evening - 1) * 100

                analysis["evening_spikes"] = {
                    "periods_identified": len(evening_spike_data),
                    "baseline_evening_price": baseline_evening,
                    "spike_median_price": spike_median,
                    "price_increase_percent": spike_effect,
                    "expensive_category_rate": (evening_spike_data['y'] >= 18.0).mean() * 100,
                    "avg_temperature": evening_spike_data['temperature'].mean(),
                    "max_temperature": evening_spike_data['temperature'].max()
                }

            # 3. Weather Pattern Correlations
            correlations = {}
            for weather_var in ['cloud_cover', 'temperature', 'solar_factor']:
                if weather_var in recent_df.columns:
                    correlation = recent_df[weather_var].corr(recent_df['y'])
                    correlations[weather_var] = correlation

            analysis["weather_correlations"] = correlations

            analysis["recommendations"] = []
            if "solar_suppression" in analysis and analysis["solar_suppression"]["price_reduction_percent"] < -20:
                analysis["recommendations"].append(
                    "Strong solar suppression detected - increase solar regressor weights")

            return analysis

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _add_prophet_regressors(self, model, df):
        """Add all Prophet regressors with AGGRESSIVE prior scales for solar impact"""
        # Time regressors with enhanced off-peak emphasis
        time_regressors = ['hour', 'is_weekend',
                           'is_peak', 'is_offpeak', 'month', 'is_workday']
        for reg in time_regressors:
            prior_scale = 75.0 if reg == 'is_offpeak' else 10.0  # Increased off-peak influence
            model.add_regressor(reg, prior_scale=prior_scale)

        # Off-peak specific regressors with MUCH higher prior scales
        offpeak_regressors = ['solar_peak_hours',
                              'is_midday_solar', 'summer_extreme_solar']
        for reg in offpeak_regressors:
            if reg in df.columns:
                prior_scale = 100.0 if reg == 'summer_extreme_solar' else 40.0  # Doubled influence
                model.add_regressor(reg, prior_scale=prior_scale)

        # Price-based regressors (minimal influence)
        for reg in ['price_lag1', 'price_lag24', 'price_ma7', 'demand_proxy']:
            if reg in df.columns:
                model.add_regressor(reg, prior_scale=0.1)

        if 'offpeak_ma3' in df.columns:
            # Slightly increased
            model.add_regressor('offpeak_ma3', prior_scale=0.5)

        # NEW: Enhanced off-peak price statistics with moderate influence
        if 'offpeak_min_6h' in df.columns:
            model.add_regressor('offpeak_min_6h', prior_scale=5.0)
        if 'offpeak_median_24h' in df.columns:
            model.add_regressor('offpeak_median_24h', prior_scale=3.0)

        # Weather regressors with enhanced solar emphasis
        weather_regressors = ['cloud_cover', 'temperature', 'solar_factor',
                              'solar_production_factor', 'temp_demand_factor', 'weather_price_impact']
        for reg in weather_regressors:
            if reg in df.columns:
                # Nearly doubled solar influence
                prior_scale = 15.0 if 'solar' in reg or 'impact' in reg else 3.0
                model.add_regressor(reg, prior_scale=prior_scale)

        # NEW: Aggressive non-linear solar interaction regressors
        solar_interaction_regressors = [
            'solar_factor_squared_offpeak', 'solar_factor_cubed_offpeak',
            'solar_hour_squared_offpeak', 'solar_exponential_offpeak',
            'solar_rapid_increase', 'solar_temp_mild_interaction'
        ]
        for reg in solar_interaction_regressors:
            if reg in df.columns:
                # Very high influence for non-linear terms
                model.add_regressor(reg, prior_scale=60.0)

        # NEW: Enhanced solar oversupply levels with extreme influence
        if 'solar_oversupply_high' in df.columns:
            model.add_regressor('solar_oversupply_high', prior_scale=80.0)
        if 'solar_oversupply_extreme' in df.columns:
            model.add_regressor('solar_oversupply_extreme',
                                prior_scale=120.0)  # Maximum influence

        # Critical solar collapse regressors with EXTREME prior scales
        if 'midday_solar_collapse' in df.columns:
            model.add_regressor('midday_solar_collapse',
                                prior_scale=100.0)  # Quadrupled influence
        if 'extreme_solar_collapse' in df.columns:
            # Maximum collapse influence
            model.add_regressor('extreme_solar_collapse', prior_scale=150.0)
        if 'solar_oversupply' in df.columns:
            # Quadrupled influence
            model.add_regressor('solar_oversupply', prior_scale=60.0)

        # NEW CRITICAL: Solar Suppression Features (cloud cover < 30% effect)
        if 'solar_flood_effect' in df.columns:
            model.add_regressor('solar_flood_effect',
                                prior_scale=200.0)  # Highest priority
        if 'midday_flood_multiplier' in df.columns:
            model.add_regressor('midday_flood_multiplier',
                                prior_scale=180.0)  # Very high
        if 'clear_sky_suppression' in df.columns:
            model.add_regressor('clear_sky_suppression',
                                prior_scale=220.0)  # Maximum suppression
        if 'solar_suppression_active' in df.columns:
            model.add_regressor('solar_suppression_active',
                                prior_scale=160.0)  # Strong indicator

        # NEW CRITICAL: Evening Demand Spike Features
        if 'evening_demand_spike' in df.columns:
            model.add_regressor('evening_demand_spike',
                                prior_scale=120.0)  # Strong evening spike
        if 'no_solar_evening_effect' in df.columns:
            model.add_regressor('no_solar_evening_effect',
                                prior_scale=100.0)  # High evening demand
        if 'is_hot_evening' in df.columns:
            # Hot evening indicator
            model.add_regressor('is_hot_evening', prior_scale=80.0)
        if 'is_evening_peak' in df.columns:
            # Evening peak indicator
            model.add_regressor('is_evening_peak', prior_scale=60.0)
        if 'extreme_heat_evening' in df.columns:
            # Extreme heat evenings
            model.add_regressor('extreme_heat_evening', prior_scale=140.0)
        if 'compound_evening_demand' in df.columns:
            # Compound evening effects
            model.add_regressor('compound_evening_demand', prior_scale=90.0)

        # NEW: Weather Pattern Matching Features
        if 'weather_baseline_adjustment' in df.columns:
            # Historical pattern adjustment
            model.add_regressor(
                'weather_baseline_adjustment', prior_scale=50.0)
        if 'similar_day_baseline' in df.columns:
            model.add_regressor('similar_day_baseline',
                                prior_scale=30.0)  # Similar day baseline

        # Off-peak weather regressors with much higher scales
        for reg in ['offpeak_solar_impact', 'offpeak_weather_vol']:
            if reg in df.columns:
                # Quadrupled solar impact influence
                prior_scale = 80.0 if 'solar' in reg else 20.0
                model.add_regressor(reg, prior_scale=prior_scale)

        # Solar change indicators with high influence
        if 'solar_factor_change_1h' in df.columns:
            model.add_regressor('solar_factor_change_1h', prior_scale=40.0)

        # Volatility and other regressors
        if 'price_volatility' in df.columns:
            model.add_regressor('price_volatility', prior_scale=0.1)
        if 'weather_volatility' in df.columns:
            model.add_regressor('weather_volatility', prior_scale=0.2)
        if 'recent_weight' in df.columns:
            model.add_regressor('recent_weight', prior_scale=0.8)

        # Enhanced seasonalities with higher fourier orders for better solar pattern capture
        model.add_seasonality(name='hourly', period=24,
                              fourier_order=15)  # Increased from 12
        model.add_seasonality(name='daily_weather', period=24,
                              fourier_order=8)  # Increased from 5

        if 'solar_seasonality_condition' in df.columns:
            model.add_seasonality(name='offpeak_solar', period=24,
                                  fourier_order=10, condition_name='solar_seasonality_condition')  # Increased from 6

        # Enhanced midday seasonality with very high prior scale for sharp solar effects
        model.add_seasonality(name='midday_sharp', period=24,
                              fourier_order=12, prior_scale=30.0)  # Doubled prior scale

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

        # Calculate enhanced weather features for all periods (includes all new solar features)
        future = self._enhance_weather_features(future)

        # Add summer solar feature AFTER weather features are calculated
        future['summer_extreme_solar'] = ((future['month'].between(6, 8)) & (
            future['hour'].between(11, 15)) & (future['solar_factor'] > 0.7)).astype(int)

        # Ensure the enhanced solar collapse features are available
        # (These are already created in _enhance_weather_features, but ensure consistency)
        if 'midday_solar_collapse' not in future.columns:
            future['midday_solar_collapse'] = np.where(
                (future['hour'].between(13, 15)) & (
                    future['solar_factor'] > 0.6),
                future['solar_factor'] ** 1.5 * -40.0, 0).round(2)

        if 'extreme_solar_collapse' not in future.columns:
            future['extreme_solar_collapse'] = np.where(
                (future['hour'].between(12, 14)) & (
                    future['solar_factor'] > 0.85),
                future['solar_factor'] ** 2 * -60.0, 0).round(2)

        if 'solar_oversupply' not in future.columns:
            future['solar_oversupply'] = np.where(
                (future['solar_factor'] > 0.8) & (future['hour'].between(12, 15)), 1, 0)

        # Ensure critical new solar suppression features are available
        if 'solar_suppression_active' not in future.columns:
            future['solar_suppression_active'] = ((future['cloud_cover'] < 30) &
                                                  (future['hour'].between(11, 16))).astype(int)
        if 'solar_flood_effect' not in future.columns:
            future['solar_flood_effect'] = np.where(
                future['solar_suppression_active'] == 1,
                (future['solar_factor'] ** 3) * -80.0, 0).round(2)
        if 'midday_flood_multiplier' not in future.columns:
            future['midday_flood_multiplier'] = np.where(
                (future['hour'].between(11, 16)) & (
                    future['cloud_cover'] < 30),
                future['solar_factor'] * -120.0, 0).round(2)
        if 'clear_sky_suppression' not in future.columns:
            future['clear_sky_suppression'] = np.where(
                (future['hour'].between(11, 16)) & (
                    future['cloud_cover'] < 20),
                (1 - future['cloud_cover']/100) * future['solar_factor'] * -150.0, 0).round(2)

        # Ensure critical evening demand spike features are available
        if 'is_evening_peak' not in future.columns:
            future['is_evening_peak'] = future['hour'].between(
                19, 22).astype(int)
        if 'is_hot_evening' not in future.columns:
            future['is_hot_evening'] = ((future['temperature'] > 25) &
                                        future['is_evening_peak']).astype(int)
        if 'evening_demand_spike' not in future.columns:
            cooling_demand = np.where(future['temperature'] > 25,
                                      ((future['temperature'] - 25) / 5) ** 1.5, 0)
            future['evening_demand_spike'] = np.where(
                future['is_hot_evening'] == 1,
                cooling_demand * 80.0, 0).round(2)
        if 'extreme_heat_evening' not in future.columns:
            future['extreme_heat_evening'] = np.where(
                (future['temperature'] > 30) & future['is_evening_peak'],
                ((future['temperature'] - 30) / 2) * 120.0, 0).round(2)
        if 'no_solar_evening_effect' not in future.columns:
            temp_demand = np.where(future['temperature'] < 15, (15 - future['temperature']) / 10,
                                   np.where(future['temperature'] > 25, (future['temperature'] - 25) / 10, 0))
            future['no_solar_evening_effect'] = np.where(
                future['is_evening_peak'] == 1,
                temp_demand * 50.0, 0).round(2)
        if 'compound_evening_demand' not in future.columns:
            temp_demand = np.where(future['temperature'] < 15, (15 - future['temperature']) / 10,
                                   np.where(future['temperature'] > 25, (future['temperature'] - 25) / 10, 0))
            future['compound_evening_demand'] = np.where(
                (future['is_hot_evening'] == 1) & (temp_demand > 0.5),
                temp_demand * future['evening_demand_spike'] * 0.3, 0).round(2)

        # Ensure weather pattern matching features are available
        if 'weather_baseline_adjustment' not in future.columns:
            # No adjustment for future predictions
            future['weather_baseline_adjustment'] = 0
        if 'similar_day_baseline' not in future.columns:
            future['similar_day_baseline'] = 10.0  # Default baseline

        # Ensure enhanced off-peak price features for future predictions
        if 'offpeak_min_6h' not in future.columns:
            future['offpeak_min_6h'] = 2.0  # Conservative estimate for future
        if 'offpeak_median_24h' not in future.columns:
            # Conservative estimate for future
            future['offpeak_median_24h'] = 8.0

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
        """Enhanced off-peak analysis with bias detection for solar scenarios"""
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
            very_low_count = len(offpeak_data[offpeak_data['y'] < 1.0])

            # Phase 3.1: Enhanced bias analysis for solar scenarios
            bias_analysis = self._analyze_prediction_bias(recent_df)

            return {
                'analysis_period': f"Last {days_back} days",
                'offpeak_hours_analyzed': len(offpeak_data),
                'avg_offpeak_price': round(offpeak_mean, 2),
                'avg_midday_price': round(midday_mean, 2),
                'offpeak_price_std': round(offpeak_std, 2),
                'offpeak_volatility': round(offpeak_volatility, 2),
                'extreme_low_prices_count': extreme_low_count,
                'very_low_prices_count': very_low_count,
                'extreme_low_percentage': round(extreme_low_count / len(offpeak_data) * 100, 1),
                'solar_price_correlation': round(solar_correlation, 3),
                'bias_analysis': bias_analysis,
                'recommendation': self._get_enhanced_offpeak_recommendation(
                    offpeak_volatility, solar_correlation, extreme_low_count, bias_analysis)
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _analyze_prediction_bias(self, df):
        """Analyze prediction bias across different time segments and solar conditions"""
        if len(df) < 24:
            return {"error": "Insufficient data for bias analysis"}

        try:
            # Generate recent predictions for comparison
            recent_forecast = self.forecast(hours_ahead=min(48, len(df)))

            if not recent_forecast:
                return {"error": "Could not generate predictions for bias analysis"}

            # Convert to DataFrame for analysis
            forecast_df = pd.DataFrame(recent_forecast)
            forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

            # Merge with actual data
            df_analysis = df.copy()
            df_analysis = df_analysis.merge(
                forecast_df[['timestamp', 'predicted_price_cents_kwh']],
                left_on='ds', right_on='timestamp', how='inner'
            )

            if len(df_analysis) < 10:
                return {"error": "Insufficient overlap for bias analysis"}

            # Calculate bias (predicted - actual)
            df_analysis['bias'] = df_analysis['predicted_price_cents_kwh'] - \
                df_analysis['y']

            bias_results = {}

            # Overall bias
            bias_results['overall'] = {
                'mean_bias': round(df_analysis['bias'].mean(), 3),
                'abs_bias': round(df_analysis['bias'].abs().mean(), 3),
                'samples': len(df_analysis)
            }

            # Off-peak bias (critical for solar scenarios)
            offpeak_mask = df_analysis['hour'].between(10, 15)
            if offpeak_mask.any():
                offpeak_bias = df_analysis[offpeak_mask]['bias']
                bias_results['offpeak'] = {
                    'mean_bias': round(offpeak_bias.mean(), 3),
                    'abs_bias': round(offpeak_bias.abs().mean(), 3),
                    'samples': len(offpeak_bias),
                    'overestimation_rate': round((offpeak_bias > 2.0).sum() / len(offpeak_bias) * 100, 1)
                }

            # High solar bias (if solar data available)
            if 'solar_factor' in df_analysis.columns:
                high_solar_mask = df_analysis['solar_factor'] > 0.7
                if high_solar_mask.any():
                    high_solar_bias = df_analysis[high_solar_mask]['bias']
                    bias_results['high_solar'] = {
                        'mean_bias': round(high_solar_bias.mean(), 3),
                        'abs_bias': round(high_solar_bias.abs().mean(), 3),
                        'samples': len(high_solar_bias),
                        'overestimation_rate': round((high_solar_bias > 3.0).sum() / len(high_solar_bias) * 100, 1)
                    }

            # Extreme low price bias
            extreme_low_mask = df_analysis['y'] < 3.0
            if extreme_low_mask.any():
                extreme_low_bias = df_analysis[extreme_low_mask]['bias']
                bias_results['extreme_low_prices'] = {
                    'mean_bias': round(extreme_low_bias.mean(), 3),
                    'abs_bias': round(extreme_low_bias.abs().mean(), 3),
                    'samples': len(extreme_low_bias),
                    'overestimation_rate': round((extreme_low_bias > 1.0).sum() / len(extreme_low_bias) * 100, 1)
                }

            return bias_results

        except Exception as e:
            return {"error": f"Bias analysis failed: {str(e)}"}

    def _get_enhanced_offpeak_recommendation(self, volatility, solar_corr, extreme_low_count, bias_analysis):
        """Enhanced recommendations based on off-peak analysis and bias detection"""
        recommendations = []

        # Traditional volatility and correlation checks
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

        # New bias-based recommendations
        if isinstance(bias_analysis, dict) and 'offpeak' in bias_analysis:
            offpeak_bias = bias_analysis['offpeak']['mean_bias']
            overest_rate = bias_analysis['offpeak'].get(
                'overestimation_rate', 0)

            if offpeak_bias > 3.0:
                recommendations.append(
                    f"⚠️ CRITICAL: Severe off-peak overestimation bias ({offpeak_bias:.1f} cents/kWh)")
                recommendations.append(
                    "🔧 URGENT: Increase solar regressor prior scales and retrain model")
            elif offpeak_bias > 1.5:
                recommendations.append(
                    f"⚠️ Moderate off-peak overestimation bias ({offpeak_bias:.1f} cents/kWh)")
                recommendations.append(
                    "🔧 Consider: Enhance solar interaction features and sample weighting")

            if overest_rate > 50:
                recommendations.append(
                    f"⚠️ High overestimation rate: {overest_rate}% of off-peak predictions too high")

        if isinstance(bias_analysis, dict) and 'high_solar' in bias_analysis:
            solar_bias = bias_analysis['high_solar']['mean_bias']
            if solar_bias > 4.0:
                recommendations.append(
                    f"⚠️ CRITICAL: High solar scenario overestimation ({solar_bias:.1f} cents/kWh)")
                recommendations.append(
                    "🔧 URGENT: Implement aggressive solar collapse features")

        if isinstance(bias_analysis, dict) and 'extreme_low_prices' in bias_analysis:
            extreme_bias = bias_analysis['extreme_low_prices']['mean_bias']
            if extreme_bias > 2.0:
                recommendations.append(
                    f"⚠️ Model failing on extreme low prices (bias: {extreme_bias:.1f} cents/kWh)")
                recommendations.append(
                    "🔧 Implement extreme price sample weighting and non-linear solar terms")

        if not recommendations:
            recommendations.append(
                "✅ Off-peak patterns stable - current model performing well")

        return recommendations

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
