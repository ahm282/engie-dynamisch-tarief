# Prophet Service Configuration Management

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class ProphetModelConfig:
    """Prophet model configuration"""
    changepoint_prior_scale: float = 0.02
    seasonality_prior_scale: float = 2.0
    seasonality_mode: str = 'additive'
    daily_seasonality: bool = False
    weekly_seasonality: bool = True
    yearly_seasonality: bool = False
    interval_width: float = 0.8
    changepoint_range: float = 0.8


@dataclass
class XGBoostConfig:
    """XGBoost model configuration"""
    objective: str = 'reg:squarederror'
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    tree_method: str = 'auto'


@dataclass
class LightGBMConfig:
    """LightGBM model configuration"""
    objective: str = 'regression'
    metric: str = 'mae'
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    learning_rate: float = 0.1
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    verbose: int = -1
    random_state: int = 42
    num_boost_round: int = 200


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration"""
    max_features: int = 20
    include_price_lags: bool = True
    include_weather_interactions: bool = True
    include_solar_features: bool = True
    price_lag_hours: List[int] = None
    rolling_window_sizes: List[int] = None
    feature_selection_enabled: bool = True
    min_data_for_complex_features: int = 168  # 1 week

    def __post_init__(self):
        if self.price_lag_hours is None:
            self.price_lag_hours = [1, 24, 168]  # 1h, 1d, 1w
        if self.rolling_window_sizes is None:
            self.rolling_window_sizes = [6, 12, 24]  # 6h, 12h, 24h


@dataclass
class DataQualityConfig:
    """Data quality thresholds"""
    min_data_points: int = 168  # 1 week
    max_missing_percentage: float = 15.0
    outlier_z_threshold: float = 3.0
    max_temporal_gap_hours: int = 6
    min_quality_score: float = 0.6
    enable_data_validation: bool = True
    enable_outlier_detection: bool = True


@dataclass
class EnsembleConfig:
    """Ensemble forecasting configuration"""
    prophet_weight: float = 0.4
    xgboost_weight: float = 0.35
    lightgbm_weight: float = 0.25
    adaptive_weights: bool = True
    cv_splits: int = 3
    uncertainty_quantification: bool = True
    parallel_training: bool = True
    model_timeout_seconds: int = 120


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    performance_tracking: bool = True
    error_tracking: bool = True
    prediction_validation: bool = True
    max_log_entries: int = 1000
    enable_metrics_export: bool = False


@dataclass
class ProphetServiceConfig:
    """Complete Prophet service configuration"""
    prophet: ProphetModelConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    features: FeatureEngineeringConfig
    data_quality: DataQualityConfig
    ensemble: EnsembleConfig
    monitoring: MonitoringConfig

    # Service settings
    default_forecast_hours: int = 48
    cache_ttl_seconds: int = 3600
    enable_ml_models: bool = True
    enable_caching: bool = True

    # Paths
    model_cache_dir: str = "../utils/prophet_models"
    config_file: str = "prophet_service.json"

    @classmethod
    def from_file(cls, config_path: str) -> 'ProphetServiceConfig':
        """Load configuration from JSON file"""
        config_path = Path(config_path)

        if not config_path.exists():
            # Create default config
            default_config = cls.default()
            default_config.save_to_file(config_path)
            return default_config

        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Error loading config from {config_path}: {e}")
            print("Using default configuration")
            return cls.default()

        # Create configuration with nested dataclasses
        try:
            return cls(
                prophet=ProphetModelConfig(**config_dict.get('prophet', {})),
                xgboost=XGBoostConfig(**config_dict.get('xgboost', {})),
                lightgbm=LightGBMConfig(**config_dict.get('lightgbm', {})),
                features=FeatureEngineeringConfig(
                    **config_dict.get('features', {})),
                data_quality=DataQualityConfig(
                    **config_dict.get('data_quality', {})),
                ensemble=EnsembleConfig(**config_dict.get('ensemble', {})),
                monitoring=MonitoringConfig(
                    **config_dict.get('monitoring', {})),
                **{k: v for k, v in config_dict.items() if k not in
                   ['prophet', 'xgboost', 'lightgbm', 'features', 'data_quality', 'ensemble', 'monitoring']}
            )
        except TypeError as e:
            print(f"⚠️ Error creating config from file: {e}")
            print("Using default configuration")
            return cls.default()

    @classmethod
    def default(cls) -> 'ProphetServiceConfig':
        """Create default configuration"""
        return cls(
            prophet=ProphetModelConfig(),
            xgboost=XGBoostConfig(),
            lightgbm=LightGBMConfig(),
            features=FeatureEngineeringConfig(),
            data_quality=DataQualityConfig(),
            ensemble=EnsembleConfig(),
            monitoring=MonitoringConfig()
        )

    @classmethod
    def from_environment(cls) -> 'ProphetServiceConfig':
        """Create configuration from environment variables"""
        config = cls.default()

        # Override with environment variables
        if os.getenv('PROPHET_CACHE_TTL'):
            config.cache_ttl_seconds = int(os.getenv('PROPHET_CACHE_TTL'))

        if os.getenv('PROPHET_ENABLE_ML'):
            config.enable_ml_models = os.getenv(
                'PROPHET_ENABLE_ML').lower() == 'true'

        if os.getenv('PROPHET_MODEL_DIR'):
            config.model_cache_dir = os.getenv('PROPHET_MODEL_DIR')

        if os.getenv('PROPHET_LOG_LEVEL'):
            config.monitoring.log_level = os.getenv('PROPHET_LOG_LEVEL')

        if os.getenv('PROPHET_LOG_DIR'):
            config.monitoring.log_dir = os.getenv('PROPHET_LOG_DIR')

        # XGBoost overrides
        if os.getenv('XGBOOST_N_ESTIMATORS'):
            config.xgboost.n_estimators = int(
                os.getenv('XGBOOST_N_ESTIMATORS'))

        if os.getenv('XGBOOST_MAX_DEPTH'):
            config.xgboost.max_depth = int(os.getenv('XGBOOST_MAX_DEPTH'))

        # Data quality overrides
        if os.getenv('PROPHET_MIN_DATA_POINTS'):
            config.data_quality.min_data_points = int(
                os.getenv('PROPHET_MIN_DATA_POINTS'))

        return config

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)

        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"✅ Configuration saved to {config_path}")
        except Exception as e:
            print(f"⚠️ Failed to save configuration: {e}")

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get parameters for specific model type"""
        model_type = model_type.lower()
        if model_type == 'prophet':
            return asdict(self.prophet)
        elif model_type == 'xgboost':
            return asdict(self.xgboost)
        elif model_type == 'lightgbm':
            # Convert to LightGBM format
            params = asdict(self.lightgbm)
            # Remove non-parameter fields
            params.pop('num_boost_round', None)
            return params
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def validate(self) -> Dict[str, Any]:
        """Validate configuration parameters"""
        issues = []
        warnings = []

        # Prophet validation
        if not 0.001 <= self.prophet.changepoint_prior_scale <= 1.0:
            warnings.append(
                "Prophet changepoint_prior_scale should be between 0.001 and 1.0")

        if self.prophet.seasonality_prior_scale <= 0:
            issues.append("Prophet seasonality_prior_scale must be positive")

        # XGBoost validation
        if self.xgboost.n_estimators < 50:
            warnings.append(
                "XGBoost n_estimators might be too low for good performance")
        elif self.xgboost.n_estimators > 1000:
            warnings.append(
                "XGBoost n_estimators might be too high, could cause overfitting")

        if not 0.01 <= self.xgboost.learning_rate <= 0.3:
            warnings.append(
                "XGBoost learning_rate should typically be between 0.01 and 0.3")

        if not 3 <= self.xgboost.max_depth <= 10:
            warnings.append(
                "XGBoost max_depth should typically be between 3 and 10")

        # LightGBM validation
        if self.lightgbm.num_leaves >= 2 ** self.xgboost.max_depth:
            warnings.append(
                "LightGBM num_leaves might be too high relative to max_depth")

        # Data quality validation
        if self.data_quality.min_data_points < 24:
            issues.append("Minimum data points should be at least 24 hours")

        if self.data_quality.max_missing_percentage > 50:
            issues.append("Maximum missing percentage should not exceed 50%")

        if self.data_quality.min_quality_score < 0 or self.data_quality.min_quality_score > 1:
            issues.append("Quality score must be between 0 and 1")

        # Ensemble validation
        weight_sum = self.ensemble.prophet_weight + \
            self.ensemble.xgboost_weight + self.ensemble.lightgbm_weight
        if abs(weight_sum - 1.0) > 0.01:
            issues.append(
                f"Ensemble weights should sum to 1.0, got {weight_sum:.3f}")

        if any(w < 0 for w in [self.ensemble.prophet_weight, self.ensemble.xgboost_weight, self.ensemble.lightgbm_weight]):
            issues.append("Ensemble weights must be non-negative")

        # Feature engineering validation
        if self.features.max_features < 5:
            warnings.append(
                "Max features might be too low for good performance")
        elif self.features.max_features > 50:
            warnings.append(
                "Max features might be too high, could cause overfitting")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'forecast_hours': self.default_forecast_hours,
            'ml_enabled': self.enable_ml_models,
            'caching_enabled': self.enable_caching,
            'cache_ttl': self.cache_ttl_seconds,
            'parallel_training': self.ensemble.parallel_training,
            'data_validation': self.data_quality.enable_data_validation,
            'monitoring_enabled': self.monitoring.performance_tracking,
            'log_level': self.monitoring.log_level,
            'model_weights': {
                'prophet': self.ensemble.prophet_weight,
                'xgboost': self.ensemble.xgboost_weight,
                'lightgbm': self.ensemble.lightgbm_weight
            }
        }


class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Default to config directory relative to this file
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.main_config_path = self.config_dir / "prophet_service.json"
        self._config: Optional[ProphetServiceConfig] = None

    @property
    def config(self) -> ProphetServiceConfig:
        """Get current configuration (lazy loading)"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self, config_path: Optional[str] = None) -> ProphetServiceConfig:
        """Load configuration from file or environment"""
        if config_path:
            return ProphetServiceConfig.from_file(config_path)

        # Try main config file first
        if self.main_config_path.exists():
            config = ProphetServiceConfig.from_file(str(self.main_config_path))
        else:
            # Fall back to environment variables and create default
            config = ProphetServiceConfig.from_environment()
            # Save for future use
            self.save_config(config)

        # Validate configuration
        validation = config.validate()
        if not validation['is_valid']:
            print(f"❌ Invalid configuration: {validation['issues']}")
            # Don't raise error, just warn and use anyway
            print("⚠️ Continuing with invalid configuration - please review settings")

        if validation['warnings']:
            print(f"⚠️ Configuration warnings: {validation['warnings']}")

        return config

    def save_config(self, config: ProphetServiceConfig, config_path: Optional[str] = None):
        """Save configuration to file"""
        save_path = config_path or str(self.main_config_path)
        config.save_to_file(save_path)

    def reload_config(self):
        """Reload configuration from file"""
        self._config = None
        return self.config

    def update_config_value(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if self._config is None:
            self._config = self.load_config()

        if hasattr(self._config, section):
            section_obj = getattr(self._config, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                # Save updated config
                self.save_config(self._config)
                print(f"✅ Updated {section}.{key} = {value}")
            else:
                print(f"⚠️ Key '{key}' not found in section '{section}'")
        else:
            print(f"⚠️ Section '{section}' not found")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get human-readable configuration summary"""
        return self.config.get_summary()


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_dir: str = None) -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def get_config() -> ProphetServiceConfig:
    """Get the current configuration"""
    return get_config_manager().config


# Usage example and testing
if __name__ == "__main__":
    # Test configuration management
    config_manager = ConfigManager()
    config = config_manager.config

    print("🔧 Prophet Service Configuration Summary:")
    summary = config.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n📊 Configuration validation:")
    validation = config.validate()
    print(f"  Valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"  Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")

    # Test model parameter extraction
    print(f"\n🤖 XGBoost Parameters:")
    xgb_params = config.get_model_params('xgboost')
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
