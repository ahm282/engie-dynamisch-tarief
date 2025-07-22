import pyowm
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from .weather_utils import OWM_API_KEY, GENT_COORDS


class WeatherCollector:
    def __init__(self):
        if not OWM_API_KEY:
            raise ValueError(
                "❌ OWM_API_KEY not found. Please check your .env file contains: OWM_API_KEY=your_api_key")

        try:
            self.owm = pyowm.OWM(OWM_API_KEY)
            self.mgr = self.owm.weather_manager()
            print(f"✅ Weather API initialized with key: {OWM_API_KEY[:10]}...")
        except Exception as e:
            raise ValueError(f"❌ Failed to initialize Weather API: {e}")

    def get_current_weather(self):
        """Get current weather data with proper rounding"""
        try:
            observation = self.mgr.weather_at_coords(
                GENT_COORDS['lat'],
                GENT_COORDS['lon']
            )
            weather = observation.weather

            return {
                'timestamp': datetime.now(timezone.utc),
                'cloud_cover': round(weather.clouds, 2),
                'temperature': round(weather.temperature('celsius')['temp'], 2),
                'solar_factor': round(self._calculate_solar_factor(weather.clouds), 2)
            }
        except Exception as e:
            print(f"❌ Error fetching current weather: {e}")
            return None

    def _calculate_solar_factor(self, cloud_cover):
        """Calculate solar production factor based on cloud cover"""
        solar_factor = max(0, (100 - cloud_cover) / 100)
        return round(solar_factor, 2)

    def get_forecast(self, hours=24):
        """Get weather forecast for next N hours with timezone awareness and proper filtering"""
        try:
            forecast = self.mgr.forecast_at_coords(
                GENT_COORDS['lat'],
                GENT_COORDS['lon'],
                '3h'
            )

            weather_data = []
            current_time = datetime.now(timezone.utc)

            # Filter forecasts to only include future data
            for weather in forecast.forecast.weathers:
                forecast_time = datetime.fromtimestamp(
                    weather.reference_time('unix'), timezone.utc)

                # Only include forecasts that are in the future
                if forecast_time > current_time:
                    weather_data.append({
                        'timestamp': forecast_time.isoformat(),
                        'cloud_cover': round(weather.clouds, 2),
                        'temperature': round(weather.temperature('celsius')['temp'], 2),
                        'solar_factor': round(self._calculate_solar_factor(weather.clouds), 2)
                    })

                # Stop when we have enough hours of forecast data
                if len(weather_data) >= hours // 3:
                    break

            if not weather_data:
                print("⚠️ No future weather forecast data available")
                return pd.DataFrame()

            df = pd.DataFrame(weather_data)
            print(f"✅ Retrieved {len(df)} future weather forecast points")
            return df

        except Exception as e:
            print(f"❌ Error fetching weather forecast: {e}")
            return pd.DataFrame()

    def get_historical_weather_proxy(self, timestamps):
        """Generate weather proxy data for historical timestamps with proper rounding"""
        # For historical data, use seasonal patterns as proxy
        weather_data = []
        for timestamp in timestamps:
            dt = pd.to_datetime(timestamp)
            # Simple seasonal model for missing historical data
            hour = dt.hour
            month = dt.month

            # Simulate cloud cover based on season and time
            base_clouds = 50 + 30 * \
                np.sin(2 * np.pi * (month - 6) / 12)  # Seasonal variation
            daily_variation = 20 * \
                np.sin(2 * np.pi * hour / 24)  # Daily variation
            cloud_cover = np.clip(base_clouds + daily_variation, 0, 100)

            # Calculate temperature and solar factor with proper rounding
            temperature = 15 + 10 * np.sin(2 * np.pi * (month - 3) / 12)
            solar_factor = self._calculate_solar_factor(cloud_cover)

            weather_data.append({
                'timestamp': timestamp,
                'cloud_cover': round(cloud_cover, 2),
                'temperature': round(temperature, 2),
                'solar_factor': round(solar_factor, 2)
            })

        return pd.DataFrame(weather_data)
