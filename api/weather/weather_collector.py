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

            timestamp = datetime.now(timezone.utc)
            return {
                'timestamp': timestamp,
                'cloud_cover': round(weather.clouds, 2),
                'temperature': round(weather.temperature('celsius')['temp'], 2),
                'solar_factor': round(self._calculate_solar_factor(weather.clouds, timestamp), 2)
            }
        except Exception as e:
            print(f"❌ Error fetching current weather: {e}")
            return None

    def _calculate_solar_factor(self, cloud_cover, timestamp=None):
        """
        Calculate solar production factor based on cloud cover AND time of day.
        Solar panels cannot generate electricity without sunlight!
        
        Args:
            cloud_cover: Cloud coverage percentage (0-100)
            timestamp: DateTime object to determine sun position (optional)
        
        Returns:
            Solar factor between 0.0 and 1.0
        """
        if timestamp is None:
            # If no timestamp provided, use current time (for backward compatibility)
            timestamp = datetime.now(timezone.utc)
        
        # Get hour in local time (approximate for Belgium - UTC+1/+2)
        # For simplicity, using UTC+1 (winter time)
        local_hour = (timestamp.hour + 1) % 24
        
        # Solar generation is only possible during daylight hours
        # For Belgium in summer: roughly sunrise ~5:30, sunset ~21:30
        # For Belgium in winter: roughly sunrise ~8:30, sunset ~17:00
        
        # Get month to adjust for seasonal variation
        month = timestamp.month
        
        # Define sunrise and sunset hours by month (approximate for Belgium)
        if month in [11, 12, 1, 2]:  # Winter months
            sunrise_hour = 8.0
            sunset_hour = 17.0
        elif month in [3, 4, 9, 10]:  # Spring/Autumn
            sunrise_hour = 7.0
            sunset_hour = 19.0
        else:  # Summer months (5-8)
            sunrise_hour = 5.5
            sunset_hour = 21.5
        
        # No solar generation during night hours
        if local_hour < sunrise_hour or local_hour > sunset_hour:
            return 0.0
        
        # Calculate solar angle efficiency (lower at dawn/dusk)
        daylight_hours = sunset_hour - sunrise_hour
        hour_from_sunrise = local_hour - sunrise_hour
        hour_from_noon = abs((sunrise_hour + daylight_hours / 2) - local_hour)
        
        # Solar efficiency based on sun angle (peaks at solar noon)
        max_efficiency_hours = daylight_hours / 4  # High efficiency for middle 50% of day
        if hour_from_noon <= max_efficiency_hours:
            sun_efficiency = 1.0  # Peak efficiency
        else:
            # Gradual reduction towards dawn/dusk
            sun_efficiency = max(0.1, 1.0 - (hour_from_noon - max_efficiency_hours) / (daylight_hours / 2 - max_efficiency_hours))
        
        # Cloud reduction factor (clear sky = 1.0, overcast = 0.1)
        cloud_efficiency = max(0.1, (100 - cloud_cover) / 100)
        
        # Combined solar factor
        solar_factor = sun_efficiency * cloud_efficiency
        
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
                        'solar_factor': round(self._calculate_solar_factor(weather.clouds, forecast_time), 2)
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

    def get_actual_weather_for_date(self, target_date):
        """Get actual weather data for today/tomorrow from API, with hourly interpolation"""
        try:
            current_time = datetime.now(timezone.utc)

            # Ensure target_date is timezone-aware
            if target_date.tzinfo is None:
                target_date = target_date.replace(tzinfo=timezone.utc)

            target_date_str = target_date.strftime('%Y-%m-%d')
            current_date_str = current_time.strftime('%Y-%m-%d')
            tomorrow_str = (current_time.replace(hour=0, minute=0, second=0, microsecond=0) +
                            pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            # Only use API data for today and tomorrow
            if target_date_str not in [current_date_str, tomorrow_str]:
                return None  # Use proxy data for other dates

            # Get forecast data from API
            forecast = self.mgr.forecast_at_coords(
                GENT_COORDS['lat'], GENT_COORDS['lon'], '3h'
            )

            # Extract forecasts for target date
            target_forecasts = []
            for weather in forecast.forecast.weathers:
                forecast_time = datetime.fromtimestamp(
                    weather.reference_time('unix'), timezone.utc)
                forecast_date = forecast_time.strftime('%Y-%m-%d')

                if forecast_date == target_date_str:
                    target_forecasts.append({
                        'time': forecast_time,
                        'temperature': weather.temperature('celsius')['temp'],
                        'cloud_cover': weather.clouds,
                        'solar_factor': self._calculate_solar_factor(weather.clouds, forecast_time)
                    })

            if not target_forecasts:
                return None  # No API data available for this date

            # Generate hourly data by interpolating between 3-hour forecasts
            weather_data = []
            for hour in range(24):
                target_hour = target_date.replace(
                    hour=hour, minute=0, second=0, microsecond=0)

                # Find the closest forecast(s)
                if len(target_forecasts) == 1:
                    # Only one forecast available, use it for all hours
                    closest = target_forecasts[0]
                    weather_data.append({
                        'timestamp': target_hour,
                        'cloud_cover': round(closest['cloud_cover'], 2),
                        'temperature': round(closest['temperature'], 2),
                        'solar_factor': round(closest['solar_factor'], 2)
                    })
                else:
                    # Multiple forecasts, interpolate
                    time_diffs = [abs((f['time'] - target_hour).total_seconds())
                                  for f in target_forecasts]
                    closest_idx = time_diffs.index(min(time_diffs))

                    # If within 1.5 hours of a forecast, use it directly
                    if time_diffs[closest_idx] <= 5400:  # 1.5 hours
                        closest = target_forecasts[closest_idx]
                        weather_data.append({
                            'timestamp': target_hour,
                            'cloud_cover': round(closest['cloud_cover'], 2),
                            'temperature': round(closest['temperature'], 2),
                            'solar_factor': round(closest['solar_factor'], 2)
                        })
                    else:
                        # Interpolate between two closest forecasts
                        if closest_idx > 0:
                            f1 = target_forecasts[closest_idx - 1]
                            f2 = target_forecasts[closest_idx]
                        elif closest_idx < len(target_forecasts) - 1:
                            f1 = target_forecasts[closest_idx]
                            f2 = target_forecasts[closest_idx + 1]
                        else:
                            f1 = f2 = target_forecasts[closest_idx]

                        # Simple linear interpolation
                        if f1 != f2:
                            t1_seconds = (
                                f1['time'] - target_hour).total_seconds()
                            t2_seconds = (
                                f2['time'] - target_hour).total_seconds()
                            if t2_seconds != t1_seconds:
                                weight = abs(t1_seconds) / \
                                    (abs(t1_seconds) + abs(t2_seconds))
                                temperature = f1['temperature'] * \
                                    (1 - weight) + f2['temperature'] * weight
                                cloud_cover = f1['cloud_cover'] * \
                                    (1 - weight) + f2['cloud_cover'] * weight
                            else:
                                temperature = f1['temperature']
                                cloud_cover = f1['cloud_cover']
                        else:
                            temperature = f1['temperature']
                            cloud_cover = f1['cloud_cover']

                        weather_data.append({
                            'timestamp': target_hour,
                            'cloud_cover': round(cloud_cover, 2),
                            'temperature': round(temperature, 2),
                            'solar_factor': round(self._calculate_solar_factor(cloud_cover, target_hour), 2)
                        })

            return pd.DataFrame(weather_data)

        except Exception as e:
            print(f"❌ Error getting actual weather for {target_date_str}: {e}")
            return None

    def get_weather_for_date(self, target_date):
        """Get weather data for a specific date, using actual API data for today/tomorrow, proxy for historical"""
        try:
            # Try to get actual weather data for today/tomorrow
            actual_data = self.get_actual_weather_for_date(target_date)
            if actual_data is not None and not actual_data.empty:
                print(
                    f"✅ Using actual weather data for {target_date.strftime('%Y-%m-%d')}")
                return actual_data

            # Fallback to proxy data for historical dates
            print(
                f"📊 Using proxy weather data for {target_date.strftime('%Y-%m-%d')}")
            timestamps = [target_date.replace(hour=h, minute=0, second=0, microsecond=0)
                          for h in range(24)]
            return self.get_historical_weather_proxy(timestamps)

        except Exception as e:
            print(f"❌ Error in get_weather_for_date: {e}")
            # Fallback to proxy data
            timestamps = [target_date.replace(hour=h, minute=0, second=0, microsecond=0)
                          for h in range(24)]
            return self.get_historical_weather_proxy(timestamps)
            return self.get_historical_weather_proxy(timestamps)

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
            # Base temperature varies by season
            base_temperature = 15 + 10 * np.sin(2 * np.pi * (month - 3) / 12)
            # Add daily temperature variation (cooler at night, warmer in afternoon)
            # Peak at hour 18 (6 PM)
            daily_temp_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = base_temperature + daily_temp_variation
            solar_factor = self._calculate_solar_factor(cloud_cover, timestamp)

            weather_data.append({
                'timestamp': timestamp,
                'cloud_cover': round(cloud_cover, 2),
                'temperature': round(temperature, 2),
                'solar_factor': round(solar_factor, 2)
            })

        return pd.DataFrame(weather_data)
