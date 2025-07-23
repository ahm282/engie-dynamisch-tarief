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

        Uses accurate astronomical sunrise/sunset calculations for Belgium.

        Args:
            cloud_cover: Cloud coverage percentage (0-100)
            timestamp: DateTime object to determine sun position (optional)

        Returns:
            Solar factor between 0.0 and 1.0
        """
        if timestamp is None:
            # If no timestamp provided, use current time (for backward compatibility)
            timestamp = datetime.now(timezone.utc)

        # Convert to local time for Belgium (UTC+1 in winter, UTC+2 in summer)
        month = timestamp.month
        if month >= 4 and month <= 9:  # Summer time (UTC+2)
            local_hour = (timestamp.hour + 2) % 24
            timezone_offset = 2
        else:  # Winter time (UTC+1)
            local_hour = (timestamp.hour + 1) % 24
            timezone_offset = 1

        # Accurate astronomical sunrise/sunset calculation for Belgium (50.8°N, 4.4°E)
        # Using proper Julian day and solar position calculations

        # Convert to Julian day number
        a = (14 - timestamp.month) // 12
        y = timestamp.year - a
        m = timestamp.month + 12 * a - 3
        julian_day = timestamp.day + \
            (153 * m + 2) // 5 + 365 * y + \
            y // 4 - y // 100 + y // 400 + 1721119

        # Days since J2000.0
        n = julian_day - 2451545.0

        # Mean solar longitude
        L = (280.460 + 0.9856474 * n) % 360

        # Mean anomaly
        g = np.radians((357.528 + 0.9856003 * n) % 360)

        # Ecliptic longitude
        lambda_sun = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))

        # Solar declination
        declination = np.arcsin(
            np.sin(np.radians(23.439)) * np.sin(lambda_sun))

        # Belgium coordinates
        latitude = 50.8
        longitude = 4.4

        # Equation of time correction (longitude difference from standard meridian)
        # Standard meridian is always 15°E for Central European Time, regardless of DST
        equation_of_time_minutes = 4 * (longitude - 15)

        # Hour angle at sunrise/sunset
        lat_rad = np.radians(latitude)
        cos_hour_angle = -np.tan(lat_rad) * np.tan(declination)

        if cos_hour_angle > 1:
            # Polar night (not applicable to Belgium)
            sunrise_hour = 12
            sunset_hour = 12
        elif cos_hour_angle < -1:
            # Polar day (not applicable to Belgium)
            sunrise_hour = 0
            sunset_hour = 24
        else:
            hour_angle = np.arccos(cos_hour_angle)
            hour_angle_degrees = np.degrees(hour_angle)

            # Calculate sunrise and sunset in local solar time
            sunrise_solar = 12 - hour_angle_degrees / 15 + equation_of_time_minutes / 60
            sunset_solar = 12 + hour_angle_degrees / 15 + equation_of_time_minutes / 60

            # Convert to local clock time by adding timezone offset
            sunrise_hour = sunrise_solar + timezone_offset
            sunset_hour = sunset_solar + timezone_offset

        # Apply safety bounds for Belgium
        sunrise_hour = max(2.0, min(10.0, sunrise_hour))
        sunset_hour = max(15.0, min(23.5, sunset_hour))

        # No solar generation during night hours
        if local_hour < sunrise_hour or local_hour > sunset_hour:
            return 0.0

        # Calculate solar efficiency based on sun elevation angle
        daylight_hours = sunset_hour - sunrise_hour
        solar_noon = (sunrise_hour + sunset_hour) / 2
        hour_from_noon = abs(local_hour - solar_noon)

        # Sun elevation angle approximation
        # Maximum elevation at solar noon, decreasing towards sunrise/sunset
        max_hour_angle = daylight_hours / 2
        if max_hour_angle > 0:
            relative_position = hour_from_noon / \
                max_hour_angle  # 0 at noon, 1 at sunrise/sunset
        else:
            relative_position = 0

        # Solar elevation efficiency (cosine relationship)
        sun_elevation_factor = np.cos(np.radians(90 * relative_position))

        # Apply minimum efficiency threshold (atmospheric scattering, etc.)
        sun_efficiency = max(0.1, sun_elevation_factor)

        # Seasonal solar intensity adjustment based on solar declination
        # Maximum at summer solstice, minimum at winter solstice
        seasonal_factor = 0.85 + 0.15 * np.sin(declination)
        sun_efficiency *= seasonal_factor

        # Cloud reduction factor
        # Clear sky = 1.0, completely overcast = 0.05
        # Even on overcast days, some diffuse light gets through
        cloud_efficiency = max(0.05, (100 - cloud_cover) / 100)

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
