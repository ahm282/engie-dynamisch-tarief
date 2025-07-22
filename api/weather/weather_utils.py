import os
from dotenv import load_dotenv
from pathlib import Path
import pytz

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Weather API configuration
OWM_API_KEY = os.getenv('OWM_API_KEY')

# Fallback: if still None, try to load from parent directory
if not OWM_API_KEY:
    load_dotenv()
    OWM_API_KEY = os.getenv('OWM_API_KEY')

# Debug: Print whether API key is loaded (first 10 chars only)
if OWM_API_KEY:
    print(f"✅ Weather API key loaded: {OWM_API_KEY[:10]}...")
else:
    print("❌ Weather API key not found - check .env file")
GENT_COORDS = {
    'lat': 51.0543,  # Gent coordinates as central point
    'lon': 3.7174
}

# Belgium timezone
BELGIUM_TIMEZONE = 'Europe/Brussels'

# Weather features for prediction
WEATHER_FEATURES = [
    'cloud_cover',
    'solar_irradiance',
    'temperature'
]


def round_weather_values(weather_dict):
    """
    Round all weather values to 2 decimal places.

    Args:
        weather_dict: Dictionary containing weather data

    Returns:
        Dictionary with rounded values
    """
    rounded_dict = weather_dict.copy()

    # Round numerical weather values to 2 decimal places
    for key in ['cloud_cover', 'temperature', 'solar_factor', 'solar_irradiance']:
        if key in rounded_dict and isinstance(rounded_dict[key], (int, float)):
            rounded_dict[key] = round(rounded_dict[key], 2)

    return rounded_dict


def convert_to_belgium_timezone(utc_datetime):
    """
    Convert UTC datetime to Belgium timezone.

    Args:
        utc_datetime: UTC datetime object

    Returns:
        Datetime in Belgium timezone
    """
    belgium_tz = pytz.timezone(BELGIUM_TIMEZONE)
    if utc_datetime.tzinfo is None:
        utc_datetime = pytz.utc.localize(utc_datetime)
    return utc_datetime.astimezone(belgium_tz)


def is_valid_forecast_time(timestamp, max_days_ahead=5):
    """
    Check if a timestamp is within valid forecast range.

    Args:
        timestamp: Timestamp to validate
        max_days_ahead: Maximum days ahead for valid forecast

    Returns:
        Boolean indicating if timestamp is valid for forecast
    """
    from datetime import datetime, timezone
    import pandas as pd

    current_time = datetime.now(timezone.utc)
    target_time = pd.to_datetime(timestamp, utc=True)

    return current_time < target_time <= current_time + pd.Timedelta(days=max_days_ahead)
