from functools import wraps
from datetime import datetime


def cache_forecast(ttl_seconds=3600):
    cache = {}

    def decorator(func):
        @wraps(func)
        def wrapper(self, hours_ahead):
            now = datetime.utcnow()
            if hours_ahead in cache:
                timestamp, data = cache[hours_ahead]
                if (now - timestamp).total_seconds() < ttl_seconds:
                    return data
            result = func(self, hours_ahead)
            cache[hours_ahead] = (now, result)
            return result
        return wrapper

    return decorator


@staticmethod
def clear_cache():
    from ..utils.cache_utils import cache_forecast
    cache_forecast.cache.clear()
