from pydantic import BaseModel
from typing import List
from datetime import datetime
import pandas as pd


class ForecastPoint(BaseModel):
    timestamp: datetime
    date: str
    hour: int
    predicted_price_cents_kwh: float
    lower_bound_cents_kwh: float
    upper_bound_cents_kwh: float
    confidence: float
    price_category: str


class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]

    @staticmethod
    def from_list(data: list) -> 'ForecastResponse':
        return ForecastResponse(forecast=[ForecastPoint(**item) for item in data])
