"""
Response models for API endpoints.
"""

from pydantic import BaseModel
from typing import List
from .price_models import PriceRecord


class CurrentPricesResponse(BaseModel):
    """Model for current prices response."""
    current_prices: List[PriceRecord]
    count: int


class APIInfo(BaseModel):
    """Model for API information."""
    message: str
    version: str
    endpoints: dict


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    service: str
