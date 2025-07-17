"""
Service for extreme price analysis operations.
"""

from typing import List, Dict
from fastapi import HTTPException

from .base_service import BaseService
from ..repositories import ExtremePriceRepository
from ..models import ExtremePrice


class ExtremeAnalysisService(BaseService):
    """Service for extreme price analysis operations."""

    def __init__(self, repository: ExtremePriceRepository = None):
        """Initialize service with repository dependency injection."""
        super().__init__(repository or ExtremePriceRepository())

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for extreme analysis queries."""
        limit = kwargs.get('limit', 10)

        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 100"
            )

        return True

    def get_extreme_prices(self, limit: int = 10) -> Dict[str, List[ExtremePrice]]:
        """Get highest and lowest prices."""
        try:
            # Validate input
            self.validate_input(limit=limit)

            result = self.repository.find_extreme_prices(limit=limit)
            highest_df = result["highest"]
            lowest_df = result["lowest"]

            # Convert to Pydantic models
            highest_prices = []
            for _, row in highest_df.iterrows():
                highest_prices.append(ExtremePrice(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw'],
                    consumer_price_cents_kwh=round(
                        float(row['consumer_price_cents_kwh']), 3),
                    price_type="highest"
                ))

            lowest_prices = []
            for _, row in lowest_df.iterrows():
                lowest_prices.append(ExtremePrice(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw'],
                    consumer_price_cents_kwh=round(
                        float(row['consumer_price_cents_kwh']), 3),
                    price_type="lowest"
                ))

            return {
                "highest_prices": highest_prices,
                "lowest_prices": lowest_prices
            }

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving extreme prices")
