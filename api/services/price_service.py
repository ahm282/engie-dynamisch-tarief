"""
Service for electricity price data operations.
"""

from typing import List, Optional
from fastapi import HTTPException

from .base_service import BaseService
from ..repositories import PriceDataRepository
from ..models import PriceRecord


class PriceService(BaseService):
    """Service for price data operations."""

    def __init__(self, repository: PriceDataRepository = None):
        """Initialize service with repository dependency injection."""
        super().__init__(repository or PriceDataRepository())

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for price queries."""
        hour = kwargs.get('hour')
        limit = kwargs.get('limit')
        order = kwargs.get('order', 'desc')

        if hour is not None and (hour < 0 or hour > 23):
            raise HTTPException(
                status_code=400, detail="Hour must be between 0 and 23")

        if limit and limit > 10000:
            raise HTTPException(
                status_code=400, detail="Limit cannot exceed 10000 records")

        if order and order.lower() not in ["asc", "desc"]:
            raise HTTPException(
                status_code=400, detail="Order must be 'asc' or 'desc'")

        return True

    def get_prices(
        self,
        start_date: str = None,
        end_date: str = None,
        days_back: int = None,
        hour: int = None,
        limit: int = None,
        order: str = "desc"
    ) -> List[PriceRecord]:
        """Get electricity price data with various filters."""
        try:
            # Validate input
            self.validate_input(hour=hour, limit=limit, order=order)

            df = self.repository.find_with_filters(
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                hour=hour,
                limit=limit,
                order=order
            )

            # Convert to Pydantic models
            records = []
            for _, row in df.iterrows():
                records.append(PriceRecord(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw']
                ))

            return records

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving price data")

    def get_current_prices(self, hours: int = 24) -> dict:
        """Get the most recent price data."""
        try:
            if hours < 1 or hours > 168:  # Max 1 week
                raise HTTPException(
                    status_code=400,
                    detail="Hours must be between 1 and 168 (1 week)"
                )

            df = self.repository.find_current_prices(hours=hours)

            # Convert to Pydantic models
            records = []
            for _, row in df.iterrows():
                records.append(PriceRecord(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw']
                ))

            return {
                "current_prices": records,
                "count": len(records)
            }

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving current prices")

    def get_all_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
        hour: Optional[int] = None,
        order: str = "desc"
    ) -> List[PriceRecord]:
        """Get all electricity price data with filters - no limit applied."""
        try:
            # Validate input
            self.validate_input(
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                hour=hour,
                order=order
            )

            df = self.repository.find_all_with_filters(
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                hour=hour,
                order=order
            )

            # Convert to Pydantic models
            records = []
            for _, row in df.iterrows():
                records.append(PriceRecord(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw']
                ))

            return records

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving all price data")
