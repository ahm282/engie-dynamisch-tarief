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

    @staticmethod
    def categorize_price(consumer_price_cents_kwh: float) -> str:
        """
        Categorize consumer price into cheap, regular, expensive, or extremely expensive.

        Thresholds based on statistical analysis of consumer pricing:
        - Cheap: < 7.5 c€/kWh (below average, including negative consumer prices)
        - Regular: 7.5 - 13.0 c€/kWh (around average of ~10.3 c€/kWh)
        - Expensive: 13.0 - 20.0 c€/kWh (above average) 
        - Extremely Expensive: > 20.0 c€/kWh (very high, top 5% of prices)

        Args:
            consumer_price_cents_kwh: Consumer price in euro cents per kWh

        Returns:
            str: Category as "cheap", "regular", "expensive", or "extremely_expensive"
        """
        if consumer_price_cents_kwh < 7.5:
            return "cheap"
        elif consumer_price_cents_kwh < 13.0:
            return "regular"
        elif consumer_price_cents_kwh < 20.0:
            return "expensive"
        else:
            return "extremely_expensive"

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

            # Check if DataFrame is empty
            if df.empty:
                return []

            # Verify required columns exist
            required_columns = ['timestamp', 'date', 'hour',
                                'price_eur', 'price_raw', 'consumer_price_cents_kwh']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required columns in data: {missing_columns}"
                )

            # Convert to Pydantic models with categorization
            records = []
            for _, row in df.iterrows():
                consumer_price = float(row['consumer_price_cents_kwh'])
                category = self.categorize_price(consumer_price)

                records.append(PriceRecord(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw'],
                    consumer_price_cents_kwh=round(consumer_price, 3),
                    price_category=category
                ))

            return records

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving price data")

    def get_current_prices(self, hours: int = 24) -> dict:
        """Get the most recent price data with categorization."""
        try:
            if hours < 1 or hours > 168:  # Max 1 week
                raise HTTPException(
                    status_code=400,
                    detail="Hours must be between 1 and 168 (1 week)"
                )

            df = self.repository.find_current_prices(hours=hours)

            # Check if DataFrame is empty or missing required columns
            if df.empty:
                return {
                    "current_prices": [],
                    "count": 0,
                    "category_distribution": {},
                    "hours_analyzed": hours
                }

            # Verify required columns exist
            required_columns = ['timestamp', 'date', 'hour',
                                'price_eur', 'price_raw', 'consumer_price_cents_kwh']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required columns in data: {missing_columns}"
                )

            # Convert to Pydantic models with categorization
            records = []
            for _, row in df.iterrows():
                consumer_price = float(row['consumer_price_cents_kwh'])
                category = self.categorize_price(consumer_price)

                records.append(PriceRecord(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw'],
                    consumer_price_cents_kwh=round(consumer_price, 3),
                    price_category=category
                ))

            # Calculate category distribution
            category_counts = {}
            for record in records:
                category = record.price_category
                category_counts[category] = category_counts.get(
                    category, 0) + 1

            return {
                "current_prices": records,
                "count": len(records),
                "category_distribution": category_counts,
                "hours_analyzed": hours
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
            self.validate_input(hour=hour, order=order)

            df = self.repository.find_all_with_filters(
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                hour=hour,
                order=order
            )

            # Check if DataFrame is empty
            if df.empty:
                return []

            # Verify required columns exist
            required_columns = ['timestamp', 'date',
                                'hour', 'price_eur', 'price_raw', 'consumer_price_cents_kwh']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required columns in data: {missing_columns}"
                )

            # Convert to Pydantic models with categorization
            records = []
            for _, row in df.iterrows():
                consumer_price = float(row['consumer_price_cents_kwh'])
                category = self.categorize_price(consumer_price)

                records.append(PriceRecord(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    price_eur=round(float(row['price_eur']), 3),
                    price_raw=row['price_raw'],
                    consumer_price_cents_kwh=round(consumer_price, 3),
                    price_category=category
                ))

            return records

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving all price data")
