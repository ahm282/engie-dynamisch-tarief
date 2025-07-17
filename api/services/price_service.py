"""
Service for electricity price data operations.
"""

from typing import List, Optional
from fastapi import HTTPException
import random
import numpy as np
from datetime import datetime, timedelta

from .base_service import BaseService
from ..repositories import PriceDataRepository
from ..models import PriceRecord, ConsumptionCostAnalysis, HourlyConsumption


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

    def get_today_prices(self) -> dict:
        """Get today's electricity prices with categorization."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            df = self.repository.find_by_date(today)

            # Check if DataFrame is empty or missing required columns
            if df.empty:
                return {
                    "today_prices": [],
                    "count": 0,
                    "category_distribution": {},
                    "date": today
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
                "today_prices": records,
                "count": len(records),
                "category_distribution": category_counts,
                "date": today
            }

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving today's prices")

    def get_next_day_prices(self) -> dict:
        """Get next day's (tomorrow's) electricity prices with categorization."""
        try:
            tomorrow = (datetime.now() + timedelta(days=1)
                        ).strftime('%Y-%m-%d')
            df = self.repository.find_by_date(tomorrow)

            # Check if DataFrame is empty or missing required columns
            if df.empty:
                return {
                    "next_day_prices": [],
                    "count": 0,
                    "category_distribution": {},
                    "date": tomorrow,
                    "available": False,
                    "message": f"Price data for {tomorrow} is not yet available"
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
                "next_day_prices": records,
                "count": len(records),
                "category_distribution": category_counts,
                "date": tomorrow,
                "available": True,
                "message": f"Price data for {tomorrow} is available"
            }

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error retrieving next day's prices")

    def get_prices_by_date(self, date: str) -> dict:
        """Get electricity prices for a specific date with categorization."""
        try:
            # Validate date format
            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD format."
                )

            df = self.repository.find_by_date(date)

            # Check if DataFrame is empty or missing required columns
            if df.empty:
                return {
                    "date_prices": [],
                    "count": 0,
                    "category_distribution": {},
                    "date": date,
                    "available": False,
                    "message": f"No price data available for {date}"
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

            # Calculate statistics
            consumer_prices = [
                record.consumer_price_cents_kwh for record in records]
            wholesale_prices = [record.price_eur for record in records]

            return {
                "date_prices": records,
                "count": len(records),
                "category_distribution": category_counts,
                "date": date,
                "available": True,
                "message": f"Price data for {date} is available",
                "statistics": {
                    "consumer_price_avg": round(sum(consumer_prices) / len(consumer_prices), 3) if consumer_prices else 0,
                    "consumer_price_min": round(min(consumer_prices), 3) if consumer_prices else 0,
                    "consumer_price_max": round(max(consumer_prices), 3) if consumer_prices else 0,
                    "wholesale_price_avg": round(sum(wholesale_prices) / len(wholesale_prices), 3) if wholesale_prices else 0,
                    "wholesale_price_min": round(min(wholesale_prices), 3) if wholesale_prices else 0,
                    "wholesale_price_max": round(max(wholesale_prices), 3) if wholesale_prices else 0
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(
                e, f"Error retrieving prices for date {date}")

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

    def calculate_consumption_cost(
        self,
        total_consumption_kwh: float,
        days_back: int = 30
    ) -> ConsumptionCostAnalysis:
        """
        Calculate electricity costs based on consumption with realistic distribution patterns.

        This method distributes the total consumption across the specified time period
        using realistic consumption patterns that simulate real-life usage:
        - Higher consumption during morning (7-9h) and evening (18-22h) peaks
        - Lower consumption during night hours (0-6h)
        - Weekend vs weekday variations
        - Random variations to simulate real usage patterns

        Args:
            total_consumption_kwh: Total electricity consumption in kWh
            days_back: Number of days to analyze (default: 30, max: 365)

        Returns:
            ConsumptionCostAnalysis: Detailed cost analysis with hourly breakdown
        """
        try:
            # Validate input
            if total_consumption_kwh <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Total consumption must be greater than 0 kWh"
                )

            if days_back < 1 or days_back > 365:
                raise HTTPException(
                    status_code=400,
                    detail="Days back must be between 1 and 365"
                )

            # Get price data for the specified period
            df = self.repository.find_current_prices(hours=days_back * 24)

            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail="No price data available for the specified period"
                )

            # Ensure we have the required columns
            required_columns = ['timestamp', 'date', 'hour',
                                'price_eur', 'price_raw', 'consumer_price_cents_kwh']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required columns in data: {missing_columns}"
                )

            # Create realistic consumption distribution
            consumption_distribution = self._generate_realistic_consumption_pattern(
                len(df), total_consumption_kwh)

            # Calculate costs for each hour
            hourly_breakdown = []
            total_cost = 0.0
            cost_by_category = {"cheap": 0.0, "regular": 0.0,
                                "expensive": 0.0, "extremely_expensive": 0.0}
            consumption_by_category = {
                "cheap": 0.0, "regular": 0.0, "expensive": 0.0, "extremely_expensive": 0.0}

            for i, (_, row) in enumerate(df.iterrows()):
                consumption_kwh = consumption_distribution[i]
                price_cents_kwh = float(row['consumer_price_cents_kwh'])
                cost_euros = (consumption_kwh * price_cents_kwh) / \
                    100  # Convert cents to euros
                category = self.categorize_price(price_cents_kwh)

                hourly_consumption = HourlyConsumption(
                    timestamp=row['timestamp'],
                    date=row['date'],
                    hour=int(row['hour']),
                    consumption_kwh=round(consumption_kwh, 3),
                    price_cents_kwh=round(price_cents_kwh, 3),
                    cost_euros=round(cost_euros, 4),
                    price_category=category
                )

                hourly_breakdown.append(hourly_consumption)
                total_cost += cost_euros
                cost_by_category[category] += cost_euros
                consumption_by_category[category] += consumption_kwh

            # Calculate statistics
            prices = [float(row['consumer_price_cents_kwh'])
                      for _, row in df.iterrows()]
            average_price = total_cost * 100 / \
                total_consumption_kwh  # Convert back to cents/kWh

            # Calculate savings analysis
            cheapest_price = min(prices)
            most_expensive_price = max(prices)
            cost_at_cheapest = (total_consumption_kwh * cheapest_price) / 100
            cost_at_most_expensive = (
                total_consumption_kwh * most_expensive_price) / 100

            savings_analysis = {
                "potential_savings_vs_most_expensive": round(cost_at_most_expensive - total_cost, 2),
                "extra_cost_vs_cheapest": round(total_cost - cost_at_cheapest, 2),
                "cheapest_possible_cost": round(cost_at_cheapest, 2),
                "most_expensive_possible_cost": round(cost_at_most_expensive, 2)
            }

            # Additional statistics
            statistics = {
                "min_hourly_price": round(min(prices), 3),
                "max_hourly_price": round(max(prices), 3),
                "price_volatility": round(np.std(prices), 3),
                "median_price": round(np.median(prices), 3),
                "cost_efficiency_score": round((cost_at_most_expensive - total_cost) / (cost_at_most_expensive - cost_at_cheapest) * 100, 1)
            }

            return ConsumptionCostAnalysis(
                total_consumption_kwh=round(total_consumption_kwh, 3),
                total_cost_euros=round(total_cost, 2),
                average_price_cents_kwh=round(average_price, 3),
                days_analyzed=days_back,
                period_start=df.iloc[-1]['timestamp'],  # Oldest record
                period_end=df.iloc[0]['timestamp'],     # Newest record
                hourly_breakdown=hourly_breakdown,
                cost_by_category={k: round(v, 2)
                                  for k, v in cost_by_category.items()},
                consumption_by_category={
                    k: round(v, 3) for k, v in consumption_by_category.items()},
                savings_analysis=savings_analysis,
                statistics=statistics
            )

        except HTTPException:
            raise
        except Exception as e:
            self.handle_exception(e, "Error calculating consumption costs")

    def _generate_realistic_consumption_pattern(self, num_hours: int, total_consumption: float) -> List[float]:
        """
        Generate realistic electricity consumption pattern based on typical household usage.

        This creates a realistic consumption distribution with:
        - Morning peak (7-9h): Higher consumption
        - Evening peak (18-22h): Highest consumption  
        - Night valley (0-6h): Lower consumption
        - Weekend variations: Different patterns on weekends
        - Random variations: ±20% to simulate real-life unpredictability

        Args:
            num_hours: Number of hours to distribute consumption across
            total_consumption: Total consumption to distribute

        Returns:
            List[float]: Hourly consumption values in kWh
        """
        # Define base consumption pattern by hour (0-23)
        # Values represent relative consumption intensity
        base_pattern = {
            0: 0.3,   # Night - very low
            1: 0.25,  # Night - very low
            2: 0.2,   # Night - lowest
            3: 0.2,   # Night - lowest
            4: 0.25,  # Night - very low
            5: 0.3,   # Early morning - low
            6: 0.5,   # Morning start - medium
            7: 0.8,   # Morning peak start - high
            8: 1.0,   # Morning peak - very high
            9: 0.7,   # Post morning - medium-high
            10: 0.5,  # Mid morning - medium
            11: 0.6,  # Late morning - medium
            12: 0.7,  # Lunch - medium-high
            13: 0.6,  # Afternoon - medium
            14: 0.5,  # Afternoon - medium
            15: 0.6,  # Late afternoon - medium
            16: 0.7,  # Early evening - medium-high
            17: 0.9,  # Evening start - high
            18: 1.2,  # Evening peak start - very high
            19: 1.3,  # Evening peak - highest
            20: 1.2,  # Evening peak - very high
            21: 1.0,  # Late evening - high
            22: 0.8,  # Night transition - medium-high
            23: 0.5   # Late night - medium
        }

        consumption_pattern = []

        for hour_index in range(num_hours):
            # Determine the hour of day and day of week
            hour_of_day = hour_index % 24
            day_index = hour_index // 24
            is_weekend = (day_index % 7) in [5, 6]  # Saturday and Sunday

            # Get base consumption for this hour
            base_consumption = base_pattern[hour_of_day]

            # Apply weekend modification (generally lower consumption during work hours)
            if is_weekend:
                if 9 <= hour_of_day <= 17:  # Work hours - less difference on weekends
                    base_consumption *= 0.8
                else:  # Non-work hours - similar or slightly higher on weekends
                    base_consumption *= 1.1

            # Add random variation (±20%)
            variation = random.uniform(0.8, 1.2)
            final_consumption = base_consumption * variation

            consumption_pattern.append(final_consumption)

        # Normalize to match total consumption
        total_pattern = sum(consumption_pattern)
        normalized_pattern = [(x / total_pattern) *
                              total_consumption for x in consumption_pattern]

        return normalized_pattern
