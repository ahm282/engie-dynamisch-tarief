"""
Repository for basic electricity price data operations.
Handles CRUD operations and filtered queries.
"""

import pandas as pd
from typing import Optional, Any

from .base_repository import BaseRepository
from ..config import db_manager


class PriceDataRepository(BaseRepository):
    """Repository for basic price data operations."""

    def __init__(self):
        self.db_manager = db_manager

    def find_all(self) -> pd.DataFrame:
        """Find all electricity price records."""
        conn = self.db_manager.get_connection()
        try:
            query = "SELECT * FROM electricity_prices ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def find_by_id(self, record_id: Any) -> Optional[pd.Series]:
        """Find price record by timestamp."""
        conn = self.db_manager.get_connection()
        try:
            query = "SELECT * FROM electricity_prices WHERE timestamp = ?"
            df = pd.read_sql_query(query, conn, params=[record_id])
            return df.iloc[0] if not df.empty else None
        finally:
            conn.close()

    def count(self) -> int:
        """Count total price records."""
        conn = self.db_manager.get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM electricity_prices")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def find_with_filters(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
        hour: Optional[int] = None,
        limit: int = None,
        order: str = "desc"
    ) -> pd.DataFrame:
        """Find price records with various filters including weather data."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh,
                       cloud_cover, temperature, solar_factor
                FROM electricity_prices
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            if days_back:
                query += " AND date >= date('now', '-{} days')".format(days_back)

            if hour is not None:
                query += " AND hour = ?"
                params.append(hour)

            order_clause = "ASC" if order.lower() == "asc" else "DESC"
            query += f" ORDER BY timestamp {order_clause}"

            # Apply default limit of 1000 if no limit specified
            if limit is None:
                limit = 1000

            if limit > 0:
                query += " LIMIT ?"
                params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def find_all_with_filters(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
        hour: Optional[int] = None,
        order: str = "desc"
    ) -> pd.DataFrame:
        """Find all price records with filters including weather data - no limit applied."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh,
                       cloud_cover, temperature, solar_factor
                FROM electricity_prices
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            if days_back:
                query += " AND date >= date('now', '-{} days')".format(days_back)

            if hour is not None:
                query += " AND hour = ?"
                params.append(hour)

            order_clause = "ASC" if order.lower() == "asc" else "DESC"
            query += f" ORDER BY timestamp {order_clause}"

            # No limit applied - returns all matching records
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def find_current_prices(self, hours: int = 24) -> pd.DataFrame:
        """Find most recent price data with weather information."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh,
                       cloud_cover, temperature, solar_factor
                FROM electricity_prices
                ORDER BY timestamp DESC
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=[hours])
            return df
        finally:
            conn.close()
