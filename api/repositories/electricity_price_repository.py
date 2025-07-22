"""
Repository for electricity price data access.
Spring Boot-like data access layer.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base_repository import BaseRepository
from ..config import db_manager


class ElectricityPriceRepository(BaseRepository):
    """Repository for electricity price data operations."""

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

    def find_daily_stats(self, days_back: int = 30) -> pd.DataFrame:
        """Find daily statistics."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT 
                    date,
                    COUNT(*) as hours_count,
                    MIN(price_eur) as min_price,
                    MAX(price_eur) as max_price,
                    AVG(price_eur) as avg_price,
                    (MAX(price_eur) - MIN(price_eur)) as price_range,
                    COUNT(CASE WHEN price_eur < 0 THEN 1 END) as negative_hours
                FROM electricity_prices
                WHERE date >= date('now', '-{} days')
                GROUP BY date
                ORDER BY date DESC
            """.format(days_back)

            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def find_hourly_stats(self) -> pd.DataFrame:
        """Find hourly statistics across all data."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT 
                    hour,
                    COUNT(*) as total_occurrences,
                    MIN(price_eur) as min_price,
                    MAX(price_eur) as max_price,
                    AVG(price_eur) as avg_price,
                    COUNT(CASE WHEN price_eur < 0 THEN 1 END) as negative_occurrences
                FROM electricity_prices
                GROUP BY hour
                ORDER BY hour
            """

            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def find_database_stats(self) -> pd.Series:
        """Find overall database statistics."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(date) as date_start,
                    MAX(date) as date_end,
                    MIN(price_eur) as min_price,
                    MAX(price_eur) as max_price,
                    AVG(price_eur) as avg_price,
                    COUNT(CASE WHEN price_eur < 0 THEN 1 END) as negative_count
                FROM electricity_prices
            """

            df = pd.read_sql_query(query, conn)
            return df.iloc[0]
        finally:
            conn.close()

    def find_extreme_prices(self, limit: int = 10) -> Dict[str, pd.DataFrame]:
        """Find highest and lowest prices."""
        conn = self.db_manager.get_connection()
        try:
            highest_query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
                FROM electricity_prices
                ORDER BY price_eur DESC
                LIMIT ?
            """

            lowest_query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
                FROM electricity_prices
                ORDER BY price_eur ASC
                LIMIT ?
            """

            highest_df = pd.read_sql_query(highest_query, conn, params=[limit])
            lowest_df = pd.read_sql_query(lowest_query, conn, params=[limit])

            return {
                "highest": highest_df,
                "lowest": lowest_df
            }
        finally:
            conn.close()

    def find_negative_price_stats(self) -> Dict[str, pd.DataFrame]:
        """Find negative price analysis data."""
        conn = self.db_manager.get_connection()
        try:
            # Overall stats
            stats_query = """
                SELECT 
                    COUNT(*) as total_negative,
                    MIN(price_eur) as lowest_price,
                    AVG(price_eur) as avg_negative_price,
                    MIN(date) as first_negative_date,
                    MAX(date) as last_negative_date
                FROM electricity_prices
                WHERE price_eur < 0
            """

            # By hour
            hourly_query = """
                SELECT 
                    hour,
                    COUNT(*) as negative_count,
                    AVG(price_eur) as avg_negative_price
                FROM electricity_prices
                WHERE price_eur < 0
                GROUP BY hour
                ORDER BY negative_count DESC
            """

            # By month
            monthly_query = """
                SELECT 
                    strftime('%Y-%m', date) as month,
                    COUNT(*) as negative_count,
                    AVG(price_eur) as avg_negative_price
                FROM electricity_prices
                WHERE price_eur < 0
                GROUP BY month
                ORDER BY month DESC
            """

            stats_df = pd.read_sql_query(stats_query, conn)
            hourly_df = pd.read_sql_query(hourly_query, conn)
            monthly_df = pd.read_sql_query(monthly_query, conn)

            return {
                "overall": stats_df,
                "hourly": hourly_df,
                "monthly": monthly_df
            }
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

    def find_by_date(self, date: str) -> pd.DataFrame:
        """Find all price data for a specific date with weather information."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh,
                       cloud_cover, temperature, solar_factor
                FROM electricity_prices
                WHERE date = ?
                ORDER BY hour ASC
            """

            df = pd.read_sql_query(query, conn, params=[date])
            return df
        finally:
            conn.close()
