"""
Repository for statistical analysis of electricity price data.
Handles aggregations and statistical calculations.
"""

import pandas as pd

from .base_repository import BaseRepository
from ..database import db_manager


class StatisticsRepository(BaseRepository):
    """Repository for statistical analysis operations."""

    def __init__(self):
        self.db_manager = db_manager

    def find_all(self) -> pd.DataFrame:
        """Find all records - implemented for base class compliance."""
        return self.find_database_stats_as_dataframe()

    def find_by_id(self, record_id) -> pd.Series:
        """Find by ID - implemented for base class compliance."""
        return self.find_database_stats()

    def count(self) -> int:
        """Count total records."""
        conn = self.db_manager.get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM electricity_prices")
            return cursor.fetchone()[0]
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

    def find_database_stats_as_dataframe(self) -> pd.DataFrame:
        """Find overall database statistics as DataFrame."""
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
