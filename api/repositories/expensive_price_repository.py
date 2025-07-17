"""
Repository for expensive price analysis.
Handles queries related to high electricity prices.
"""

import pandas as pd
from typing import Dict, Optional, Any

from .base_repository import BaseRepository
from ..config import db_manager


class ExpensivePriceRepository(BaseRepository):
    """Repository for expensive price analysis operations."""

    def __init__(self):
        self.db_manager = db_manager

    def find_all(self, threshold: float = 15.0) -> pd.DataFrame:
        """Find all expensive consumer price records above threshold (in c€/kWh)."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
                FROM electricity_prices
                WHERE consumer_price_cents_kwh > ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=[threshold])
            return df
        finally:
            conn.close()

    def find_by_id(self, record_id: Any, threshold: float = 15.0) -> Optional[pd.Series]:
        """Find expensive consumer price record by timestamp."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT * FROM electricity_prices 
                WHERE timestamp = ? AND consumer_price_cents_kwh > ?
            """
            df = pd.read_sql_query(query, conn, params=[record_id, threshold])
            return df.iloc[0] if not df.empty else None
        finally:
            conn.close()

    def count(self, threshold: float = 15.0) -> int:
        """Count expensive consumer price records."""
        conn = self.db_manager.get_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM electricity_prices WHERE consumer_price_cents_kwh > ?",
                [threshold]
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def find_expensive_price_stats(self, threshold: float = 15.0) -> Dict[str, pd.DataFrame]:
        """Find expensive consumer price analysis data."""
        conn = self.db_manager.get_connection()
        try:
            # Overall stats
            stats_query = """
                SELECT 
                    COUNT(*) as total_expensive,
                    MAX(consumer_price_cents_kwh) as highest_price,
                    AVG(consumer_price_cents_kwh) as avg_expensive_price,
                    MIN(date) as first_expensive_date,
                    MAX(date) as last_expensive_date,
                    ? as threshold_cents
                FROM electricity_prices
                WHERE consumer_price_cents_kwh > ?
            """

            # By hour
            hourly_query = """
                SELECT 
                    hour,
                    COUNT(*) as expensive_count,
                    AVG(consumer_price_cents_kwh) as avg_expensive_price
                FROM electricity_prices
                WHERE consumer_price_cents_kwh > ?
                GROUP BY hour
                ORDER BY expensive_count DESC
            """

            # By month
            monthly_query = """
                SELECT 
                    strftime('%Y-%m', date) as month,
                    COUNT(*) as expensive_count,
                    AVG(consumer_price_cents_kwh) as avg_expensive_price
                FROM electricity_prices
                WHERE consumer_price_cents_kwh > ?
                GROUP BY month
                ORDER BY month DESC
            """

            stats_df = pd.read_sql_query(
                stats_query, conn, params=[threshold, threshold])
            hourly_df = pd.read_sql_query(
                hourly_query, conn, params=[threshold])
            monthly_df = pd.read_sql_query(
                monthly_query, conn, params=[threshold])

            return {
                "overall": stats_df,
                "hourly": hourly_df,
                "monthly": monthly_df
            }
        finally:
            conn.close()

    def find_expensive_prices_by_hour(self, hour: int, threshold: float = 15.0) -> pd.DataFrame:
        """Find expensive consumer prices for a specific hour."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
                FROM electricity_prices
                WHERE consumer_price_cents_kwh > ? AND hour = ?
                ORDER BY consumer_price_cents_kwh DESC
            """

            df = pd.read_sql_query(query, conn, params=[threshold, hour])
            return df
        finally:
            conn.close()

    def find_expensive_prices_by_date_range(
        self, start_date: str, end_date: str, threshold: float = 1.500
    ) -> pd.DataFrame:
        """Find expensive consumer prices within a date range."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
                FROM electricity_prices
                WHERE consumer_price_cents_kwh > ? 
                AND date >= ? 
                AND date <= ?
                ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(query, conn, params=[
                                   threshold, start_date, end_date])
            return df
        finally:
            conn.close()

    def find_top_expensive_prices(self, limit: int = 10) -> pd.DataFrame:
        """Find the most expensive consumer price records."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
                FROM electricity_prices
                ORDER BY consumer_price_cents_kwh DESC
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=[limit])
            return df
        finally:
            conn.close()

    def get_price_percentiles(self) -> Dict[str, float]:
        """Get price percentiles to help determine expensive thresholds."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT 
                    MIN(price_eur) as min_price,
                    MAX(price_eur) as max_price,
                    AVG(price_eur) as avg_price
                FROM electricity_prices
            """

            percentile_query = """
                WITH ordered_prices AS (
                    SELECT price_eur,
                           ROW_NUMBER() OVER (ORDER BY price_eur) as row_num,
                           COUNT(*) OVER () as total_count
                    FROM electricity_prices
                )
                SELECT 
                    'p75' as percentile,
                    price_eur as value
                FROM ordered_prices 
                WHERE row_num = CAST(total_count * 0.75 AS INTEGER)
                UNION ALL
                SELECT 
                    'p90' as percentile,
                    price_eur as value
                FROM ordered_prices 
                WHERE row_num = CAST(total_count * 0.90 AS INTEGER)
                UNION ALL
                SELECT 
                    'p95' as percentile,
                    price_eur as value
                FROM ordered_prices 
                WHERE row_num = CAST(total_count * 0.95 AS INTEGER)
            """

            basic_df = pd.read_sql_query(query, conn)
            percentile_df = pd.read_sql_query(percentile_query, conn)

            result = basic_df.iloc[0].to_dict()
            for _, row in percentile_df.iterrows():
                result[row['percentile']] = row['value']

            return result
        finally:
            conn.close()
