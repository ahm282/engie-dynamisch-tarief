"""
Repository for negative price analysis.
Handles queries related to negative electricity prices.
"""

import pandas as pd
from typing import Dict, Optional, Any

from .base_repository import BaseRepository
from ..database import db_manager


class NegativePriceRepository(BaseRepository):
    """Repository for negative price analysis operations."""

    def __init__(self):
        self.db_manager = db_manager

    def find_all(self) -> pd.DataFrame:
        """Find all negative price records."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw
                FROM electricity_prices
                WHERE price_eur < 0
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def find_by_id(self, record_id: Any) -> Optional[pd.Series]:
        """Find negative price record by timestamp."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT * FROM electricity_prices 
                WHERE timestamp = ? AND price_eur < 0
            """
            df = pd.read_sql_query(query, conn, params=[record_id])
            return df.iloc[0] if not df.empty else None
        finally:
            conn.close()

    def count(self) -> int:
        """Count negative price records."""
        conn = self.db_manager.get_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM electricity_prices WHERE price_eur < 0"
            )
            return cursor.fetchone()[0]
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

    def find_negative_prices_by_hour(self, hour: int) -> pd.DataFrame:
        """Find negative prices for a specific hour."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw
                FROM electricity_prices
                WHERE price_eur < 0 AND hour = ?
                ORDER BY price_eur ASC
            """

            df = pd.read_sql_query(query, conn, params=[hour])
            return df
        finally:
            conn.close()

    def find_negative_prices_by_date_range(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Find negative prices within a date range."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw
                FROM electricity_prices
                WHERE price_eur < 0 
                AND date >= ? 
                AND date <= ?
                ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            return df
        finally:
            conn.close()
