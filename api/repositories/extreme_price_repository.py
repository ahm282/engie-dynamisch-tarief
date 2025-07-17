"""
Repository for extreme price analysis.
Handles finding highest and lowest prices.
"""

import pandas as pd
from typing import Dict, Optional, Any

from .base_repository import BaseRepository
from ..database import db_manager


class ExtremePriceRepository(BaseRepository):
    """Repository for extreme price analysis operations."""

    def __init__(self):
        self.db_manager = db_manager

    def find_all(self) -> pd.DataFrame:
        """Find all records - implemented for base class compliance."""
        return self.find_extreme_prices()["highest"]

    def find_by_id(self, record_id: Any) -> Optional[pd.Series]:
        """Find by ID - implemented for base class compliance."""
        # Return the most extreme price record
        extremes = self.find_extreme_prices(limit=1)
        highest = extremes["highest"]
        return highest.iloc[0] if not highest.empty else None

    def count(self) -> int:
        """Count total records."""
        conn = self.db_manager.get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM electricity_prices")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def find_extreme_prices(self, limit: int = 10) -> Dict[str, pd.DataFrame]:
        """Find highest and lowest prices."""
        conn = self.db_manager.get_connection()
        try:
            highest_query = """
                SELECT timestamp, date, hour, price_eur, price_raw
                FROM electricity_prices
                ORDER BY price_eur DESC
                LIMIT ?
            """

            lowest_query = """
                SELECT timestamp, date, hour, price_eur, price_raw
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

    def find_highest_prices(self, limit: int = 10) -> pd.DataFrame:
        """Find only the highest prices."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw
                FROM electricity_prices
                ORDER BY price_eur DESC
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=[limit])
            return df
        finally:
            conn.close()

    def find_lowest_prices(self, limit: int = 10) -> pd.DataFrame:
        """Find only the lowest prices."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT timestamp, date, hour, price_eur, price_raw
                FROM electricity_prices
                ORDER BY price_eur ASC
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=[limit])
            return df
        finally:
            conn.close()
