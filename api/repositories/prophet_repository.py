"""
Repository to load time series data for forecasting.
"""

import pandas as pd
from ..config import db_manager


class ProphetRepository:
    """Handles time series data extraction for Prophet."""

    def __init__(self):
        self.db_manager = db_manager

    def get_all_data(self) -> pd.DataFrame:
        """Fetch complete historical time series data."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT
                    timestamp,
                    consumer_price_cents_kwh
                FROM electricity_prices
                WHERE consumer_price_cents_kwh IS NOT NULL
                ORDER BY timestamp ASC
            """
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def get_latest_timestamp(self) -> str:
        """Get the latest timestamp from the database."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT MAX(timestamp) as latest_timestamp
                FROM electricity_prices
                WHERE consumer_price_cents_kwh IS NOT NULL
            """
            result = pd.read_sql_query(query, conn)
            return result.iloc[0]['latest_timestamp'] if not result.empty else None
        finally:
            conn.close()

    def get_data_summary(self) -> dict:
        """Get summary information about the available data."""
        conn = self.db_manager.get_connection()
        try:
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as latest_timestamp,
                    COUNT(DISTINCT date) as days_available
                FROM electricity_prices
                WHERE consumer_price_cents_kwh IS NOT NULL
            """
            result = pd.read_sql_query(query, conn)
            return result.iloc[0].to_dict() if not result.empty else {}
        finally:
            conn.close()
