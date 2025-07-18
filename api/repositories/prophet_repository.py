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
