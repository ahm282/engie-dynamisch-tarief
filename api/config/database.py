"""
Database Configuration and Management Module

This module provides centralized database connection and query management for the
Electricity Price Analysis API. It implements the Repository pattern with a
singleton DatabaseManager for consistent data access across all services.

Tags:
    - database
    - configuration
    - sqlite
    - data-access
    - repository-pattern

Features:
    - Singleton database manager for consistent connections
    - Automatic connection cleanup and transaction handling
    - Type-safe parameter binding for SQL injection prevention
    - Pandas DataFrame integration for data analysis
    - Comprehensive error handling and logging support

Usage:
    The db_manager instance is automatically available for import:
    
    ```python
    from api.config import db_manager
    
    # Query data for analysis
    df = db_manager.execute_query("SELECT * FROM electricity_prices LIMIT 100")
    
    # Get aggregated statistics
    total_records = db_manager.execute_scalar("SELECT COUNT(*) FROM electricity_prices")
    
    # Update processed status
    updated = db_manager.execute_update("UPDATE electricity_prices SET processed = 1")
    ```

Database Schema:
    - Table: electricity_prices
    - Columns: timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh
    - Indexes: ON timestamp, date, hour for query performance
    
Author: Electricity Price Analysis System
Version: 2.0.0
"""

import sqlite3
import pandas as pd
from typing import Any, List, Optional, Union
from .settings import app_config


class DatabaseManager:
    """
    Centralized database connection and query management for electricity price data.

    This class provides a unified interface for all database operations including:
    - Connection management with automatic cleanup
    - Query execution with parameter binding
    - Data retrieval as pandas DataFrames
    - Scalar value queries for aggregations
    - Update operations with transaction handling

    Features:
    - SQLite row factory for column access by name
    - Automatic connection cleanup with context management
    - Type-safe parameter binding
    - Pandas integration for data analysis

    Examples:
        >>> db = DatabaseManager()
        >>> df = db.execute_query("SELECT * FROM electricity_prices LIMIT 5")
        >>> count = db.execute_scalar("SELECT COUNT(*) FROM electricity_prices")
        >>> rows_affected = db.execute_update("UPDATE electricity_prices SET processed = 1")
    """

    def __init__(self):
        """
        Initialize the DatabaseManager with configuration settings.

        The database path is loaded from application configuration.
        """
        self.database_path = app_config.database_path

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory enabled.

        Returns:
            sqlite3.Connection: Database connection with row factory for column access by name.

        Note:
            The connection should be closed after use. Consider using the higher-level
            execute_* methods which handle connection cleanup automatically.

        Example:
            >>> conn = db.get_connection()
            >>> try:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT COUNT(*) FROM electricity_prices")
            ...     result = cursor.fetchone()
            ... finally:
            ...     conn.close()
        """
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as a pandas DataFrame.

        This method is ideal for data analysis and reporting operations.
        Automatically handles connection management and cleanup.

        Args:
            query (str): SQL SELECT query to execute. Use ? placeholders for parameters.
            params (Optional[List[Any]]): List of parameters to bind to query placeholders.
                                        Defaults to None (empty list).

        Returns:
            pd.DataFrame: Query results as a pandas DataFrame with column names preserved.

        Raises:
            sqlite3.Error: If the query execution fails.

        Examples:
            >>> # Simple query without parameters
            >>> df = db.execute_query("SELECT * FROM electricity_prices LIMIT 10")

            >>> # Query with parameters
            >>> df = db.execute_query(
            ...     "SELECT * FROM electricity_prices WHERE date >= ? AND date <= ?",
            ...     ["2024-01-01", "2024-12-31"]
            ... )

            >>> # Aggregation query
            >>> df = db.execute_query(
            ...     "SELECT date, AVG(price_eur) as avg_price FROM electricity_prices GROUP BY date"
            ... )
        """
        conn = self.get_connection()
        try:
            if params is None:
                params = []
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def execute_update(self, query: str, params: Optional[List[Any]] = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query with transaction handling.

        This method automatically commits the transaction and returns the number
        of affected rows. Ideal for data modification operations.

        Args:
            query (str): SQL INSERT, UPDATE, or DELETE query. Use ? placeholders for parameters.
            params (Optional[List[Any]]): List of parameters to bind to query placeholders.
                                        Defaults to None (empty list).

        Returns:
            int: Number of rows affected by the query.

        Raises:
            sqlite3.Error: If the query execution fails or transaction cannot be committed.

        Examples:
            >>> # Insert new record
            >>> rows = db.execute_update(
            ...     "INSERT INTO electricity_prices (timestamp, price_eur) VALUES (?, ?)",
            ...     ["2024-01-01 12:00:00", 45.67]
            ... )

            >>> # Update existing records
            >>> rows = db.execute_update(
            ...     "UPDATE electricity_prices SET processed = 1 WHERE date = ?",
            ...     ["2024-01-01"]
            ... )

            >>> # Delete records
            >>> rows = db.execute_update(
            ...     "DELETE FROM electricity_prices WHERE price_eur < ?",
            ...     [0]
            ... )
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params is None:
                params = []
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def execute_scalar(self, query: str, params: Optional[List[Any]] = None) -> Union[Any, None]:
        """
        Execute a query and return a single scalar value.

        This method is perfect for COUNT, SUM, MAX, MIN, AVG queries or any query
        that returns a single value. Returns None if no result is found.

        Args:
            query (str): SQL query that returns a single value. Use ? placeholders for parameters.
            params (Optional[List[Any]]): List of parameters to bind to query placeholders.
                                        Defaults to None (empty list).

        Returns:
            Union[Any, None]: The scalar value returned by the query, or None if no result.

        Raises:
            sqlite3.Error: If the query execution fails.

        Examples:
            >>> # Count total records
            >>> total_count = db.execute_scalar("SELECT COUNT(*) FROM electricity_prices")

            >>> # Get maximum price
            >>> max_price = db.execute_scalar("SELECT MAX(price_eur) FROM electricity_prices")

            >>> # Get average price for specific date
            >>> avg_price = db.execute_scalar(
            ...     "SELECT AVG(price_eur) FROM electricity_prices WHERE date = ?",
            ...     ["2024-01-01"]
            ... )

            >>> # Check if record exists
            >>> exists = db.execute_scalar(
            ...     "SELECT 1 FROM electricity_prices WHERE timestamp = ? LIMIT 1",
            ...     ["2024-01-01 12:00:00"]
            ... )
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params is None:
                params = []
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()


# Create a singleton instance for use throughout the application
# This ensures consistent database connection handling across all repositories
db_manager = DatabaseManager()
