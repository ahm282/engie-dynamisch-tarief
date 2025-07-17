"""
Database connection and management utilities.
"""

import sqlite3
import os
from fastapi import HTTPException

from ..config import app_config


class DatabaseManager:
    """Handles database connections and basic operations."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or app_config.database_path

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with error handling."""
        if not os.path.exists(self.db_path):
            raise HTTPException(status_code=500, detail="Database not found")
        return sqlite3.connect(self.db_path)

    def execute_query(self, query: str, params: list = None) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        conn = self.get_connection()
        try:
            if params:
                cursor = conn.execute(query, params)
            else:
                cursor = conn.execute(query)
            return cursor, conn
        except Exception as e:
            conn.close()
            raise e

    def close_connection(self, conn: sqlite3.Connection):
        """Close database connection."""
        if conn:
            conn.close()


# Global database manager instance
db_manager = DatabaseManager()
