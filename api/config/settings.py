"""
Application configuration settings.
Spring Boot-like configuration management.
"""

import os
from typing import Optional
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    database_path: str = "db/electricity_prices.db"  # Relative to api directory
    connection_timeout: int = 30


class APIConfig(BaseModel):
    """API configuration settings."""

    title: str = "Engie Dynamic Electricity Price API"
    description: str = "REST API for tracking and analyzing Engie Belgium's dynamic electricity prices with detailed analytics"
    version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False

    # CORS settings
    allow_origins: list = ["*"]
    allow_credentials: bool = True
    allow_methods: list = ["*"]
    allow_headers: list = ["*"]


class ApplicationConfig:
    """Main application configuration."""

    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()

    @property
    def database_path(self) -> str:
        """Get database path."""
        # Convert relative path to absolute path based on this file's location
        if os.path.isabs(self.database.database_path):
            return self.database.database_path
        else:
            # Get the absolute path relative to the api directory
            # This file is in: api/config/settings.py
            # Database should be at: api/db/electricity_prices.db
            config_dir = os.path.dirname(
                os.path.abspath(__file__))  # api/config
            api_dir = os.path.dirname(config_dir)  # api
            # Remove the 'api/' prefix from the database path since we're already in the api directory
            db_relative_path = self.database.database_path
            if db_relative_path.startswith('api/'):
                db_relative_path = db_relative_path[4:]  # Remove 'api/' prefix
            return os.path.join(api_dir, db_relative_path)

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.api.debug


# Global configuration instance
app_config = ApplicationConfig()
