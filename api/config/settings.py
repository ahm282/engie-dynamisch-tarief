"""
Application configuration settings.
Spring Boot-like configuration management.
"""

import os
from typing import Optional
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    database_path: str = "api/db/electricity_prices.db"
    connection_timeout: int = 30


class APIConfig(BaseModel):
    """API configuration settings."""

    title: str = "Electricity Price API"
    description: str = "REST API for electricity price data analysis with MVC architecture"
    version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
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
        return self.database.database_path

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.api.debug


# Global configuration instance
app_config = ApplicationConfig()
