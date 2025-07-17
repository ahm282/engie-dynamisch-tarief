"""
Base service interface for business logic.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    """Abstract base service interface."""

    def __init__(self, repository=None):
        """Initialize service with repository dependency."""
        self.repository = repository

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters."""
        pass

    def handle_exception(self, e: Exception, context: str = None) -> None:
        """Handle exceptions consistently across services."""
        from fastapi import HTTPException
        error_message = f"{context}: {str(e)}" if context else str(e)
        raise HTTPException(status_code=500, detail=error_message)
