"""
Base controller interface for API endpoints.
"""

from abc import ABC, abstractmethod
from fastapi import APIRouter, HTTPException
from typing import Any


class BaseController(ABC):
    """Abstract base controller interface."""

    def __init__(self):
        """Initialize controller with router."""
        self.router = APIRouter()
        self._setup_routes()

    @abstractmethod
    def _setup_routes(self):
        """Setup routes for this controller."""
        pass

    def validate_request(self, **kwargs) -> bool:
        """Validate request parameters."""
        return True

    def handle_exception(self, e: Exception, context: str = None) -> None:
        """Handle exceptions consistently across controllers."""
        error_message = f"{context}: {str(e)}" if context else str(e)
        raise HTTPException(status_code=500, detail=error_message)
