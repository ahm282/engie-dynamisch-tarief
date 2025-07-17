"""
Base controller interface for API endpoints.

This module provides the abstract base class for all API controllers in the
electricity price analysis system. It enforces consistent patterns and provides
common functionality across all endpoint handlers.

Tags:
    - base-controller
    - abstract-interface
    - mvc-pattern
    - error-handling
    - api-standards

Features:
    - Standardized router initialization
    - Consistent error handling patterns
    - Request validation framework
    - Exception handling with context
    - FastAPI integration patterns

Architecture:
    All controllers inherit from BaseController and must implement:
    - _setup_routes(): Define endpoint routes and handlers
    - Optional: Custom validation and error handling
    
Usage:
    ```python
    class MyController(BaseController):
        def _setup_routes(self):
            @self.router.get("/my-endpoint")
            async def my_endpoint():
                return {"message": "Hello World"}
    ```

Author: Electricity Price Analysis System
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from fastapi import APIRouter, HTTPException
from typing import Any, Dict, Optional


class BaseController(ABC):
    """
    Abstract base controller for consistent API endpoint patterns.

    This class provides the foundation for all API controllers with:
    - Standardized FastAPI router setup
    - Consistent error handling and validation
    - Common utility methods for request processing
    - Exception handling with contextual error messages

    All concrete controllers must inherit from this class and implement
    the _setup_routes() method to define their specific endpoints.

    Attributes:
        router (APIRouter): FastAPI router instance for endpoint registration

    Methods:
        _setup_routes(): Abstract method for route definition (must implement)
        validate_request(): Request validation with parameter checking
        handle_exception(): Standardized exception handling with context

    Example:
        >>> class PriceController(BaseController):
        ...     def _setup_routes(self):
        ...         @self.router.get("/prices")
        ...         async def get_prices():
        ...             return {"prices": []}
    """

    def __init__(self):
        """
        Initialize controller with FastAPI router.

        Creates a new APIRouter instance and calls _setup_routes() to register
        all endpoint handlers defined by the concrete controller implementation.
        """
        self.router = APIRouter()
        self._setup_routes()

    @abstractmethod
    def _setup_routes(self):
        """
        Setup routes for this controller.

        This abstract method must be implemented by all concrete controllers
        to define their specific API endpoints using the self.router instance.

        Example:
            def _setup_routes(self):
                @self.router.get("/endpoint")
                async def my_endpoint():
                    return {"status": "success"}
        """
        pass

    def validate_request(self, **kwargs) -> bool:
        """
        Validate request parameters before processing.

        Provides a hook for custom request validation logic. Can be overridden
        by concrete controllers to implement specific validation rules.

        Args:
            **kwargs: Request parameters to validate

        Returns:
            bool: True if validation passes, False otherwise

        Example:
            def validate_request(self, start_date=None, end_date=None):
                if start_date and end_date:
                    return start_date <= end_date
                return True
        """
        return True

    def handle_exception(self, e: Exception, context: Optional[str] = None) -> None:
        """
        Handle exceptions consistently across all controllers.

        Provides standardized error handling by converting exceptions into
        HTTP 500 errors with contextual information for debugging.

        Args:
            e (Exception): The exception that occurred
            context (Optional[str]): Additional context about where the error occurred

        Raises:
            HTTPException: Always raises HTTP 500 with error details

        Example:
            try:
                result = risky_operation()
            except Exception as e:
                self.handle_exception(e, "Error in risky operation")
        """
        error_message = f"{context}: {str(e)}" if context else str(e)
        raise HTTPException(status_code=500, detail=error_message)
