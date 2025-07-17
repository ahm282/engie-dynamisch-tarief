"""
Controllers package for API endpoint handlers.
Imports all controllers for easy access.
"""

from fastapi import APIRouter

# Base controller
from .base_controller import BaseController

# Individual controllers
from .info_controller import InfoController
from .price_controller import PriceController
from .statistics_controller import StatisticsController
from .extreme_analysis_controller import ExtremeAnalysisController
from .negative_price_controller import NegativePriceController
from .expensive_price_controller import ExpensivePriceController

# Legacy imports for backwards compatibility
from ..services import ElectricityPriceService
from ..repositories import ElectricityPriceRepository
from ..models import (
    PriceRecord, DailyStats, HourlyStats, DatabaseStats,
    APIInfo, HealthResponse
)


def get_electricity_price_service() -> ElectricityPriceService:
    """Dependency injection for ElectricityPriceService (legacy compatibility)."""
    repository = ElectricityPriceRepository()
    return ElectricityPriceService(repository)


class ElectricityPriceController:
    """
    Aggregate controller that combines all electricity price controllers.
    Maintains backwards compatibility while using the new modular controllers.
    """

    def __init__(self):
        """Initialize aggregate controller with all sub-controllers."""
        self.router = APIRouter()

        # Initialize individual controllers
        self.info_controller = InfoController()
        self.price_controller = PriceController()
        self.statistics_controller = StatisticsController()
        self.extreme_controller = ExtremeAnalysisController()
        self.negative_price_controller = NegativePriceController()
        self.expensive_price_controller = ExpensivePriceController()

        # Include all routers
        self._setup_aggregate_routes()

    def _setup_aggregate_routes(self):
        """Setup aggregate routes by including all controller routers."""
        # Include all individual controller routes
        self.router.include_router(self.info_controller.router)
        self.router.include_router(self.price_controller.router)
        self.router.include_router(self.statistics_controller.router)
        self.router.include_router(self.extreme_controller.router)
        self.router.include_router(self.negative_price_controller.router)
        self.router.include_router(self.expensive_price_controller.router)


# Create controller instances
info_controller = InfoController()
price_controller = PriceController()
statistics_controller = StatisticsController()
extreme_analysis_controller = ExtremeAnalysisController()
negative_price_controller = NegativePriceController()
expensive_price_controller = ExpensivePriceController()

# Create aggregate controller for backwards compatibility
electricity_controller = ElectricityPriceController()

__all__ = [
    # Base controller
    "BaseController",

    # Individual controllers
    "InfoController",
    "PriceController",
    "StatisticsController",
    "ExtremeAnalysisController",
    "NegativePriceController",
    "ExpensivePriceController",

    # Controller instances
    "info_controller",
    "price_controller",
    "statistics_controller",
    "extreme_analysis_controller",
    "negative_price_controller",
    "expensive_price_controller",

    # Aggregate controller
    "ElectricityPriceController",
    "electricity_controller",

    # Legacy compatibility
    "get_electricity_price_service"
]
