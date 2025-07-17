"""
Base repository interface for data access.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd


class BaseRepository(ABC):
    """Abstract base repository interface."""

    @abstractmethod
    def find_all(self) -> pd.DataFrame:
        """Find all records."""
        pass

    @abstractmethod
    def find_by_id(self, record_id: Any) -> Optional[pd.Series]:
        """Find record by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total records."""
        pass
