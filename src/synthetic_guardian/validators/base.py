"""
Base validator class for synthetic data validation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import asyncio
import time

from ..utils.logger import get_logger


class BaseValidator(ABC):
    """Base class for all data validators."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the validator."""
        pass
    
    @abstractmethod
    async def validate(
        self,
        data: Any,
        reference_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate the data.
        
        Returns:
            Dict with validation results including:
            - passed: bool
            - score: float (0.0 to 1.0)
            - message: str
            - details: dict
            - errors: list
            - warnings: list
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up validator resources."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get validator information."""
        return {
            'name': self.__class__.__name__,
            'initialized': self.initialized,
            'config': self.config
        }