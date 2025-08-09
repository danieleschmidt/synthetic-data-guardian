"""
Base watermarker class for data watermarking
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import asyncio

from ..utils.logger import get_logger


class BaseWatermarker(ABC):
    """Base class for all data watermarkers."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the watermarker."""
        pass
    
    @abstractmethod
    async def embed(
        self,
        data: Any,
        message: str,
        key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Embed watermark in data.
        
        Args:
            data: Data to watermark
            message: Watermark message
            key: Watermark key
            **kwargs: Additional parameters
            
        Returns:
            Dict with watermarked data and metadata
        """
        pass
    
    @abstractmethod
    async def extract(
        self,
        data: Any,
        key: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Extract watermark from data.
        
        Args:
            data: Watermarked data
            key: Watermark key
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (is_valid, extracted_message)
        """
        pass
    
    @abstractmethod
    async def verify(
        self,
        data: Any,
        expected_message: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify watermark in data.
        
        Args:
            data: Data to verify
            expected_message: Expected watermark message
            key: Watermark key
            **kwargs: Additional parameters
            
        Returns:
            Dict with verification results
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up watermarker resources."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get watermarker information."""
        return {
            'name': self.__class__.__name__,
            'initialized': self.initialized,
            'config': self.config
        }