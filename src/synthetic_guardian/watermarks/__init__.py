"""
Watermarks module - Data watermarking implementations
"""

from .base import BaseWatermarker
from .statistical import StatisticalWatermarker
from .stegastamp import StegaStampWatermarker

__all__ = [
    'BaseWatermarker',
    'StatisticalWatermarker',
    'StegaStampWatermarker'
]