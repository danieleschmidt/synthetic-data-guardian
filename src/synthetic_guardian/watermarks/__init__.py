"""
Watermarks module - Data watermarking implementations
"""

from .base import BaseWatermarker

# Try importing optional watermarkers
try:
    from .statistical import StatisticalWatermarker
except ImportError:
    StatisticalWatermarker = None

try:
    from .stegastamp import StegaStampWatermarker
except ImportError:
    StegaStampWatermarker = None

__all__ = [
    'BaseWatermarker',
]

if StatisticalWatermarker:
    __all__.append('StatisticalWatermarker')
    
if StegaStampWatermarker:
    __all__.append('StegaStampWatermarker')