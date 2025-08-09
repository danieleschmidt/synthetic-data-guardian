"""
Generators module - Synthetic data generation implementations
"""

from .base import BaseGenerator, GeneratorConfig, GenerationResult, GeneratorRegistry, get_generator_registry

# Optional imports for generators that require external dependencies
try:
    from .tabular import TabularGenerator
    HAS_TABULAR = True
except ImportError:
    HAS_TABULAR = False
    TabularGenerator = None

try:
    from .timeseries import TimeSeriesGenerator
    HAS_TIMESERIES = True
except ImportError:
    HAS_TIMESERIES = False
    TimeSeriesGenerator = None

try:
    from .text import TextGenerator
    HAS_TEXT = True
except ImportError:
    HAS_TEXT = False
    TextGenerator = None

try:
    from .image import ImageGenerator
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False
    ImageGenerator = None

try:
    from .graph import GraphGenerator
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False
    GraphGenerator = None

__all__ = [
    'BaseGenerator',
    'GeneratorConfig', 
    'GenerationResult',
    'GeneratorRegistry',
    'get_generator_registry',
    'HAS_TABULAR',
    'HAS_TIMESERIES', 
    'HAS_TEXT',
    'HAS_IMAGE',
    'HAS_GRAPH'
]

# Add generators to __all__ if available
if HAS_TABULAR:
    __all__.append('TabularGenerator')
if HAS_TIMESERIES:
    __all__.append('TimeSeriesGenerator')
if HAS_TEXT:
    __all__.append('TextGenerator')
if HAS_IMAGE:
    __all__.append('ImageGenerator')
if HAS_GRAPH:
    __all__.append('GraphGenerator')