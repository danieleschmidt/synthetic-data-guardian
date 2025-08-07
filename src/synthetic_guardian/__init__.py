"""
Synthetic Data Guardian - Enterprise-grade synthetic data pipeline
with built-in validation, watermarking, and auditable lineage tracking.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "info@terragonlabs.com"

# Core imports
from .core.guardian import Guardian
from .core.pipeline import GenerationPipeline, PipelineBuilder
from .core.result import GenerationResult
from .core.validation_report import ValidationReport

# Generator imports
from .generators import (
    TabularGenerator,
    TimeSeriesGenerator,
    TextGenerator,
    ImageGenerator,
    GraphGenerator,
)

# Validator imports
from .validators import (
    StatisticalValidator,
    PrivacyValidator,
    BiasValidator,
    QualityValidator,
)

# Watermarker imports
from .watermarks import (
    StegaStampWatermarker,
    StatisticalWatermarker,
)

# Utility imports
from .utils import (
    Logger,
    ConfigLoader,
    DataLoader,
    DataExporter,
)

# Configuration
from .config import Config, load_config, create_default_config

# CLI
from .cli import main as cli_main

__all__ = [
    # Core classes
    'Guardian',
    'GenerationPipeline',
    'PipelineBuilder',
    'GenerationResult',
    'ValidationReport',
    
    # Generators
    'TabularGenerator',
    'TimeSeriesGenerator',
    'TextGenerator',
    'ImageGenerator',
    'GraphGenerator',
    
    # Validators
    'StatisticalValidator',
    'PrivacyValidator',
    'BiasValidator',
    'QualityValidator',
    
    # Watermarkers
    'StegaStampWatermarker',
    'StatisticalWatermarker',
    
    # Utilities
    'Logger',
    'ConfigLoader',
    'DataLoader',
    'DataExporter',
    
    # Configuration
    'Config',
    'load_config',
    'create_default_config',
    
    # CLI
    'cli_main',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
]

# Package-level configuration
def configure_logging(level='INFO', format=None):
    """Configure package-wide logging settings."""
    from .utils.logger import configure_logging as _configure_logging
    return _configure_logging(level, format)

def get_version():
    """Get the current package version."""
    return __version__

def create_guardian(**kwargs):
    """Convenience function to create a Guardian instance with default settings."""
    return Guardian(**kwargs)

def create_pipeline(**kwargs):
    """Convenience function to create a GenerationPipeline instance."""
    return GenerationPipeline(**kwargs)

# Initialize package logging
try:
    configure_logging()
except Exception:
    pass  # Fail silently if logging setup fails
