"""
Validators module - Data validation implementations
"""

from .base import BaseValidator

# Optional imports for validators that require external dependencies
try:
    from .statistical import StatisticalValidator
    HAS_STATISTICAL = True
except ImportError:
    HAS_STATISTICAL = False
    StatisticalValidator = None

try:
    from .privacy import PrivacyValidator
    HAS_PRIVACY = True
except ImportError:
    HAS_PRIVACY = False
    PrivacyValidator = None

try:
    from .bias import BiasValidator
    HAS_BIAS = True
except ImportError:
    HAS_BIAS = False
    BiasValidator = None

try:
    from .quality import QualityValidator
    HAS_QUALITY = True
except ImportError:
    HAS_QUALITY = False
    QualityValidator = None

__all__ = [
    'BaseValidator',
    'HAS_STATISTICAL',
    'HAS_PRIVACY',
    'HAS_BIAS', 
    'HAS_QUALITY'
]

# Add validators to __all__ if available
if HAS_STATISTICAL:
    __all__.append('StatisticalValidator')
if HAS_PRIVACY:
    __all__.append('PrivacyValidator')
if HAS_BIAS:
    __all__.append('BiasValidator')
if HAS_QUALITY:
    __all__.append('QualityValidator')