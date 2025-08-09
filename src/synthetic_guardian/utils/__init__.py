"""
Utility modules for Synthetic Data Guardian
"""

from .logger import get_logger, configure_logging
from .validators import validate_data_schema

# Data handling utilities
class DataLoader:
    """Simple data loader utility."""
    
    @staticmethod
    def load_csv(path: str):
        """Load CSV file."""
        try:
            import pandas as pd
            return pd.read_csv(path)
        except ImportError:
            raise RuntimeError("pandas is required for CSV loading")
    
    @staticmethod 
    def load_json(path: str):
        """Load JSON file."""
        import json
        with open(path, 'r') as f:
            return json.load(f)


class DataExporter:
    """Simple data exporter utility."""
    
    @staticmethod
    def save_csv(data, path: str):
        """Save data as CSV."""
        try:
            import pandas as pd
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_csv(path, index=False)
        except ImportError:
            raise RuntimeError("pandas is required for CSV export")
    
    @staticmethod
    def save_json(data, path: str):
        """Save data as JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class ConfigLoader:
    """Configuration loader utility."""
    pass


class Logger:
    """Logger utility wrapper."""
    
    def __init__(self, name: str):
        self._logger = get_logger(name)
    
    def info(self, message: str):
        self._logger.info(message)
    
    def warning(self, message: str):
        self._logger.warning(message)
    
    def error(self, message: str):
        self._logger.error(message)
    
    def debug(self, message: str):
        self._logger.debug(message)


__all__ = [
    'get_logger',
    'configure_logging',
    'validate_data_schema',
    'DataLoader',
    'DataExporter',
    'ConfigLoader',
    'Logger'
]
