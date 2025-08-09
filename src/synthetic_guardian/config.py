"""
Configuration management for Synthetic Guardian
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .utils.logger import get_logger


@dataclass
class Config:
    """Main configuration class for Synthetic Guardian."""
    
    # Core settings
    name: str = "synthetic-guardian"
    version: str = "1.0.0"
    log_level: str = "INFO"
    
    # Generator settings
    default_generator: str = "tabular"
    generator_config: Dict[str, Any] = field(default_factory=dict)
    
    # Validation settings
    enable_validation: bool = True
    validation_threshold: float = 0.8
    validators: Dict[str, Any] = field(default_factory=dict)
    
    # Watermarking settings
    enable_watermarking: bool = True
    watermark_method: str = "statistical"
    watermark_config: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage settings
    enable_lineage: bool = True
    lineage_backend: str = "file"
    lineage_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    max_workers: int = 4
    timeout: int = 3600
    cache_enabled: bool = True
    
    # Storage settings
    temp_dir: str = "/tmp/synthetic-guardian"
    output_dir: str = "./output"
    
    # Security settings
    encryption_enabled: bool = False
    api_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'log_level': self.log_level,
            'default_generator': self.default_generator,
            'generator_config': self.generator_config,
            'enable_validation': self.enable_validation,
            'validation_threshold': self.validation_threshold,
            'validators': self.validators,
            'enable_watermarking': self.enable_watermarking,
            'watermark_method': self.watermark_method,
            'watermark_config': self.watermark_config,
            'enable_lineage': self.enable_lineage,
            'lineage_backend': self.lineage_backend,
            'lineage_config': self.lineage_config,
            'max_workers': self.max_workers,
            'timeout': self.timeout,
            'cache_enabled': self.cache_enabled,
            'temp_dir': self.temp_dir,
            'output_dir': self.output_dir,
            'encryption_enabled': self.encryption_enabled,
            'api_key': self.api_key
        }


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from file."""
    config_path = Path(config_path)
    logger = get_logger("config")
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return create_default_config()
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                logger.error(f"Unsupported config format: {config_path.suffix}")
                return create_default_config()
        
        # Create config object from dictionary
        config = Config(**config_dict)
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return create_default_config()


def create_default_config() -> Config:
    """Create default configuration."""
    logger = get_logger("config")
    
    config = Config(
        generators={
            'tabular': {
                'backend': 'gaussian_copula',
                'epochs': 100,
                'batch_size': 500
            },
            'timeseries': {
                'sequence_length': 100,
                'sampling_frequency': '1min'
            }
        },
        validators={
            'statistical': {
                'threshold': 0.8,
                'metrics': ['ks_test', 'correlation', 'mean_diff']
            },
            'privacy': {
                'threshold': 0.8,
                'epsilon': 1.0
            },
            'bias': {
                'threshold': 0.8,
                'protected_attributes': []
            },
            'quality': {
                'threshold': 0.8
            }
        },
        watermark_config={
            'strength': 0.01,
            'method': 'mean_shift'
        }
    )
    
    logger.info("Created default configuration")
    return config


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    config_path = Path(config_path)
    logger = get_logger("config")
    
    try:
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                # Default to YAML
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {str(e)}")
        raise


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    logger = get_logger("config")
    
    config = create_default_config()
    
    # Override with environment variables
    if os.getenv('SYNTHETIC_GUARDIAN_LOG_LEVEL'):
        config.log_level = os.getenv('SYNTHETIC_GUARDIAN_LOG_LEVEL')
    
    if os.getenv('SYNTHETIC_GUARDIAN_DEFAULT_GENERATOR'):
        config.default_generator = os.getenv('SYNTHETIC_GUARDIAN_DEFAULT_GENERATOR')
    
    if os.getenv('SYNTHETIC_GUARDIAN_VALIDATION_THRESHOLD'):
        try:
            config.validation_threshold = float(os.getenv('SYNTHETIC_GUARDIAN_VALIDATION_THRESHOLD'))
        except ValueError:
            logger.warning("Invalid validation threshold in environment, using default")
    
    if os.getenv('SYNTHETIC_GUARDIAN_MAX_WORKERS'):
        try:
            config.max_workers = int(os.getenv('SYNTHETIC_GUARDIAN_MAX_WORKERS'))
        except ValueError:
            logger.warning("Invalid max workers in environment, using default")
    
    if os.getenv('SYNTHETIC_GUARDIAN_TEMP_DIR'):
        config.temp_dir = os.getenv('SYNTHETIC_GUARDIAN_TEMP_DIR')
    
    if os.getenv('SYNTHETIC_GUARDIAN_OUTPUT_DIR'):
        config.output_dir = os.getenv('SYNTHETIC_GUARDIAN_OUTPUT_DIR')
    
    if os.getenv('SYNTHETIC_GUARDIAN_API_KEY'):
        config.api_key = os.getenv('SYNTHETIC_GUARDIAN_API_KEY')
    
    logger.info("Loaded configuration from environment variables")
    return config