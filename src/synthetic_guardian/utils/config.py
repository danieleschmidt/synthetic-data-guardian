"""
Configuration management with validation, environment variable support, and schema validation
"""

import os
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "synthetic_guardian"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = "disable"
    
    def to_url(self) -> str:
        """Convert to database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    backend: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ttl: int = 3600  # seconds
    max_size: int = 1000
    compression: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_auth: bool = True
    api_key_required: bool = True
    api_keys: List[str] = field(default_factory=list)
    jwt_secret: Optional[str] = None
    jwt_expiry: int = 3600  # seconds
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # seconds
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_honeypot: bool = False
    encryption_key: Optional[str] = None


@dataclass 
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    health_check_enabled: bool = True
    prometheus_port: int = 9090
    jaeger_endpoint: Optional[str] = None
    sentry_dsn: Optional[str] = None
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None


@dataclass
class GenerationConfig:
    """Default generation configuration."""
    max_records_per_request: int = 1000000
    default_timeout: int = 3600  # seconds
    enable_watermarking: bool = True
    enable_validation: bool = True
    enable_lineage: bool = True
    temp_dir: str = "/tmp/synthetic-guardian"
    max_concurrent_generations: int = 10
    default_generator: str = "tabular"


@dataclass
class Config:
    """Main configuration class."""
    name: str = "synthetic-data-guardian"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Additional settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and processing."""
        # Environment-specific settings
        if self.environment == "production":
            self.debug = False
            self.monitoring.log_level = "INFO"
            self.security.enable_auth = True
        elif self.environment == "development":
            self.debug = True
            self.monitoring.log_level = "DEBUG"
            self.security.enable_auth = False
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate database config
        if not self.database.host:
            issues.append("Database host is required")
        if not (1 <= self.database.port <= 65535):
            issues.append("Database port must be between 1 and 65535")
        
        # Validate security config
        if self.security.enable_auth and not self.security.api_keys and not self.security.jwt_secret:
            issues.append("Authentication is enabled but no API keys or JWT secret configured")
        
        # Validate monitoring config
        if self.monitoring.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            issues.append("Invalid log level")
        
        # Validate generation config
        if self.generation.max_records_per_request <= 0:
            issues.append("Max records per request must be positive")
        if self.generation.default_timeout <= 0:
            issues.append("Default timeout must be positive")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        if not HAS_YAML:
            raise RuntimeError("PyYAML is required for YAML support")
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() in [".yaml", ".yml"] and HAS_YAML:
            with open(path, 'w') as f:
                f.write(self.to_yaml())
        else:
            with open(path, 'w') as f:
                f.write(self.to_json())
        
        logger.info(f"Configuration saved to {path}")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = "SGD_",
    validate: bool = True
) -> Config:
    """Load configuration from file and environment variables."""
    config_dict = {}
    
    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            
            with open(config_path) as f:
                if config_path.suffix.lower() in [".yaml", ".yml"] and HAS_YAML:
                    config_dict = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() in [".yaml", ".yml"] and not HAS_YAML:
                    logger.warning("PyYAML not available, cannot load YAML config")
                    config_dict = {}
                else:
                    config_dict = json.load(f)
        else:
            logger.warning(f"Configuration file not found: {config_path}")
    
    # Override with environment variables
    env_overrides = load_env_vars(env_prefix)
    config_dict = merge_configs(config_dict, env_overrides)
    
    # Create config instance
    config = create_config_from_dict(config_dict)
    
    # Validate if requested
    if validate:
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
    
    return config


def load_env_vars(prefix: str = "SGD_") -> Dict[str, Any]:
    """Load configuration from environment variables."""
    env_config = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert key from ENV_FORMAT to config format
            config_key = key[len(prefix):].lower().replace("_", ".")
            
            # Parse value
            parsed_value = parse_env_value(value)
            
            # Set nested value
            set_nested_value(env_config, config_key, parsed_value)
    
    return env_config


def parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type."""
    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    
    # Integer
    if re.match(r"^-?\d+$", value):
        return int(value)
    
    # Float
    if re.match(r"^-?\d+\.\d+$", value):
        return float(value)
    
    # JSON array/object
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    # Comma-separated list
    if "," in value:
        return [item.strip() for item in value.split(",")]
    
    # String
    return value


def set_nested_value(dict_obj: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set nested dictionary value using dot notation."""
    keys = key_path.split(".")
    current = dict_obj
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration dictionaries recursively."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """Create Config instance from dictionary."""
    # Extract component configs
    database_dict = config_dict.pop("database", {})
    cache_dict = config_dict.pop("cache", {})
    security_dict = config_dict.pop("security", {})
    monitoring_dict = config_dict.pop("monitoring", {})
    generation_dict = config_dict.pop("generation", {})
    extra_dict = config_dict.pop("extra", {})
    
    # Create component instances
    database_config = DatabaseConfig(**database_dict)
    cache_config = CacheConfig(**cache_dict)
    security_config = SecurityConfig(**security_dict)
    monitoring_config = MonitoringConfig(**monitoring_dict)
    generation_config = GenerationConfig(**generation_dict)
    
    # Create main config
    return Config(
        database=database_config,
        cache=cache_config,
        security=security_config,
        monitoring=monitoring_config,
        generation=generation_config,
        extra=extra_dict,
        **config_dict
    )


def create_default_config() -> Config:
    """Create default configuration."""
    return Config()


def validate_config_schema(config_dict: Dict[str, Any]) -> List[str]:
    """Validate configuration dictionary against schema."""
    # This would typically use a schema validation library like jsonschema
    # For now, implement basic validation
    issues = []
    
    required_fields = ["name", "version"]
    for field in required_fields:
        if field not in config_dict:
            issues.append(f"Missing required field: {field}")
    
    return issues


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config


# Load configuration on import
try:
    # Try to load from default locations
    config_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path("config.json"),
        Path("/etc/synthetic-guardian/config.yaml"),
        Path(os.path.expanduser("~/.synthetic-guardian/config.yaml"))
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    # Also check environment variable
    env_config_path = os.getenv("SGD_CONFIG_PATH")
    if env_config_path:
        config_path = Path(env_config_path)
    
    _global_config = load_config(config_path)
    
except Exception as e:
    logger.warning(f"Failed to load configuration: {e}")
    _global_config = create_default_config()
