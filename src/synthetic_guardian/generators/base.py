"""
Base classes for data generators with robust error handling and validation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import asyncio
import time

from ..utils.logger import get_logger
from ..utils.validators import validate_data_schema


@dataclass
class GeneratorConfig:
    """Base configuration for generators."""
    name: str
    type: str
    version: str = "1.0.0"
    description: str = ""
    parameters: Dict[str, Any] = None
    schema: Dict[str, Any] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.schema is None:
            self.schema = {}
        if self.constraints is None:
            self.constraints = {}


@dataclass
class GenerationResult:
    """Result from data generation."""
    data: Any
    metadata: Dict[str, Any]
    statistics: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class BaseGenerator(ABC):
    """Base class for all data generators."""
    
    def __init__(self, config: Optional[GeneratorConfig] = None, logger=None):
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)
        self.initialized = False
        self.generation_count = 0
        self.total_records_generated = 0
        self.average_generation_time = 0.0
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the generator."""
        pass
        
    @abstractmethod
    async def generate(
        self,
        num_records: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate synthetic data."""
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up generator resources."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate generator configuration."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get generator information."""
        return {
            'name': self.config.name if self.config else self.__class__.__name__,
            'type': self.config.type if self.config else 'unknown',
            'version': self.config.version if self.config else '1.0.0',
            'initialized': self.initialized,
            'generation_count': self.generation_count,
            'total_records_generated': self.total_records_generated,
            'average_generation_time': self.average_generation_time
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get data schema."""
        return self.config.schema if self.config else {}
    
    def get_constraints(self) -> Dict[str, Any]:
        """Get data constraints."""
        return self.config.constraints if self.config else {}
    
    async def _validate_generation_params(
        self,
        num_records: int,
        **kwargs
    ) -> List[str]:
        """Validate generation parameters."""
        issues = []
        
        if not isinstance(num_records, int) or num_records <= 0:
            issues.append("num_records must be a positive integer")
        
        if num_records > 10_000_000:
            issues.append("num_records cannot exceed 10,000,000")
        
        return issues
    
    async def _update_statistics(self, generation_time: float, num_records: int):
        """Update generation statistics."""
        self.generation_count += 1
        self.total_records_generated += num_records
        
        # Update rolling average
        self.average_generation_time = (
            (self.average_generation_time * (self.generation_count - 1) + generation_time) 
            / self.generation_count
        )
    
    async def _validate_generated_data(
        self,
        data: Any,
        expected_count: int
    ) -> List[str]:
        """Validate generated data."""
        issues = []
        
        if data is None:
            issues.append("Generated data is None")
            return issues
        
        # Check record count for array-like data
        if hasattr(data, '__len__'):
            actual_count = len(data)
            if actual_count != expected_count:
                issues.append(
                    f"Generated {actual_count} records, expected {expected_count}"
                )
        
        # Validate against schema if available
        if self.config and self.config.schema:
            schema_issues = validate_data_schema(data, self.config.schema)
            issues.extend(schema_issues)
        
        return issues


class GeneratorRegistry:
    """Registry for managing generators."""
    
    def __init__(self):
        self._generators: Dict[str, BaseGenerator] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def register(self, name: str, generator: BaseGenerator) -> None:
        """Register a generator."""
        if name in self._generators:
            self.logger.warning(f"Overriding existing generator: {name}")
        
        self._generators[name] = generator
        self.logger.info(f"Registered generator: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a generator."""
        if name in self._generators:
            del self._generators[name]
            self.logger.info(f"Unregistered generator: {name}")
        else:
            self.logger.warning(f"Generator not found for unregistration: {name}")
    
    def get(self, name: str) -> Optional[BaseGenerator]:
        """Get a generator by name."""
        return self._generators.get(name)
    
    def list(self) -> List[str]:
        """List all registered generators."""
        return list(self._generators.keys())
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about all generators."""
        return {
            name: generator.get_info()
            for name, generator in self._generators.items()
        }
    
    async def initialize_all(self) -> None:
        """Initialize all registered generators."""
        for name, generator in self._generators.items():
            try:
                if not generator.initialized:
                    await generator.initialize()
                    self.logger.info(f"Initialized generator: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize generator {name}: {str(e)}")
    
    async def cleanup_all(self) -> None:
        """Clean up all registered generators."""
        for name, generator in self._generators.items():
            try:
                await generator.cleanup()
                self.logger.info(f"Cleaned up generator: {name}")
            except Exception as e:
                self.logger.error(f"Failed to clean up generator {name}: {str(e)}")
        
        self._generators.clear()


# Global generator registry
_global_registry: Optional[GeneratorRegistry] = None


def get_generator_registry() -> GeneratorRegistry:
    """Get global generator registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = GeneratorRegistry()
    return _global_registry
