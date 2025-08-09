"""
Generation Pipeline - Core orchestration for synthetic data generation
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import get_logger
from ..generators.base import BaseGenerator, GeneratorConfig, GenerationResult


@dataclass
class PipelineConfig:
    """Configuration for generation pipeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default_pipeline"
    description: str = ""
    generator_type: str = "tabular"
    generator_params: Dict[str, Any] = field(default_factory=dict)
    data_type: str = "tabular"
    schema: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    watermark_config: Optional[Dict[str, Any]] = None
    output_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenerationPipeline:
    """
    Generation Pipeline - Orchestrates synthetic data generation with validation and watermarking.
    """
    
    def __init__(self, config: Union[Dict, PipelineConfig] = None, logger=None):
        """Initialize generation pipeline."""
        if isinstance(config, dict):
            self.config = PipelineConfig(**config)
        elif config is None:
            self.config = PipelineConfig()
        else:
            self.config = config
            
        self.logger = logger or get_logger(self.__class__.__name__)
        self.id = self.config.id
        self.generator: Optional[BaseGenerator] = None
        self.initialized = False
        self.created_at = time.time()
        self.last_used = None
        
        # Pipeline state
        self.generation_history: List[Dict] = []
        self.metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_records': 0,
            'average_time': 0.0
        }
        
        self.logger.info(f"Created pipeline: {self.config.name} ({self.id})")
    
    @property
    def generator_type(self) -> str:
        """Get generator type."""
        return self.config.generator_type
    
    @property
    def data_type(self) -> str:
        """Get data type."""
        return self.config.data_type
    
    @property
    def validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.config.validation_config
    
    @property
    def watermark_config(self) -> Optional[Dict[str, Any]]:
        """Get watermark configuration."""
        return self.config.watermark_config
    
    async def initialize(self) -> None:
        """Initialize pipeline and components."""
        if self.initialized:
            return
            
        self.logger.info(f"Initializing pipeline: {self.config.name}")
        
        try:
            # Initialize generator
            await self._initialize_generator()
            
            self.initialized = True
            self.logger.info(f"Pipeline initialized successfully: {self.config.name}")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            raise
    
    async def _initialize_generator(self) -> None:
        """Initialize the appropriate generator."""
        generator_type = self.config.generator_type
        
        # Create generator config
        generator_config = GeneratorConfig(
            name=f"{self.config.name}_generator",
            type=generator_type,
            parameters=self.config.generator_params,
            schema=self.config.schema
        )
        
        # Import and create generator based on type
        try:
            if generator_type == "tabular":
                from ..generators import TabularGenerator, HAS_TABULAR
                if not HAS_TABULAR:
                    raise ImportError("TabularGenerator not available - missing dependencies")
                self.generator = TabularGenerator(config=generator_config, logger=self.logger)
            elif generator_type == "timeseries":
                from ..generators import TimeSeriesGenerator, HAS_TIMESERIES
                if not HAS_TIMESERIES:
                    raise ImportError("TimeSeriesGenerator not available - missing dependencies")
                self.generator = TimeSeriesGenerator(config=generator_config, logger=self.logger)
            elif generator_type == "text":
                from ..generators import TextGenerator, HAS_TEXT
                if not HAS_TEXT:
                    raise ImportError("TextGenerator not available - missing dependencies")
                self.generator = TextGenerator(config=generator_config, logger=self.logger)
            elif generator_type == "image":
                from ..generators import ImageGenerator, HAS_IMAGE
                if not HAS_IMAGE:
                    raise ImportError("ImageGenerator not available - missing dependencies")
                self.generator = ImageGenerator(config=generator_config, logger=self.logger)
            elif generator_type == "graph":
                from ..generators import GraphGenerator, HAS_GRAPH
                if not HAS_GRAPH:
                    raise ImportError("GraphGenerator not available - missing dependencies")
                self.generator = GraphGenerator(config=generator_config, logger=self.logger)
            else:
                raise ValueError(f"Unknown generator type: {generator_type}")
        except ImportError as e:
            raise RuntimeError(f"Failed to initialize {generator_type} generator: {str(e)}")
        
        await self.generator.initialize()
        self.logger.info(f"Initialized {generator_type} generator")
    
    async def generate(
        self,
        num_records: int,
        seed: Optional[int] = None,
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate synthetic data using the configured pipeline.
        
        Args:
            num_records: Number of records to generate
            seed: Random seed for reproducibility
            conditions: Generation conditions
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result dictionary
        """
        if not self.initialized:
            await self.initialize()
        
        generation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting generation {generation_id}: {num_records} records")
            
            # Update last used timestamp
            self.last_used = start_time
            
            # Generate data using the generator
            result = await self.generator.generate(
                num_records=num_records,
                seed=seed,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            
            # Create result dictionary
            generation_result = {
                'generation_id': generation_id,
                'data': result.data,
                'metadata': {
                    'pipeline_id': self.id,
                    'pipeline_name': self.config.name,
                    'generator_type': self.config.generator_type,
                    'data_type': self.config.data_type,
                    'num_records': num_records,
                    'seed': seed,
                    'conditions': conditions or {},
                    'generation_time': generation_time,
                    'timestamp': start_time,
                    **result.metadata
                }
            }
            
            # Update metrics
            self._update_metrics('success', generation_time, num_records)
            
            # Add to history
            self.generation_history.append({
                'generation_id': generation_id,
                'timestamp': start_time,
                'num_records': num_records,
                'generation_time': generation_time,
                'status': 'success'
            })
            
            self.logger.info(
                f"Generation {generation_id} completed successfully "
                f"in {generation_time:.2f}s"
            )
            
            return generation_result
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Generation {generation_id} failed: {str(e)}")
            
            # Update metrics
            self._update_metrics('failure', generation_time, 0)
            
            # Add to history
            self.generation_history.append({
                'generation_id': generation_id,
                'timestamp': start_time,
                'num_records': num_records,
                'generation_time': generation_time,
                'status': 'failed',
                'error': str(e)
            })
            
            raise
    
    def _update_metrics(self, status: str, generation_time: float, num_records: int) -> None:
        """Update pipeline metrics."""
        self.metrics['total_generations'] += 1
        
        if status == 'success':
            self.metrics['successful_generations'] += 1
            self.metrics['total_records'] += num_records
        elif status == 'failure':
            self.metrics['failed_generations'] += 1
        
        # Update average time
        total_gens = self.metrics['total_generations']
        current_avg = self.metrics['average_time']
        self.metrics['average_time'] = (
            (current_avg * (total_gens - 1) + generation_time) / total_gens
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            'id': self.id,
            'name': self.config.name,
            'description': self.config.description,
            'generator_type': self.config.generator_type,
            'data_type': self.config.data_type,
            'initialized': self.initialized,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'metrics': self.metrics.copy(),
            'history_count': len(self.generation_history)
        }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get generation history."""
        return self.generation_history[-limit:]
    
    async def validate_config(self) -> List[str]:
        """Validate pipeline configuration."""
        issues = []
        
        if not self.config.name:
            issues.append("Pipeline name is required")
        
        if not self.config.generator_type:
            issues.append("Generator type is required")
        
        if not self.config.data_type:
            issues.append("Data type is required")
        
        # Validate generator if initialized
        if self.generator:
            generator_issues = self.generator.validate_config()
            issues.extend(generator_issues)
        
        return issues
    
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        self.logger.info(f"Cleaning up pipeline: {self.config.name}")
        
        if self.generator:
            await self.generator.cleanup()
        
        self.initialized = False
        self.logger.info(f"Pipeline cleanup completed: {self.config.name}")


class PipelineBuilder:
    """Builder for creating generation pipelines."""
    
    def __init__(self):
        self.config = PipelineConfig()
        self.logger = get_logger(self.__class__.__name__)
    
    def with_name(self, name: str) -> 'PipelineBuilder':
        """Set pipeline name."""
        self.config.name = name
        return self
    
    def with_description(self, description: str) -> 'PipelineBuilder':
        """Set pipeline description."""
        self.config.description = description
        return self
    
    def with_generator(self, generator_type: str, **params) -> 'PipelineBuilder':
        """Set generator type and parameters."""
        self.config.generator_type = generator_type
        self.config.generator_params.update(params)
        return self
    
    def with_data_type(self, data_type: str) -> 'PipelineBuilder':
        """Set data type."""
        self.config.data_type = data_type
        return self
    
    def with_schema(self, schema: Dict[str, Any]) -> 'PipelineBuilder':
        """Set data schema."""
        self.config.schema = schema
        return self
    
    def add_validator(self, validator_type: str, **params) -> 'PipelineBuilder':
        """Add validator configuration."""
        if 'validators' not in self.config.validation_config:
            self.config.validation_config['validators'] = []
        
        self.config.validation_config['validators'].append({
            'type': validator_type,
            'params': params
        })
        return self
    
    def add_watermark(self, method: str, **params) -> 'PipelineBuilder':
        """Add watermark configuration."""
        self.config.watermark_config = {
            'method': method,
            'params': params
        }
        return self
    
    def with_output(self, format: str, path: str = None, **params) -> 'PipelineBuilder':
        """Set output configuration."""
        self.config.output_config = {
            'format': format,
            'path': path,
            **params
        }
        return self
    
    def with_metadata(self, **metadata) -> 'PipelineBuilder':
        """Add metadata."""
        self.config.metadata.update(metadata)
        return self
    
    def build(self) -> GenerationPipeline:
        """Build the generation pipeline."""
        pipeline = GenerationPipeline(config=self.config, logger=self.logger)
        self.logger.info(f"Built pipeline: {self.config.name}")
        return pipeline