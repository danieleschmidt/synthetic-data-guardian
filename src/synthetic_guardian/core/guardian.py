"""
Guardian - Core orchestrator for synthetic data generation and validation
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import Config
from .pipeline import GenerationPipeline
from .result import GenerationResult, ValidationReport
from ..validators.base import BaseValidator
from ..generators.base import BaseGenerator
from ..watermarks.base import BaseWatermarker


@dataclass
class GuardianConfig:
    """Configuration for Guardian instance."""
    name: str = "synthetic-data-guardian"
    version: str = "1.0.0"
    log_level: str = "INFO"
    enable_lineage: bool = True
    enable_watermarking: bool = True
    enable_validation: bool = True
    max_concurrent_generations: int = 10
    default_timeout: int = 3600  # 1 hour
    temp_dir: Optional[Path] = None
    cache_enabled: bool = True
    metrics_enabled: bool = True


class Guardian:
    """
    Guardian - Core orchestrator for enterprise-grade synthetic data generation.
    
    The Guardian provides a unified interface for:
    - Synthetic data generation across multiple backends
    - Comprehensive validation and quality assessment
    - Privacy preservation and compliance checking
    - Watermarking and lineage tracking
    - Enterprise-grade security and monitoring
    """
    
    def __init__(self, config: Optional[Union[GuardianConfig, Dict]] = None, logger=None):
        """Initialize Guardian with configuration."""
        # Setup configuration
        if isinstance(config, dict):
            self.config = GuardianConfig(**config)
        elif config is None:
            self.config = GuardianConfig()
        else:
            self.config = config
            
        # Setup logging
        self.logger = logger or get_logger(self.__class__.__name__, level=self.config.log_level)
        
        # Internal state
        self.pipelines: Dict[str, GenerationPipeline] = {}
        self.active_tasks: Dict[str, Dict] = {}
        self.generators: Dict[str, BaseGenerator] = {}
        self.validators: Dict[str, BaseValidator] = {}
        self.watermarkers: Dict[str, BaseWatermarker] = {}
        self.initialized = False
        
        # Metrics and monitoring
        self.metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_records_generated': 0,
            'average_generation_time': 0,
            'total_validations': 0,
            'validation_pass_rate': 0,
        }
        
        self.logger.info(f"Guardian initialized with config: {self.config.name}")
        
    async def initialize(self) -> None:
        """Initialize Guardian and all components."""
        if self.initialized:
            return
            
        self.logger.info("Initializing Synthetic Data Guardian...")
        
        try:
            # Initialize built-in components
            await self._initialize_generators()
            await self._initialize_validators()
            await self._initialize_watermarkers()
            
            # Setup temporary directory
            if self.config.temp_dir is None:
                self.config.temp_dir = Path("/tmp/synthetic-guardian")
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            
            self.initialized = True
            self.logger.info("Guardian initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Guardian initialization failed: {str(e)}")
            raise
    
    async def _initialize_generators(self) -> None:
        """Initialize available generators."""
        from ..generators import (
            TabularGenerator,
            TimeSeriesGenerator,
            TextGenerator,
            ImageGenerator,
            GraphGenerator
        )
        
        generator_classes = {
            'tabular': TabularGenerator,
            'timeseries': TimeSeriesGenerator,
            'text': TextGenerator,
            'image': ImageGenerator,
            'graph': GraphGenerator
        }
        
        for name, generator_class in generator_classes.items():
            try:
                generator = generator_class(logger=self.logger)
                await generator.initialize()
                self.generators[name] = generator
                self.logger.debug(f"Initialized {name} generator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {name} generator: {str(e)}")
    
    async def _initialize_validators(self) -> None:
        """Initialize available validators."""
        from ..validators import (
            StatisticalValidator,
            PrivacyValidator,
            BiasValidator,
            QualityValidator
        )
        
        validator_classes = {
            'statistical': StatisticalValidator,
            'privacy': PrivacyValidator,
            'bias': BiasValidator,
            'quality': QualityValidator
        }
        
        for name, validator_class in validator_classes.items():
            try:
                validator = validator_class(logger=self.logger)
                await validator.initialize()
                self.validators[name] = validator
                self.logger.debug(f"Initialized {name} validator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {name} validator: {str(e)}")
    
    async def _initialize_watermarkers(self) -> None:
        """Initialize available watermarkers."""
        from ..watermarks import (
            StegaStampWatermarker,
            StatisticalWatermarker
        )
        
        watermarker_classes = {
            'stegastamp': StegaStampWatermarker,
            'statistical': StatisticalWatermarker
        }
        
        for name, watermarker_class in watermarker_classes.items():
            try:
                watermarker = watermarker_class(logger=self.logger)
                await watermarker.initialize()
                self.watermarkers[name] = watermarker
                self.logger.debug(f"Initialized {name} watermarker")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {name} watermarker: {str(e)}")
    
    async def generate(
        self,
        pipeline_config: Union[str, Dict, GenerationPipeline],
        num_records: int,
        seed: Optional[int] = None,
        conditions: Optional[Dict] = None,
        validate: bool = True,
        watermark: bool = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate synthetic data using specified pipeline.
        
        Args:
            pipeline_config: Pipeline configuration or existing pipeline
            num_records: Number of records to generate
            seed: Random seed for reproducibility
            conditions: Conditions to apply during generation
            validate: Whether to run validation
            watermark: Whether to apply watermarking
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with synthetic data and metadata
        """
        if not self.initialized:
            await self.initialize()
            
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting generation task {task_id}")
            
            # Get or create pipeline
            pipeline = await self._get_or_create_pipeline(pipeline_config)
            
            # Create task tracking
            task = {
                'id': task_id,
                'pipeline_id': pipeline.id,
                'status': 'running',
                'start_time': start_time,
                'num_records': num_records,
                'seed': seed,
                'conditions': conditions or {}
            }
            self.active_tasks[task_id] = task
            
            # Generate synthetic data
            generation_result = await pipeline.generate(
                num_records=num_records,
                seed=seed,
                conditions=conditions,
                **kwargs
            )
            
            # Create result object
            result = GenerationResult(
                task_id=task_id,
                pipeline_id=pipeline.id,
                data=generation_result['data'],
                metadata={
                    'generation_time': time.time() - start_time,
                    'num_records': len(generation_result['data']),
                    'seed': seed,
                    'conditions': conditions or {},
                    'generator': pipeline.generator_type,
                    'pipeline_config': pipeline.config,
                    **generation_result.get('metadata', {})
                }
            )
            
            # Apply validation if enabled
            if validate and self.config.enable_validation:
                validation_result = await self._validate_result(result, pipeline)
                result.validation_report = validation_result
            
            # Apply watermarking if enabled
            watermark_enabled = watermark if watermark is not None else self.config.enable_watermarking
            if watermark_enabled and pipeline.watermark_config:
                watermark_result = await self._apply_watermarking(result, pipeline)
                result.watermark_info = watermark_result
            
            # Track lineage if enabled
            if self.config.enable_lineage:
                lineage_id = await self._track_lineage(task, result)
                result.lineage_id = lineage_id
            
            # Update metrics
            self._update_metrics('successful_generation', result)
            
            # Update task status
            task['status'] = 'completed'
            task['end_time'] = time.time()
            task['result'] = result
            
            self.logger.info(f"Generation task {task_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Generation task {task_id} failed: {str(e)}")
            
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'failed'
                self.active_tasks[task_id]['error'] = str(e)
                self.active_tasks[task_id]['end_time'] = time.time()
            
            # Update metrics
            self._update_metrics('failed_generation', None)
            
            raise
    
    async def _get_or_create_pipeline(self, pipeline_config: Union[str, Dict, GenerationPipeline]) -> GenerationPipeline:
        """Get existing pipeline or create new one."""
        if isinstance(pipeline_config, GenerationPipeline):
            return pipeline_config
            
        if isinstance(pipeline_config, str):
            # Pipeline ID or name
            if pipeline_config in self.pipelines:
                return self.pipelines[pipeline_config]
            else:
                raise ValueError(f"Pipeline not found: {pipeline_config}")
        
        if isinstance(pipeline_config, dict):
            # Create new pipeline from config
            pipeline_id = pipeline_config.get('id', str(uuid.uuid4()))
            
            if pipeline_id not in self.pipelines:
                pipeline = GenerationPipeline(config=pipeline_config, logger=self.logger)
                await pipeline.initialize()
                self.pipelines[pipeline_id] = pipeline
                self.logger.info(f"Created new pipeline: {pipeline_id}")
            
            return self.pipelines[pipeline_id]
        
        raise ValueError(f"Invalid pipeline configuration type: {type(pipeline_config)}")
    
    async def _validate_result(self, result: GenerationResult, pipeline: GenerationPipeline) -> ValidationReport:
        """Validate generation result."""
        validation_report = ValidationReport(task_id=result.task_id)
        
        # Run configured validators
        validator_configs = pipeline.validation_config.get('validators', [])
        
        for validator_config in validator_configs:
            validator_type = validator_config.get('type')
            validator_params = validator_config.get('params', {})
            
            if validator_type in self.validators:
                try:
                    validator = self.validators[validator_type]
                    validator_result = await validator.validate(result.data, **validator_params)
                    validation_report.add_validator_result(validator_type, validator_result)
                except Exception as e:
                    self.logger.error(f"Validator {validator_type} failed: {str(e)}")
                    validation_report.add_validator_result(validator_type, {
                        'passed': False,
                        'score': 0.0,
                        'errors': [str(e)]
                    })
        
        return validation_report
    
    async def _apply_watermarking(self, result: GenerationResult, pipeline: GenerationPipeline) -> Dict:
        """Apply watermarking to result."""
        watermark_config = pipeline.watermark_config
        watermarker_type = watermark_config.get('method', 'statistical')
        
        if watermarker_type in self.watermarkers:
            watermarker = self.watermarkers[watermarker_type]
            return await watermarker.embed(
                data=result.data,
                message=watermark_config.get('message', f'synthetic:{result.task_id}'),
                **watermark_config.get('params', {})
            )
        
        raise ValueError(f"Watermarker not found: {watermarker_type}")
    
    async def _track_lineage(self, task: Dict, result: GenerationResult) -> str:
        """Track lineage information."""
        # In a full implementation, this would use a graph database
        # For now, return a simple lineage ID
        lineage_id = f"lineage_{task['id']}_{int(time.time())}"
        
        lineage_info = {
            'lineage_id': lineage_id,
            'task_id': task['id'],
            'pipeline_id': task['pipeline_id'],
            'generation_time': task['start_time'],
            'num_records': result.metadata['num_records'],
            'generator': result.metadata['generator']
        }
        
        self.logger.debug(f"Tracked lineage: {lineage_id}")
        return lineage_id
    
    def _update_metrics(self, event_type: str, result: Optional[GenerationResult]) -> None:
        """Update internal metrics."""
        self.metrics['total_generations'] += 1
        
        if event_type == 'successful_generation':
            self.metrics['successful_generations'] += 1
            if result:
                self.metrics['total_records_generated'] += result.metadata['num_records']
        elif event_type == 'failed_generation':
            self.metrics['failed_generations'] += 1
    
    async def validate(
        self,
        data: Any,
        validators: List[str] = None,
        reference_data: Any = None,
        **kwargs
    ) -> ValidationReport:
        """
        Validate data using specified validators.
        
        Args:
            data: Data to validate
            validators: List of validator names to use
            reference_data: Reference data for comparison
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationReport with results
        """
        if not self.initialized:
            await self.initialize()
            
        validation_report = ValidationReport()
        
        # Use all validators if none specified
        if validators is None:
            validators = list(self.validators.keys())
        
        for validator_name in validators:
            if validator_name in self.validators:
                try:
                    validator = self.validators[validator_name]
                    validator_result = await validator.validate(
                        data=data,
                        reference_data=reference_data,
                        **kwargs
                    )
                    validation_report.add_validator_result(validator_name, validator_result)
                except Exception as e:
                    self.logger.error(f"Validator {validator_name} failed: {str(e)}")
                    validation_report.add_validator_result(validator_name, {
                        'passed': False,
                        'score': 0.0,
                        'errors': [str(e)]
                    })
        
        self.metrics['total_validations'] += 1
        return validation_report
    
    async def watermark(
        self,
        data: Any,
        method: str = 'statistical',
        message: str = None,
        **kwargs
    ) -> Dict:
        """
        Apply watermarking to data.
        
        Args:
            data: Data to watermark
            method: Watermarking method
            message: Watermark message
            **kwargs: Additional watermarking parameters
            
        Returns:
            Watermarking result
        """
        if not self.initialized:
            await self.initialize()
            
        if method not in self.watermarkers:
            raise ValueError(f"Watermarker not found: {method}")
        
        watermarker = self.watermarkers[method]
        return await watermarker.embed(
            data=data,
            message=message or f'synthetic:{uuid.uuid4()}',
            **kwargs
        )
    
    async def verify_watermark(
        self,
        data: Any,
        method: str = 'statistical',
        **kwargs
    ) -> Dict:
        """
        Verify watermark in data.
        
        Args:
            data: Data to verify
            method: Watermarking method
            **kwargs: Additional verification parameters
            
        Returns:
            Verification result
        """
        if not self.initialized:
            await self.initialize()
            
        if method not in self.watermarkers:
            raise ValueError(f"Watermarker not found: {method}")
        
        watermarker = self.watermarkers[method]
        return await watermarker.verify(data=data, **kwargs)
    
    def get_pipelines(self) -> List[Dict]:
        """Get list of registered pipelines."""
        return [
            {
                'id': pipeline.id,
                'name': pipeline.config.get('name', pipeline.id),
                'generator_type': pipeline.generator_type,
                'data_type': pipeline.data_type,
                'created': pipeline.created_at,
                'last_used': pipeline.last_used
            }
            for pipeline in self.pipelines.values()
        ]
    
    def get_active_tasks(self) -> List[Dict]:
        """Get list of active tasks."""
        return [
            {
                'id': task['id'],
                'status': task['status'],
                'pipeline_id': task['pipeline_id'],
                'start_time': task['start_time'],
                'num_records': task['num_records']
            }
            for task in self.active_tasks.values()
            if task['status'] == 'running'
        ]
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics.copy()
    
    async def cleanup(self) -> None:
        """Clean up Guardian resources."""
        self.logger.info("Starting Guardian cleanup...")
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            if task['status'] == 'running':
                task['status'] = 'cancelled'
                self.logger.info(f"Cancelled task {task_id}")
        
        # Cleanup pipelines
        for pipeline in self.pipelines.values():
            await pipeline.cleanup()
        
        # Cleanup components
        for generator in self.generators.values():
            await generator.cleanup()
        
        for validator in self.validators.values():
            await validator.cleanup()
        
        for watermarker in self.watermarkers.values():
            await watermarker.cleanup()
        
        # Clear collections
        self.pipelines.clear()
        self.active_tasks.clear()
        self.generators.clear()
        self.validators.clear()
        self.watermarkers.clear()
        
        self.initialized = False
        self.logger.info("Guardian cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
