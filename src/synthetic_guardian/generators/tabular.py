"""
Tabular Data Generator - Synthetic tabular data generation using statistical models
"""

import asyncio
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .base import BaseGenerator, GeneratorConfig, GenerationResult
from ..utils.logger import get_logger


@dataclass
class TabularGeneratorConfig(GeneratorConfig):
    """Configuration for tabular generator."""
    backend: str = "gaussian_copula"  # gaussian_copula, ctgan, tvae
    epochs: int = 100
    batch_size: int = 500
    learning_rate: float = 0.001
    categorical_columns: List[str] = None
    sequence_columns: List[str] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.categorical_columns is None:
            self.categorical_columns = []
        if self.sequence_columns is None:
            self.sequence_columns = []
        if self.constraints is None:
            self.constraints = {}


class TabularGenerator(BaseGenerator):
    """
    Tabular Data Generator - Generates synthetic tabular data using various backends.
    
    Supports multiple generation backends:
    - Gaussian Copula (statistical)
    - CTGAN (GAN-based)
    - TVAE (VAE-based)
    """
    
    def __init__(self, config: Optional[TabularGeneratorConfig] = None, logger=None):
        """Initialize tabular generator."""
        if config is None:
            config = TabularGeneratorConfig(
                name="tabular_generator",
                type="tabular"
            )
        
        super().__init__(config, logger)
        self.model = None
        self.fitted = False
        self.reference_data = None
        self.column_types = {}
        
    async def initialize(self) -> None:
        """Initialize the tabular generator."""
        if self.initialized:
            return
        
        self.logger.info("Initializing TabularGenerator...")
        
        try:
            # Initialize based on backend
            backend = getattr(self.config, 'backend', 'gaussian_copula')
            
            # Check if config has parameters with backend info
            if hasattr(self.config, 'parameters') and self.config.parameters:
                backend = self.config.parameters.get('backend', backend)
            
            if backend == "gaussian_copula":
                await self._initialize_gaussian_copula()
            elif backend == "ctgan":
                await self._initialize_ctgan()
            elif backend == "tvae":
                await self._initialize_tvae()
            elif backend == "simple":
                await self._initialize_simple()
            else:
                # Fallback to simple statistical generation
                await self._initialize_simple()
            
            self.initialized = True
            self.logger.info(f"TabularGenerator initialized with {backend} backend")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TabularGenerator: {str(e)}")
            # Fallback to simple generation
            await self._initialize_simple()
            self.initialized = True
            self.logger.warning("Initialized with simple backend as fallback")
    
    async def _initialize_gaussian_copula(self) -> None:
        """Initialize Gaussian Copula model."""
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            self.model = GaussianCopulaSynthesizer()
            self.backend_type = "sdv_gaussian"
        except ImportError:
            self.logger.warning("SDV not available, falling back to simple backend")
            await self._initialize_simple()
    
    async def _initialize_ctgan(self) -> None:
        """Initialize CTGAN model."""
        try:
            from sdv.single_table import CTGANSynthesizer
            self.model = CTGANSynthesizer(
                epochs=self.config.epochs,
                batch_size=self.config.batch_size
            )
            self.backend_type = "sdv_ctgan"
        except ImportError:
            self.logger.warning("SDV/CTGAN not available, falling back to simple backend")
            await self._initialize_simple()
    
    async def _initialize_tvae(self) -> None:
        """Initialize TVAE model."""
        try:
            from sdv.single_table import TVAESynthesizer
            self.model = TVAESynthesizer(
                epochs=self.config.epochs,
                batch_size=self.config.batch_size
            )
            self.backend_type = "sdv_tvae"
        except ImportError:
            self.logger.warning("SDV/TVAE not available, falling back to simple backend")
            await self._initialize_simple()
    
    async def _initialize_simple(self) -> None:
        """Initialize simple statistical generator (fallback)."""
        self.model = None
        self.backend_type = "simple"
    
    async def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the generator to reference data.
        
        Args:
            data: Reference dataframe to learn from
        """
        if not self.initialized:
            await self.initialize()
        
        self.logger.info(f"Fitting TabularGenerator to {len(data)} records...")
        start_time = time.time()
        
        try:
            self.reference_data = data.copy()
            self._analyze_columns(data)
            
            if self.model and hasattr(self.model, 'fit'):
                # Use SDV model
                self.model.fit(data)
                self.fitted = True
                self.logger.info("Model fitting completed using SDV")
            else:
                # Simple statistical fitting
                await self._fit_simple(data)
                self.fitted = True
                self.logger.info("Model fitting completed using simple statistics")
            
            fit_time = time.time() - start_time
            self.logger.info(f"Fitting completed in {fit_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Model fitting failed: {str(e)}")
            # Fallback to simple fitting
            await self._fit_simple(data)
            self.fitted = True
            self.logger.warning("Used simple statistical fitting as fallback")
    
    def _analyze_columns(self, data: pd.DataFrame) -> None:
        """Analyze column types and characteristics."""
        self.column_types = {}
        
        for column in data.columns:
            dtype = data[column].dtype
            
            if dtype == 'object' or dtype.name == 'category':
                # Categorical column
                unique_values = data[column].nunique()
                if unique_values / len(data) < 0.1:  # Less than 10% unique values
                    self.column_types[column] = {
                        'type': 'categorical',
                        'categories': data[column].unique().tolist()
                    }
                else:
                    self.column_types[column] = {'type': 'text'}
            elif np.issubdtype(dtype, np.integer):
                self.column_types[column] = {
                    'type': 'integer',
                    'min': int(data[column].min()),
                    'max': int(data[column].max()),
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std())
                }
            elif np.issubdtype(dtype, np.floating):
                self.column_types[column] = {
                    'type': 'float',
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std())
                }
            elif np.issubdtype(dtype, np.datetime64):
                self.column_types[column] = {
                    'type': 'datetime',
                    'min': data[column].min(),
                    'max': data[column].max()
                }
            else:
                self.column_types[column] = {'type': 'unknown'}
    
    async def _fit_simple(self, data: pd.DataFrame) -> None:
        """Simple statistical fitting."""
        # Store basic statistics for simple generation
        self.simple_stats = {}
        
        for column in data.columns:
            col_info = self.column_types[column]
            
            if col_info['type'] == 'categorical':
                # Store category probabilities
                value_counts = data[column].value_counts(normalize=True)
                self.simple_stats[column] = {
                    'type': 'categorical',
                    'categories': value_counts.index.tolist(),
                    'probabilities': value_counts.values.tolist()
                }
            elif col_info['type'] in ['integer', 'float']:
                # Store distribution parameters
                self.simple_stats[column] = {
                    'type': col_info['type'],
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max())
                }
            else:
                # Store sample values
                self.simple_stats[column] = {
                    'type': 'sample',
                    'values': data[column].dropna().sample(min(1000, len(data))).tolist()
                }
    
    async def generate(
        self,
        num_records: int,
        seed: Optional[int] = None,
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate synthetic tabular data.
        
        Args:
            num_records: Number of records to generate
            seed: Random seed for reproducibility
            conditions: Conditional generation parameters
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with synthetic data
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Validate parameters
        issues = await self._validate_generation_params(num_records, **kwargs)
        if issues:
            raise ValueError(f"Generation parameter validation failed: {'; '.join(issues)}")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        try:
            self.logger.info(f"Generating {num_records} synthetic tabular records...")
            
            if self.fitted and self.model and hasattr(self.model, 'sample'):
                # Use fitted SDV model
                synthetic_data = self.model.sample(num_records)
                self.logger.debug("Generated data using fitted SDV model")
            elif self.fitted and hasattr(self, 'simple_stats'):
                # Use simple statistical generation
                synthetic_data = await self._generate_simple(num_records)
                self.logger.debug("Generated data using simple statistics")
            else:
                # Generate from schema if no reference data
                synthetic_data = await self._generate_from_schema(num_records)
                self.logger.debug("Generated data from schema")
            
            generation_time = time.time() - start_time
            
            # Validate generated data
            validation_issues = await self._validate_generated_data(synthetic_data, num_records)
            if validation_issues:
                self.logger.warning(f"Generated data validation issues: {validation_issues}")
            
            # Update statistics
            await self._update_statistics(generation_time, num_records)
            
            # Create result
            result = GenerationResult(
                data=synthetic_data,
                metadata={
                    'backend': self.backend_type,
                    'generation_time': generation_time,
                    'num_records': len(synthetic_data),
                    'seed': seed,
                    'conditions': conditions or {},
                    'fitted': self.fitted,
                    'column_types': self.column_types,
                    'validation_issues': validation_issues
                }
            )
            
            self.logger.info(
                f"Generated {len(synthetic_data)} records in {generation_time:.2f}s "
                f"using {self.backend_type} backend"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data generation failed: {str(e)}")
            raise
    
    async def _generate_simple(self, num_records: int) -> pd.DataFrame:
        """Generate data using simple statistical methods."""
        data = {}
        
        for column, stats in self.simple_stats.items():
            if stats['type'] == 'categorical':
                # Sample from categorical distribution
                data[column] = np.random.choice(
                    stats['categories'],
                    size=num_records,
                    p=stats['probabilities']
                )
            elif stats['type'] in ['integer', 'float']:
                # Generate from normal distribution with bounds
                values = np.random.normal(
                    stats['mean'],
                    stats['std'],
                    size=num_records
                )
                # Clip to observed bounds
                values = np.clip(values, stats['min'], stats['max'])
                
                if stats['type'] == 'integer':
                    values = np.round(values).astype(int)
                
                data[column] = values
            else:
                # Sample from observed values
                data[column] = np.random.choice(
                    stats['values'],
                    size=num_records,
                    replace=True
                )
        
        return pd.DataFrame(data)
    
    async def _generate_from_schema(self, num_records: int) -> pd.DataFrame:
        """Generate data from schema definition."""
        if not self.config.schema:
            # Generate simple demo data
            return pd.DataFrame({
                'id': range(1, num_records + 1),
                'value': np.random.normal(0, 1, num_records),
                'category': np.random.choice(['A', 'B', 'C'], num_records)
            })
        
        data = {}
        
        for column_name, column_spec in self.config.schema.items():
            if isinstance(column_spec, str):
                # Simple type specification
                data[column_name] = await self._generate_column_from_type(
                    column_spec, num_records
                )
            elif isinstance(column_spec, dict):
                # Detailed specification
                data[column_name] = await self._generate_column_from_spec(
                    column_spec, num_records
                )
        
        return pd.DataFrame(data)
    
    async def _generate_column_from_type(self, column_type: str, num_records: int) -> List:
        """Generate column from simple type specification."""
        if column_type == 'integer':
            return np.random.randint(0, 1000, num_records).tolist()
        elif column_type == 'float':
            return np.random.normal(0, 1, num_records).tolist()
        elif column_type == 'categorical':
            return np.random.choice(['A', 'B', 'C', 'D'], num_records).tolist()
        elif column_type == 'boolean':
            return np.random.choice([True, False], num_records).tolist()
        elif column_type == 'text':
            return [f"text_{i}" for i in range(num_records)]
        else:
            return [f"value_{i}" for i in range(num_records)]
    
    async def _generate_column_from_spec(self, spec: Dict, num_records: int) -> List:
        """Generate column from detailed specification."""
        column_type = spec.get('type', 'string')
        
        if column_type == 'integer':
            min_val = spec.get('min', 0)
            max_val = spec.get('max', 1000)
            return np.random.randint(min_val, max_val + 1, num_records).tolist()
        elif column_type == 'float':
            min_val = spec.get('min', 0.0)
            max_val = spec.get('max', 1.0)
            return np.random.uniform(min_val, max_val, num_records).tolist()
        elif column_type == 'categorical':
            categories = spec.get('categories', ['A', 'B', 'C'])
            return np.random.choice(categories, num_records).tolist()
        else:
            return [f"generated_{i}" for i in range(num_records)]
    
    def validate_config(self) -> List[str]:
        """Validate generator configuration."""
        issues = []
        
        if not isinstance(self.config.epochs, int) or self.config.epochs <= 0:
            issues.append("epochs must be a positive integer")
        
        if not isinstance(self.config.batch_size, int) or self.config.batch_size <= 0:
            issues.append("batch_size must be a positive integer")
        
        if self.config.backend not in ["gaussian_copula", "ctgan", "tvae", "simple"]:
            issues.append("backend must be one of: gaussian_copula, ctgan, tvae, simple")
        
        return issues
    
    async def cleanup(self) -> None:
        """Clean up generator resources."""
        self.logger.info("Cleaning up TabularGenerator...")
        
        self.model = None
        self.fitted = False
        self.reference_data = None
        self.column_types.clear()
        
        if hasattr(self, 'simple_stats'):
            delattr(self, 'simple_stats')
        
        self.initialized = False
        self.logger.info("TabularGenerator cleanup completed")