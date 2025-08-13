"""
Time Series Data Generator - Synthetic time series data generation
"""

import asyncio
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import BaseGenerator, GeneratorConfig, GenerationResult
from ..utils.logger import get_logger


@dataclass
class TimeSeriesGeneratorConfig(GeneratorConfig):
    """Configuration for time series generator."""
    sequence_length: int = 100
    features: List[str] = None
    sampling_frequency: str = "1min"  # pandas frequency string
    trend_strength: float = 0.1
    seasonality_strength: float = 0.2
    noise_level: float = 0.1
    start_date: Optional[str] = None
    patterns: List[str] = None  # trend, seasonal, cyclical, random
    
    def __post_init__(self):
        super().__post_init__()
        if self.features is None:
            self.features = ["value"]
        if self.patterns is None:
            self.patterns = ["trend", "seasonal", "noise"]


class TimeSeriesGenerator(BaseGenerator):
    """
    Time Series Data Generator - Generates synthetic time series data with various patterns.
    
    Supports:
    - Trend patterns (linear, exponential)
    - Seasonal patterns (daily, weekly, yearly)
    - Cyclical patterns
    - Random walk
    - ARIMA-like patterns
    """
    
    def __init__(self, config: Optional[TimeSeriesGeneratorConfig] = None, logger=None):
        """Initialize time series generator."""
        if config is None:
            config = TimeSeriesGeneratorConfig(
                name="timeseries_generator",
                type="timeseries"
            )
        
        super().__init__(config, logger)
        self.fitted_patterns = {}
        self.reference_stats = {}
        
    async def initialize(self) -> None:
        """Initialize the time series generator."""
        if self.initialized:
            return
        
        self.logger.info("Initializing TimeSeriesGenerator...")
        
        try:
            # Set up default patterns if none specified
            patterns = getattr(self.config, 'patterns', ["trend", "seasonal", "noise"])
            if hasattr(self.config, 'parameters') and self.config.parameters:
                patterns = self.config.parameters.get('patterns', patterns)
            
            self.patterns = patterns
            
            # Initialize pattern generators
            self.pattern_generators = {
                "trend": self._generate_trend,
                "seasonal": self._generate_seasonal,
                "cyclical": self._generate_cyclical,
                "noise": self._generate_noise,
                "random_walk": self._generate_random_walk
            }
            
            self.initialized = True
            self.logger.info("TimeSeriesGenerator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TimeSeriesGenerator: {str(e)}")
            raise
    
    async def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the generator to reference time series data.
        
        Args:
            data: Reference time series dataframe
        """
        if not self.initialized:
            await self.initialize()
        
        self.logger.info(f"Fitting TimeSeriesGenerator to {len(data)} records...")
        
        try:
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                elif 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    # Create synthetic datetime index
                    data.index = pd.date_range(
                        start='2020-01-01',
                        periods=len(data),
                        freq=getattr(self.config, 'sampling_frequency', 'D')
                    )
            
            # Analyze patterns in the data
            await self._analyze_patterns(data)
            
            # Store reference statistics
            self.reference_stats = {}
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    self.reference_stats[column] = {
                        'mean': float(data[column].mean()),
                        'std': float(data[column].std()),
                        'min': float(data[column].min()),
                        'max': float(data[column].max()),
                        'trend': self._estimate_trend(data[column]),
                        'seasonality': self._estimate_seasonality(data[column])
                    }
            
            self.fitted = True
            self.logger.info("TimeSeriesGenerator fitting completed")
            
        except Exception as e:
            self.logger.error(f"Failed to fit TimeSeriesGenerator: {str(e)}")
            raise
    
    async def _analyze_patterns(self, data: pd.DataFrame) -> None:
        """Analyze patterns in the reference data."""
        self.fitted_patterns = {}
        
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                series = data[column].dropna()
                
                self.fitted_patterns[column] = {
                    'trend_slope': self._estimate_trend(series),
                    'seasonal_period': self._estimate_seasonal_period(series),
                    'volatility': float(series.std() / series.mean()) if series.mean() != 0 else 1.0,
                    'autocorrelation': self._estimate_autocorrelation(series)
                }
    
    def _estimate_trend(self, series: pd.Series) -> float:
        """Estimate linear trend slope."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _estimate_seasonal_period(self, series: pd.Series) -> int:
        """Estimate seasonal period using autocorrelation."""
        if len(series) < 10:
            return 24  # Default to daily pattern
        
        # Calculate autocorrelations for different lags
        max_lag = min(len(series) // 4, 168)  # Up to weekly pattern
        autocorrs = []
        
        for lag in range(1, max_lag):
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append((lag, abs(corr)))
        
        if autocorrs:
            # Find the lag with highest autocorrelation
            best_lag = max(autocorrs, key=lambda x: x[1])[0]
            return best_lag
        
        return 24  # Default
    
    def _estimate_seasonality(self, series: pd.Series) -> float:
        """Estimate seasonality strength."""
        if len(series) < 20:
            return 0.1
        
        period = self._estimate_seasonal_period(series)
        if period >= len(series):
            return 0.1
        
        # Calculate seasonal variation
        seasonal_means = []
        for i in range(period):
            seasonal_values = series[i::period]
            if len(seasonal_values) > 0:
                seasonal_means.append(seasonal_values.mean())
        
        if len(seasonal_means) > 1:
            seasonal_std = np.std(seasonal_means)
            overall_std = series.std()
            return min(seasonal_std / overall_std, 1.0) if overall_std > 0 else 0.1
        
        return 0.1
    
    def _estimate_autocorrelation(self, series: pd.Series, lag: int = 1) -> float:
        """Estimate first-order autocorrelation."""
        if len(series) <= lag:
            return 0.0
        
        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    async def generate(
        self,
        num_records: int,
        seed: Optional[int] = None,
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate synthetic time series data.
        
        Args:
            num_records: Number of time points to generate
            seed: Random seed for reproducibility
            conditions: Generation conditions
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with synthetic time series data
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
            self.logger.info(f"Generating {num_records} time series data points...")
            
            # Create datetime index
            start_date = kwargs.get('start_date', getattr(self.config, 'start_date', pd.Timestamp.now()))
            if start_date is None:
                start_date = '2024-01-01'
            
            date_index = pd.date_range(
                start=start_date,
                periods=num_records,
                freq=getattr(self.config, 'sampling_frequency', 'D')
            )
            
            # Generate data for each feature
            data = {}
            
            for feature in getattr(self.config, 'features', ['value']):
                series_data = await self._generate_feature_series(
                    feature, num_records, conditions
                )
                data[feature] = series_data
            
            # Create DataFrame
            synthetic_data = pd.DataFrame(data, index=date_index)
            
            generation_time = time.time() - start_time
            
            # Validate generated data
            validation_issues = await self._validate_generated_data(synthetic_data, num_records)
            
            # Update statistics
            await self._update_statistics(generation_time, num_records)
            
            # Create result
            result = GenerationResult(
                data=synthetic_data,
                metadata={
                    'generation_time': generation_time,
                    'num_records': len(synthetic_data),
                    'features': getattr(self.config, 'features', ['value']),
                    'patterns': getattr(self, 'patterns', ['trend', 'seasonal', 'noise']),
                    'seed': seed,
                    'conditions': conditions or {},
                    'sampling_frequency': getattr(self.config, 'sampling_frequency', 'D'),
                    'start_date': start_date,
                    'validation_issues': validation_issues
                }
            )
            
            self.logger.info(
                f"Generated time series with {len(synthetic_data)} points "
                f"in {generation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Time series generation failed: {str(e)}")
            raise
    
    async def _generate_feature_series(
        self,
        feature_name: str,
        num_records: int,
        conditions: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate a single feature time series."""
        # Initialize with zeros
        series = np.zeros(num_records)
        
        # Get reference stats if available
        if feature_name in self.reference_stats:
            stats = self.reference_stats[feature_name]
            base_level = stats['mean']
            scale = stats['std']
        else:
            base_level = 0.0
            scale = 1.0
        
        # Generate each pattern component
        for pattern in getattr(self, 'patterns', ['trend', 'seasonal', 'noise']):
            if pattern in self.pattern_generators:
                component = await self.pattern_generators[pattern](
                    num_records, feature_name, conditions
                )
                series += component
        
        # Scale and shift to match reference statistics
        if scale > 0:
            series = (series - np.mean(series)) / np.std(series) * scale
        series += base_level
        
        return series
    
    async def _generate_trend(
        self,
        num_records: int,
        feature_name: str,
        conditions: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate trend component."""
        # Get trend slope from fitted patterns or config
        if feature_name in self.fitted_patterns:
            slope = self.fitted_patterns[feature_name]['trend_slope']
        else:
            slope = getattr(self.config, 'trend_strength', 0.1)
        
        # Linear trend
        time_points = np.arange(num_records)
        trend = slope * time_points
        
        return trend * getattr(self.config, 'trend_strength', 0.1)
    
    async def _generate_seasonal(
        self,
        num_records: int,
        feature_name: str,
        conditions: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate seasonal component."""
        # Get seasonal period
        if feature_name in self.fitted_patterns:
            period = self.fitted_patterns[feature_name]['seasonal_period']
        else:
            period = 24  # Default daily pattern
        
        # Generate multiple seasonal cycles
        time_points = np.arange(num_records)
        seasonal = np.zeros(num_records)
        
        # Primary seasonal pattern
        seasonal += np.sin(2 * np.pi * time_points / period)
        
        # Add harmonic
        seasonal += 0.3 * np.sin(4 * np.pi * time_points / period)
        
        return seasonal * getattr(self.config, 'seasonality_strength', 1.0)
    
    async def _generate_cyclical(
        self,
        num_records: int,
        feature_name: str,
        conditions: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate cyclical component."""
        # Longer-term cycles
        time_points = np.arange(num_records)
        cycle_length = num_records // 3  # Cycle over 1/3 of the series
        
        if cycle_length < 2:
            return np.zeros(num_records)
        
        cyclical = np.sin(2 * np.pi * time_points / cycle_length)
        
        return cyclical * 0.1  # Smaller amplitude than seasonal
    
    async def _generate_noise(
        self,
        num_records: int,
        feature_name: str,
        conditions: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate noise component."""
        noise = np.random.normal(0, 1, num_records)
        return noise * getattr(self.config, 'noise_level', 0.1)
    
    async def _generate_random_walk(
        self,
        num_records: int,
        feature_name: str,
        conditions: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate random walk component."""
        steps = np.random.normal(0, 0.01, num_records)
        random_walk = np.cumsum(steps)
        return random_walk
    
    def validate_config(self) -> List[str]:
        """Validate generator configuration."""
        issues = []
        
        sequence_length = getattr(self.config, 'sequence_length', 100)
        if sequence_length <= 0:
            issues.append("sequence_length must be positive")
        
        features = getattr(self.config, 'features', ['value'])
        if not features:
            issues.append("features list cannot be empty")
        
        trend_strength = getattr(self.config, 'trend_strength', 0.1)
        if trend_strength < 0:
            issues.append("trend_strength cannot be negative")
        
        seasonality_strength = getattr(self.config, 'seasonality_strength', 1.0)
        if seasonality_strength < 0:
            issues.append("seasonality_strength cannot be negative")
        
        noise_level = getattr(self.config, 'noise_level', 0.1)
        if noise_level < 0:
            issues.append("noise_level cannot be negative")
        
        # Validate sampling frequency
        try:
            pd.Timedelta(getattr(self.config, 'sampling_frequency', 'D'))
        except ValueError:
            issues.append("sampling_frequency must be a valid pandas frequency string")
        
        return issues
    
    async def cleanup(self) -> None:
        """Clean up generator resources."""
        self.logger.info("Cleaning up TimeSeriesGenerator...")
        
        self.fitted_patterns.clear()
        self.reference_stats.clear()
        
        if hasattr(self, 'pattern_generators'):
            self.pattern_generators.clear()
        
        self.initialized = False
        self.logger.info("TimeSeriesGenerator cleanup completed")