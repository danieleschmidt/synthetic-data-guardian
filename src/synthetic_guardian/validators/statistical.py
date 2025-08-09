"""
Statistical Validator - Validates statistical properties of synthetic data
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any
from scipy import stats
import warnings

from .base import BaseValidator
from ..utils.logger import get_logger


class StatisticalValidator(BaseValidator):
    """Validates statistical fidelity of synthetic data against reference data."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        super().__init__(config, logger)
        self.threshold = self.config.get('threshold', 0.8)
        self.metrics = self.config.get('metrics', ['ks_test', 'correlation', 'mean_diff'])
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing StatisticalValidator...")
        self.initialized = True
        self.logger.info("StatisticalValidator initialized")
    
    async def validate(
        self,
        data: Any,
        reference_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate statistical properties."""
        start_time = time.time()
        
        try:
            # Convert to numpy arrays if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            if reference_data is not None:
                if hasattr(reference_data, 'values'):
                    ref_array = reference_data.values
                else:
                    ref_array = np.array(reference_data)
            else:
                # Basic statistics without reference
                return await self._validate_basic_stats(data_array)
            
            # Perform statistical tests
            results = {}
            
            if 'ks_test' in self.metrics:
                results['ks_test'] = self._kolmogorov_smirnov_test(data_array, ref_array)
            
            if 'correlation' in self.metrics and data_array.ndim > 1:
                results['correlation'] = self._correlation_test(data_array, ref_array)
            
            if 'mean_diff' in self.metrics:
                results['mean_diff'] = self._mean_difference_test(data_array, ref_array)
            
            # Calculate overall score
            scores = [result['score'] for result in results.values()]
            overall_score = np.mean(scores) if scores else 0.0
            
            passed = overall_score >= self.threshold
            
            return {
                'passed': passed,
                'score': overall_score,
                'message': f"Statistical validation {'passed' if passed else 'failed'}",
                'details': results,
                'errors': [] if passed else ['Statistical validation threshold not met'],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Statistical validation failed: {str(e)}")
            return {
                'passed': False,
                'score': 0.0,
                'message': f"Validation error: {str(e)}",
                'details': {},
                'errors': [str(e)],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
    
    def _kolmogorov_smirnov_test(self, data: np.ndarray, reference: np.ndarray) -> Dict:
        """Perform Kolmogorov-Smirnov test."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if data.ndim == 1:
                    statistic, p_value = stats.ks_2samp(data, reference)
                    score = 1.0 - statistic  # Higher statistic = worse fit
                else:
                    # For multivariate data, test each column
                    statistics = []
                    for i in range(data.shape[1]):
                        stat, _ = stats.ks_2samp(data[:, i], reference[:, i])
                        statistics.append(stat)
                    statistic = np.mean(statistics)
                    score = 1.0 - statistic
            
            return {
                'name': 'ks_test',
                'score': max(0.0, score),
                'statistic': statistic,
                'details': {'method': 'Kolmogorov-Smirnov'}
            }
        except Exception as e:
            return {
                'name': 'ks_test',
                'score': 0.0,
                'error': str(e)
            }
    
    def _correlation_test(self, data: np.ndarray, reference: np.ndarray) -> Dict:
        """Test correlation structure."""
        try:
            if data.shape[1] < 2:
                return {'name': 'correlation', 'score': 1.0, 'message': 'Single column data'}
            
            # Calculate correlation matrices
            data_corr = np.corrcoef(data.T)
            ref_corr = np.corrcoef(reference.T)
            
            # Calculate similarity (using correlation coefficient)
            corr_diff = np.abs(data_corr - ref_corr)
            score = 1.0 - np.mean(corr_diff)
            
            return {
                'name': 'correlation',
                'score': max(0.0, score),
                'correlation_difference': np.mean(corr_diff),
                'details': {'method': 'correlation_matrix_comparison'}
            }
        except Exception as e:
            return {
                'name': 'correlation',
                'score': 0.0,
                'error': str(e)
            }
    
    def _mean_difference_test(self, data: np.ndarray, reference: np.ndarray) -> Dict:
        """Test mean differences."""
        try:
            data_mean = np.mean(data, axis=0)
            ref_mean = np.mean(reference, axis=0)
            
            # Relative difference
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = np.abs((data_mean - ref_mean) / ref_mean)
                rel_diff = np.nan_to_num(rel_diff)
            
            # Score based on relative difference (closer to 0 is better)
            mean_rel_diff = np.mean(rel_diff)
            score = 1.0 / (1.0 + mean_rel_diff)  # Score between 0 and 1
            
            return {
                'name': 'mean_diff',
                'score': score,
                'mean_relative_difference': mean_rel_diff,
                'details': {'method': 'relative_mean_difference'}
            }
        except Exception as e:
            return {
                'name': 'mean_diff',
                'score': 0.0,
                'error': str(e)
            }
    
    async def _validate_basic_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Basic statistical validation without reference data."""
        try:
            # Check for basic statistical properties
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            
            # Basic score based on data quality
            score = 1.0
            issues = []
            
            if has_nan:
                score -= 0.3
                issues.append("Data contains NaN values")
            
            if has_inf:
                score -= 0.3
                issues.append("Data contains infinite values")
            
            # Check variance (data should have some variation)
            if data.ndim == 1:
                variance = np.var(data)
                if variance == 0:
                    score -= 0.2
                    issues.append("Data has zero variance")
            else:
                variances = np.var(data, axis=0)
                if np.any(variances == 0):
                    score -= 0.2
                    issues.append("Some columns have zero variance")
            
            passed = score >= self.threshold and len(issues) == 0
            
            return {
                'passed': passed,
                'score': max(0.0, score),
                'message': f"Basic statistical validation {'passed' if passed else 'failed'}",
                'details': {
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'shape': data.shape
                },
                'errors': issues if not passed else [],
                'warnings': []
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'message': f"Basic validation error: {str(e)}",
                'details': {},
                'errors': [str(e)],
                'warnings': []
            }
    
    async def cleanup(self) -> None:
        self.initialized = False
        self.logger.info("StatisticalValidator cleanup completed")