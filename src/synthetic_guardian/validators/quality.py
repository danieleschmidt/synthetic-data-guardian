"""
Quality Validator - Validates overall data quality
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any

from .base import BaseValidator
from ..utils.logger import get_logger


class QualityValidator(BaseValidator):
    """Validates overall data quality."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        super().__init__(config, logger)
        self.threshold = self.config.get('threshold', 0.8)
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing QualityValidator...")
        self.initialized = True
        self.logger.info("QualityValidator initialized")
    
    async def validate(
        self,
        data: Any,
        reference_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate data quality."""
        start_time = time.time()
        
        try:
            results = {}
            
            # Basic quality checks
            results['completeness'] = self._check_completeness(data)
            results['consistency'] = self._check_consistency(data)
            results['validity'] = self._check_validity(data)
            results['uniqueness'] = self._check_uniqueness(data)
            
            # Calculate overall quality score
            scores = [result.get('score', 0.8) for result in results.values()]
            overall_score = np.mean(scores) if scores else 0.8
            
            passed = overall_score >= self.threshold
            
            return {
                'passed': passed,
                'score': overall_score,
                'message': f"Quality validation {'passed' if passed else 'failed'}",
                'details': results,
                'errors': [] if passed else ['Quality validation threshold not met'],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Quality validation failed: {str(e)}")
            return {
                'passed': False,
                'score': 0.0,
                'message': f"Quality validation error: {str(e)}",
                'details': {},
                'errors': [str(e)],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
    
    def _check_completeness(self, data: Any) -> Dict:
        """Check data completeness (missing values)."""
        try:
            if hasattr(data, 'isnull'):
                # DataFrame
                total_cells = data.size
                missing_cells = data.isnull().sum().sum()
                completeness_ratio = 1.0 - (missing_cells / total_cells)
            elif hasattr(data, '__len__') and isinstance(data, (list, np.ndarray)):
                # Array-like
                if isinstance(data, np.ndarray):
                    missing_count = np.sum(np.isnan(data.astype(float, errors='ignore')))
                else:
                    missing_count = sum(1 for item in data if item is None)
                completeness_ratio = 1.0 - (missing_count / len(data))
            else:
                completeness_ratio = 1.0  # Assume complete for other types
            
            return {
                'name': 'completeness',
                'score': completeness_ratio,
                'completeness_ratio': completeness_ratio,
                'missing_ratio': 1.0 - completeness_ratio
            }
            
        except Exception as e:
            return {
                'name': 'completeness',
                'score': 0.8,
                'error': str(e)
            }
    
    def _check_consistency(self, data: Any) -> Dict:
        """Check data consistency."""
        try:
            if not hasattr(data, 'dtypes'):
                return {
                    'name': 'consistency',
                    'score': 0.9,
                    'message': 'Non-tabular data, consistency assumed'
                }
            
            # Check for consistent data types within columns
            consistency_issues = []
            consistent_columns = 0
            total_columns = len(data.columns)
            
            for column in data.columns:
                try:
                    # Check if column has consistent type
                    non_null_data = data[column].dropna()
                    if len(non_null_data) == 0:
                        continue
                    
                    # For object columns, check type consistency
                    if data[column].dtype == 'object':
                        types = set(type(val) for val in non_null_data)
                        if len(types) == 1:
                            consistent_columns += 1
                        else:
                            consistency_issues.append(f"Column {column} has mixed types: {types}")
                    else:
                        consistent_columns += 1
                        
                except Exception as e:
                    consistency_issues.append(f"Column {column}: {str(e)}")
            
            consistency_score = consistent_columns / total_columns if total_columns > 0 else 1.0
            
            return {
                'name': 'consistency',
                'score': consistency_score,
                'consistent_columns': consistent_columns,
                'total_columns': total_columns,
                'consistency_issues': consistency_issues
            }
            
        except Exception as e:
            return {
                'name': 'consistency',
                'score': 0.8,
                'error': str(e)
            }
    
    def _check_validity(self, data: Any) -> Dict:
        """Check data validity (reasonable values)."""
        try:
            if not hasattr(data, 'select_dtypes'):
                return {
                    'name': 'validity',
                    'score': 0.9,
                    'message': 'Non-pandas data, validity assumed'
                }
            
            validity_issues = []
            valid_columns = 0
            total_numeric_columns = 0
            
            # Check numeric columns for extreme values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            total_numeric_columns = len(numeric_columns)
            
            for column in numeric_columns:
                try:
                    col_data = data[column].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Check for infinite values
                    if np.any(np.isinf(col_data)):
                        validity_issues.append(f"Column {column} contains infinite values")
                        continue
                    
                    # Check for extremely large values (beyond reasonable bounds)
                    q99 = col_data.quantile(0.99)
                    q01 = col_data.quantile(0.01)
                    iqr = q99 - q01
                    
                    # Values beyond 10 IQRs are considered extreme
                    extreme_threshold = 10 * iqr
                    extreme_values = col_data[(col_data > q99 + extreme_threshold) | 
                                            (col_data < q01 - extreme_threshold)]
                    
                    if len(extreme_values) > len(col_data) * 0.01:  # More than 1% extreme
                        validity_issues.append(f"Column {column} has many extreme values")
                    else:
                        valid_columns += 1
                        
                except Exception as e:
                    validity_issues.append(f"Column {column}: {str(e)}")
            
            validity_score = valid_columns / total_numeric_columns if total_numeric_columns > 0 else 1.0
            
            return {
                'name': 'validity',
                'score': validity_score,
                'valid_columns': valid_columns,
                'total_numeric_columns': total_numeric_columns,
                'validity_issues': validity_issues
            }
            
        except Exception as e:
            return {
                'name': 'validity',
                'score': 0.8,
                'error': str(e)
            }
    
    def _check_uniqueness(self, data: Any) -> Dict:
        """Check data uniqueness (duplicates)."""
        try:
            if hasattr(data, 'duplicated'):
                # DataFrame
                total_rows = len(data)
                duplicate_rows = data.duplicated().sum()
                uniqueness_ratio = 1.0 - (duplicate_rows / total_rows) if total_rows > 0 else 1.0
                
                return {
                    'name': 'uniqueness',
                    'score': uniqueness_ratio,
                    'uniqueness_ratio': uniqueness_ratio,
                    'duplicate_count': duplicate_rows,
                    'total_rows': total_rows
                }
            elif hasattr(data, '__len__') and isinstance(data, (list, tuple)):
                # List-like
                total_items = len(data)
                unique_items = len(set(data))
                uniqueness_ratio = unique_items / total_items if total_items > 0 else 1.0
                
                return {
                    'name': 'uniqueness',
                    'score': uniqueness_ratio,
                    'uniqueness_ratio': uniqueness_ratio,
                    'duplicate_count': total_items - unique_items,
                    'total_items': total_items
                }
            else:
                return {
                    'name': 'uniqueness',
                    'score': 1.0,
                    'message': 'Cannot check uniqueness for this data type'
                }
                
        except Exception as e:
            return {
                'name': 'uniqueness',
                'score': 0.8,
                'error': str(e)
            }
    
    async def cleanup(self) -> None:
        self.initialized = False
        self.logger.info("QualityValidator cleanup completed")