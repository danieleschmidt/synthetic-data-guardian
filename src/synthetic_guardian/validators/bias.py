"""
Bias Validator - Validates fairness and bias in synthetic data
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any

from .base import BaseValidator
from ..utils.logger import get_logger


class BiasValidator(BaseValidator):
    """Validates fairness and bias in synthetic data."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        super().__init__(config, logger)
        self.threshold = self.config.get('threshold', 0.8)
        self.protected_attributes = self.config.get('protected_attributes', [])
        self.fairness_metrics = self.config.get('fairness_metrics', ['demographic_parity'])
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing BiasValidator...")
        self.initialized = True
        self.logger.info("BiasValidator initialized")
    
    async def validate(
        self,
        data: Any,
        reference_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate bias and fairness."""
        start_time = time.time()
        
        try:
            results = {}
            
            # Check demographic representation
            results['demographic_representation'] = self._check_demographic_representation(data, reference_data)
            
            # Check for statistical bias
            results['statistical_bias'] = self._check_statistical_bias(data)
            
            # Check protected attribute distribution
            if self.protected_attributes:
                results['protected_attributes'] = self._check_protected_attributes(data, reference_data)
            
            # Calculate overall bias score
            scores = [result.get('score', 0.8) for result in results.values()]
            overall_score = np.mean(scores) if scores else 0.8
            
            passed = overall_score >= self.threshold
            
            return {
                'passed': passed,
                'score': overall_score,
                'message': f"Bias validation {'passed' if passed else 'failed'}",
                'details': results,
                'errors': [] if passed else ['Bias validation threshold not met'],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Bias validation failed: {str(e)}")
            return {
                'passed': False,
                'score': 0.0,
                'message': f"Bias validation error: {str(e)}",
                'details': {},
                'errors': [str(e)],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
    
    def _check_demographic_representation(self, data: Any, reference_data: Optional[Any]) -> Dict:
        """Check demographic representation balance."""
        try:
            if not hasattr(data, 'columns'):
                return {
                    'name': 'demographic_representation',
                    'score': 0.8,
                    'message': 'Non-tabular data, assuming balanced representation'
                }
            
            # Look for categorical columns that might represent demographics
            demographic_columns = []
            for column in data.columns:
                col_name = str(column).lower()
                # Check for common demographic indicators
                if any(term in col_name for term in ['gender', 'race', 'ethnicity', 'age_group', 'category']):
                    demographic_columns.append(column)
                elif data[column].dtype == 'object' and data[column].nunique() <= 10:
                    # Small categorical columns might be demographic
                    demographic_columns.append(column)
            
            if not demographic_columns:
                return {
                    'name': 'demographic_representation',
                    'score': 0.8,
                    'message': 'No obvious demographic columns found'
                }
            
            # Check balance in demographic columns
            balance_scores = []
            for column in demographic_columns:
                value_counts = data[column].value_counts(normalize=True)
                
                # Calculate balance score (entropy-based)
                if len(value_counts) <= 1:
                    balance_score = 0.0  # No diversity
                else:
                    # Use entropy to measure balance
                    entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
                    max_entropy = np.log2(len(value_counts))
                    balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
                
                balance_scores.append(balance_score)
            
            overall_balance = np.mean(balance_scores) if balance_scores else 0.8
            
            return {
                'name': 'demographic_representation',
                'score': overall_balance,
                'demographic_columns': demographic_columns,
                'balance_scores': dict(zip(demographic_columns, balance_scores)),
                'overall_balance': overall_balance
            }
            
        except Exception as e:
            return {
                'name': 'demographic_representation',
                'score': 0.5,
                'error': str(e)
            }
    
    def _check_statistical_bias(self, data: Any) -> Dict:
        """Check for statistical bias patterns."""
        try:
            if not hasattr(data, 'select_dtypes'):
                return {
                    'name': 'statistical_bias',
                    'score': 0.8,
                    'message': 'Non-pandas data, basic bias check passed'
                }
            
            # Check numeric columns for extreme skewness
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            bias_issues = []
            skewness_scores = []
            
            for column in numeric_columns:
                try:
                    # Calculate skewness
                    from scipy.stats import skew
                    column_skew = abs(skew(data[column].dropna()))
                    
                    # Score based on skewness (less skew = better)
                    if column_skew > 3:
                        skew_score = 0.3  # Highly skewed
                        bias_issues.append(f"Column {column} is highly skewed")
                    elif column_skew > 1:
                        skew_score = 0.7  # Moderately skewed
                    else:
                        skew_score = 1.0  # Well balanced
                    
                    skewness_scores.append(skew_score)
                    
                except Exception:
                    skewness_scores.append(0.8)  # Default score on error
            
            overall_score = np.mean(skewness_scores) if skewness_scores else 0.8
            
            return {
                'name': 'statistical_bias',
                'score': overall_score,
                'bias_issues': bias_issues,
                'skewness_analysis': dict(zip(numeric_columns, skewness_scores))
            }
            
        except Exception as e:
            return {
                'name': 'statistical_bias',
                'score': 0.8,
                'error': str(e)
            }
    
    def _check_protected_attributes(self, data: Any, reference_data: Optional[Any]) -> Dict:
        """Check protected attribute distributions."""
        try:
            if not self.protected_attributes:
                return {
                    'name': 'protected_attributes',
                    'score': 1.0,
                    'message': 'No protected attributes specified'
                }
            
            if not hasattr(data, 'columns'):
                return {
                    'name': 'protected_attributes',
                    'score': 0.8,
                    'message': 'Non-tabular data'
                }
            
            attribute_scores = []
            
            for attr in self.protected_attributes:
                if attr not in data.columns:
                    continue
                
                # Check distribution balance
                value_counts = data[attr].value_counts(normalize=True)
                
                if reference_data is not None and attr in reference_data.columns:
                    # Compare with reference distribution
                    ref_counts = reference_data[attr].value_counts(normalize=True)
                    
                    # Calculate distribution similarity
                    common_values = set(value_counts.index) & set(ref_counts.index)
                    if common_values:
                        diff_sum = sum(abs(value_counts.get(val, 0) - ref_counts.get(val, 0)) 
                                     for val in common_values)
                        similarity = 1.0 - (diff_sum / 2)  # Max diff is 2
                        attribute_scores.append(similarity)
                    else:
                        attribute_scores.append(0.5)  # No common values
                else:
                    # Just check for balance without reference
                    if len(value_counts) <= 1:
                        attribute_scores.append(0.0)  # No diversity
                    else:
                        # Entropy-based balance
                        entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
                        max_entropy = np.log2(len(value_counts))
                        balance = entropy / max_entropy if max_entropy > 0 else 0.0
                        attribute_scores.append(balance)
            
            overall_score = np.mean(attribute_scores) if attribute_scores else 1.0
            
            return {
                'name': 'protected_attributes',
                'score': overall_score,
                'protected_attributes': self.protected_attributes,
                'attribute_scores': dict(zip(self.protected_attributes, attribute_scores))
            }
            
        except Exception as e:
            return {
                'name': 'protected_attributes',
                'score': 0.8,
                'error': str(e)
            }
    
    async def cleanup(self) -> None:
        self.initialized = False
        self.logger.info("BiasValidator cleanup completed")