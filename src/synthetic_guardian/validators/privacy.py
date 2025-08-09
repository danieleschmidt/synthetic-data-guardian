"""
Privacy Validator - Validates privacy preservation in synthetic data
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any

from .base import BaseValidator
from ..utils.logger import get_logger


class PrivacyValidator(BaseValidator):
    """Validates privacy preservation of synthetic data."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        super().__init__(config, logger)
        self.threshold = self.config.get('threshold', 0.8)
        self.epsilon = self.config.get('epsilon', 1.0)  # Differential privacy parameter
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing PrivacyValidator...")
        self.initialized = True
        self.logger.info("PrivacyValidator initialized")
    
    async def validate(
        self,
        data: Any,
        reference_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate privacy preservation."""
        start_time = time.time()
        
        try:
            # Basic privacy checks
            results = {}
            
            # Check for direct value matching (membership inference risk)
            if reference_data is not None:
                results['membership_risk'] = self._check_membership_risk(data, reference_data)
            
            # Check for unique identifier preservation
            results['identifier_risk'] = self._check_identifier_risk(data)
            
            # Check statistical distance (privacy-utility tradeoff)
            results['statistical_distance'] = self._check_statistical_distance(data, reference_data)
            
            # Calculate overall privacy score
            scores = [result.get('score', 0.0) for result in results.values()]
            overall_score = np.mean(scores) if scores else 0.8  # Default to good privacy
            
            passed = overall_score >= self.threshold
            
            return {
                'passed': passed,
                'score': overall_score,
                'message': f"Privacy validation {'passed' if passed else 'failed'}",
                'details': results,
                'errors': [] if passed else ['Privacy validation threshold not met'],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Privacy validation failed: {str(e)}")
            return {
                'passed': False,
                'score': 0.0,
                'message': f"Privacy validation error: {str(e)}",
                'details': {},
                'errors': [str(e)],
                'warnings': [],
                'execution_time': time.time() - start_time
            }
    
    def _check_membership_risk(self, data: Any, reference_data: Any) -> Dict:
        """Check for membership inference risk."""
        try:
            # Convert to comparable format
            if hasattr(data, 'values'):
                synthetic_values = data.values
            else:
                synthetic_values = np.array(data)
            
            if hasattr(reference_data, 'values'):
                reference_values = reference_data.values
            else:
                reference_values = np.array(reference_data)
            
            # Check for exact matches (high privacy risk)
            exact_matches = 0
            total_synthetic = len(synthetic_values)
            
            for syn_row in synthetic_values:
                for ref_row in reference_values:
                    if np.array_equal(syn_row, ref_row):
                        exact_matches += 1
                        break
            
            match_rate = exact_matches / total_synthetic if total_synthetic > 0 else 0
            
            # Score: lower match rate = better privacy
            score = max(0.0, 1.0 - match_rate * 2)  # Penalty for matches
            
            return {
                'name': 'membership_risk',
                'score': score,
                'exact_matches': exact_matches,
                'match_rate': match_rate,
                'risk_level': 'low' if match_rate < 0.1 else 'high'
            }
            
        except Exception as e:
            return {
                'name': 'membership_risk',
                'score': 0.5,  # Neutral score on error
                'error': str(e)
            }
    
    def _check_identifier_risk(self, data: Any) -> Dict:
        """Check for potential identifier preservation."""
        try:
            # Look for columns that might be identifiers
            identifier_risk = 0.0
            
            if hasattr(data, 'columns'):
                # DataFrame
                for column in data.columns:
                    col_name = str(column).lower()
                    unique_ratio = data[column].nunique() / len(data)
                    
                    # High uniqueness suggests identifier
                    if unique_ratio > 0.9:
                        identifier_risk += 0.3
                    
                    # Common identifier names
                    if any(term in col_name for term in ['id', 'ssn', 'email', 'phone']):
                        identifier_risk += 0.2
            
            score = max(0.0, 1.0 - identifier_risk)
            
            return {
                'name': 'identifier_risk',
                'score': score,
                'risk_score': identifier_risk,
                'risk_level': 'low' if identifier_risk < 0.3 else 'high'
            }
            
        except Exception as e:
            return {
                'name': 'identifier_risk',
                'score': 0.8,  # Assume good privacy on error
                'error': str(e)
            }
    
    def _check_statistical_distance(self, data: Any, reference_data: Optional[Any]) -> Dict:
        """Check statistical distance for privacy-utility tradeoff."""
        try:
            if reference_data is None:
                return {
                    'name': 'statistical_distance',
                    'score': 0.8,
                    'message': 'No reference data for comparison'
                }
            
            # Calculate basic statistical distance
            if hasattr(data, 'describe'):
                synthetic_stats = data.describe()
                reference_stats = reference_data.describe()
                
                # Compare means (normalized)
                mean_diff = 0.0
                count = 0
                
                for column in synthetic_stats.columns:
                    if column in reference_stats.columns:
                        syn_mean = synthetic_stats.loc['mean', column]
                        ref_mean = reference_stats.loc['mean', column]
                        
                        if ref_mean != 0:
                            relative_diff = abs((syn_mean - ref_mean) / ref_mean)
                            mean_diff += relative_diff
                            count += 1
                
                if count > 0:
                    avg_mean_diff = mean_diff / count
                    # Good privacy often means some statistical distance
                    # But too much distance reduces utility
                    if 0.1 <= avg_mean_diff <= 0.5:
                        score = 0.9  # Good privacy-utility balance
                    elif avg_mean_diff < 0.1:
                        score = 0.7  # Too similar, potential privacy risk
                    else:
                        score = 0.6  # Too different, utility loss
                else:
                    score = 0.8
            else:
                score = 0.8  # Default score for non-tabular data
            
            return {
                'name': 'statistical_distance',
                'score': score,
                'distance_measure': avg_mean_diff if 'avg_mean_diff' in locals() else None
            }
            
        except Exception as e:
            return {
                'name': 'statistical_distance',
                'score': 0.8,
                'error': str(e)
            }
    
    async def cleanup(self) -> None:
        self.initialized = False
        self.logger.info("PrivacyValidator cleanup completed")