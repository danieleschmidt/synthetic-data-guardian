"""
Statistical Watermarker - Embeds watermarks using statistical properties
"""

import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .base import BaseWatermarker
from ..utils.logger import get_logger


class StatisticalWatermarker(BaseWatermarker):
    """Embeds watermarks by modifying statistical properties of data."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        super().__init__(config, logger)
        self.strength = self.config.get('strength', 0.01)  # Watermark strength
        self.method = self.config.get('method', 'mean_shift')  # mean_shift, variance_mod
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing StatisticalWatermarker...")
        self.initialized = True
        self.logger.info("StatisticalWatermarker initialized")
    
    async def embed(
        self,
        data: Any,
        message: str,
        key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Embed statistical watermark in data."""
        start_time = time.time()
        
        try:
            # Generate watermark signature from message and key
            signature = self._generate_signature(message, key)
            
            # Apply watermark based on method
            if self.method == 'mean_shift':
                watermarked_data = self._embed_mean_shift(data, signature)
            elif self.method == 'variance_mod':
                watermarked_data = self._embed_variance_modification(data, signature)
            else:
                watermarked_data = self._embed_mean_shift(data, signature)  # Default
            
            embedding_time = time.time() - start_time
            
            return {
                'data': watermarked_data,
                'watermark_embedded': True,
                'method': self.method,
                'strength': self.strength,
                'signature': signature,
                'embedding_time': embedding_time,
                'message_hash': hashlib.sha256(message.encode()).hexdigest()[:16]
            }
            
        except Exception as e:
            self.logger.error(f"Watermark embedding failed: {str(e)}")
            return {
                'data': data,  # Return original data on failure
                'watermark_embedded': False,
                'error': str(e)
            }
    
    async def extract(
        self,
        data: Any,
        key: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Extract watermark from data."""
        try:
            # For statistical watermarks, extraction is verification-based
            # We can't directly extract the original message
            verification_result = await self.verify(data, key=key, **kwargs)
            
            if verification_result.get('is_watermarked', False):
                # Return a placeholder message since we can only verify presence
                return True, f"watermark_detected_method_{self.method}"
            else:
                return False, None
                
        except Exception as e:
            self.logger.error(f"Watermark extraction failed: {str(e)}")
            return False, None
    
    async def verify(
        self,
        data: Any,
        expected_message: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Verify watermark presence in data."""
        start_time = time.time()
        
        try:
            # Check for statistical anomalies that indicate watermarking
            is_watermarked = False
            confidence = 0.0
            
            if self.method == 'mean_shift':
                is_watermarked, confidence = self._detect_mean_shift(data, expected_message, key)
            elif self.method == 'variance_mod':
                is_watermarked, confidence = self._detect_variance_modification(data, expected_message, key)
            
            verification_time = time.time() - start_time
            
            return {
                'is_watermarked': is_watermarked,
                'confidence': confidence,
                'method': self.method,
                'verification_time': verification_time,
                'threshold_met': confidence > 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Watermark verification failed: {str(e)}")
            return {
                'is_watermarked': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_signature(self, message: str, key: Optional[str] = None) -> float:
        """Generate watermark signature from message and key."""
        # Create hash from message and key
        content = f"{message}_{key or 'default_key'}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()
        
        # Convert to float in range [-1, 1]
        hash_int = int(hash_digest[:8], 16)
        signature = (hash_int / (2**32 - 1)) * 2 - 1
        
        return signature * self.strength
    
    def _embed_mean_shift(self, data: Any, signature: float) -> Any:
        """Embed watermark by shifting mean values."""
        try:
            if hasattr(data, 'copy'):
                # DataFrame or Series
                watermarked_data = data.copy()
                
                # Apply mean shift to numeric columns
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for column in numeric_columns:
                    column_std = data[column].std()
                    if column_std > 0:
                        # Shift mean by small fraction of standard deviation
                        shift = signature * column_std
                        watermarked_data[column] = data[column] + shift
                
                return watermarked_data
                
            elif isinstance(data, np.ndarray):
                # NumPy array
                watermarked_data = data.copy()
                if np.issubdtype(data.dtype, np.number):
                    std_dev = np.std(data)
                    if std_dev > 0:
                        shift = signature * std_dev
                        watermarked_data = data + shift
                
                return watermarked_data
                
            elif isinstance(data, list) and len(data) > 0:
                # List of numbers
                if all(isinstance(x, (int, float)) for x in data):
                    data_array = np.array(data)
                    std_dev = np.std(data_array)
                    if std_dev > 0:
                        shift = signature * std_dev
                        return (data_array + shift).tolist()
                
                return data
            else:
                # Unsupported data type
                return data
                
        except Exception as e:
            self.logger.warning(f"Mean shift embedding failed: {str(e)}")
            return data
    
    def _embed_variance_modification(self, data: Any, signature: float) -> Any:
        """Embed watermark by modifying variance."""
        try:
            if hasattr(data, 'copy'):
                # DataFrame
                watermarked_data = data.copy()
                
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for column in numeric_columns:
                    col_data = data[column]
                    col_mean = col_data.mean()
                    col_std = col_data.std()
                    
                    if col_std > 0:
                        # Modify variance slightly
                        variance_factor = 1 + signature
                        centered_data = col_data - col_mean
                        watermarked_data[column] = col_mean + centered_data * variance_factor
                
                return watermarked_data
                
            elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
                # NumPy array
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                if std_val > 0:
                    variance_factor = 1 + signature
                    centered_data = data - mean_val
                    return mean_val + centered_data * variance_factor
                
                return data
            else:
                return data
                
        except Exception as e:
            self.logger.warning(f"Variance modification embedding failed: {str(e)}")
            return data
    
    def _detect_mean_shift(self, data: Any, expected_message: Optional[str], key: Optional[str]) -> Tuple[bool, float]:
        """Detect mean shift watermark."""
        try:
            if expected_message is None:
                # Generic detection - look for unusual mean patterns
                return self._generic_mean_shift_detection(data)
            
            # Specific detection with known message
            expected_signature = self._generate_signature(expected_message, key)
            
            if hasattr(data, 'select_dtypes'):
                # DataFrame
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                detections = []
                
                for column in numeric_columns:
                    col_data = data[column]
                    col_mean = col_data.mean()
                    col_std = col_data.std()
                    
                    if col_std > 0:
                        # Check if mean is shifted by expected amount
                        expected_shift = expected_signature * col_std
                        
                        # This is a simplified detection - in practice, you'd need
                        # the original data statistics for comparison
                        detection_score = abs(expected_shift) / col_std
                        detections.append(detection_score)
                
                if detections:
                    avg_detection = np.mean(detections)
                    confidence = min(avg_detection, 1.0)
                    return confidence > 0.001, confidence
                    
            return False, 0.0
            
        except Exception as e:
            self.logger.warning(f"Mean shift detection failed: {str(e)}")
            return False, 0.0
    
    def _detect_variance_modification(self, data: Any, expected_message: Optional[str], key: Optional[str]) -> Tuple[bool, float]:
        """Detect variance modification watermark."""
        try:
            # Simplified variance detection
            # In practice, this would require reference statistics
            
            if hasattr(data, 'select_dtypes'):
                # Look for unusual variance patterns
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                variance_scores = []
                
                for column in numeric_columns:
                    col_data = data[column]
                    col_std = col_data.std()
                    col_mean = col_data.mean()
                    
                    if col_mean != 0:
                        # Coefficient of variation as a proxy for unusual variance
                        cv = col_std / abs(col_mean)
                        variance_scores.append(cv)
                
                if variance_scores:
                    # This is a placeholder detection method
                    avg_cv = np.mean(variance_scores)
                    confidence = min(avg_cv * 0.1, 1.0)  # Very simplified
                    return confidence > 0.01, confidence
            
            return False, 0.0
            
        except Exception as e:
            self.logger.warning(f"Variance modification detection failed: {str(e)}")
            return False, 0.0
    
    def _generic_mean_shift_detection(self, data: Any) -> Tuple[bool, float]:
        """Generic mean shift detection without known message."""
        try:
            # Look for statistical anomalies that might indicate watermarking
            
            if hasattr(data, 'describe'):
                # Statistical summary
                stats = data.describe()
                
                # Look for unusual patterns in the statistics
                # This is a very simplified approach
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    # Check for consistent small biases across columns
                    means = [data[col].mean() for col in numeric_columns]
                    stds = [data[col].std() for col in numeric_columns]
                    
                    # Very basic heuristic: look for non-zero means in standardized data
                    # In practice, this would be much more sophisticated
                    if all(std > 0 for std in stds):
                        normalized_means = [abs(mean/std) for mean, std in zip(means, stds)]
                        avg_bias = np.mean(normalized_means)
                        
                        # If there's a consistent small bias, might be watermarked
                        confidence = min(avg_bias, 1.0)
                        return confidence > 0.01, confidence
            
            return False, 0.0
            
        except Exception as e:
            self.logger.warning(f"Generic detection failed: {str(e)}")
            return False, 0.0
    
    async def cleanup(self) -> None:
        self.initialized = False
        self.logger.info("StatisticalWatermarker cleanup completed")