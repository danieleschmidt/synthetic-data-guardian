"""
Generation Result - Container for synthetic data generation results
"""

import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import get_logger


@dataclass
class ValidationReport:
    """Validation report for synthetic data."""
    task_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    overall_score: float = 0.0
    passed: bool = False
    validator_results: Dict[str, Dict] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_validator_result(self, validator_name: str, result: Dict) -> None:
        """Add a validator result."""
        self.validator_results[validator_name] = result
        self._update_summary()
    
    def _update_summary(self) -> None:
        """Update overall summary."""
        if not self.validator_results:
            return
        
        # Calculate overall score
        scores = [
            result.get('score', 0.0) 
            for result in self.validator_results.values()
            if 'score' in result
        ]
        
        if scores:
            self.overall_score = sum(scores) / len(scores)
        
        # Check if all validators passed
        self.passed = all(
            result.get('passed', False)
            for result in self.validator_results.values()
        )
        
        # Update summary
        self.summary = {
            'total_validators': len(self.validator_results),
            'passed_validators': sum(
                1 for result in self.validator_results.values()
                if result.get('passed', False)
            ),
            'failed_validators': sum(
                1 for result in self.validator_results.values()
                if not result.get('passed', True)
            ),
            'average_score': self.overall_score,
            'validation_time': time.time() - self.timestamp
        }
    
    def get_failed_validators(self) -> List[str]:
        """Get list of failed validators."""
        return [
            name for name, result in self.validator_results.items()
            if not result.get('passed', True)
        ]
    
    def get_errors(self) -> List[str]:
        """Get all validation errors."""
        errors = []
        for name, result in self.validator_results.items():
            if 'errors' in result:
                errors.extend([f"{name}: {error}" for error in result['errors']])
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'timestamp': self.timestamp,
            'overall_score': self.overall_score,
            'passed': self.passed,
            'validator_results': self.validator_results,
            'summary': self.summary
        }


@dataclass 
class GenerationResult:
    """Result from synthetic data generation."""
    task_id: str
    pipeline_id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_report: Optional[ValidationReport] = None
    watermark_info: Optional[Dict[str, Any]] = None
    lineage_id: Optional[str] = None
    quality_score: float = 0.0
    privacy_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Post-initialization processing."""
        self._calculate_scores()
    
    def _calculate_scores(self) -> None:
        """Calculate quality and privacy scores from validation report."""
        if self.validation_report:
            # Quality score from statistical and quality validators
            quality_validators = ['statistical', 'quality', 'bias']
            quality_scores = []
            
            for validator_name in quality_validators:
                if validator_name in self.validation_report.validator_results:
                    result = self.validation_report.validator_results[validator_name]
                    if 'score' in result:
                        quality_scores.append(result['score'])
            
            if quality_scores:
                self.quality_score = sum(quality_scores) / len(quality_scores)
            
            # Privacy score from privacy validator
            if 'privacy' in self.validation_report.validator_results:
                privacy_result = self.validation_report.validator_results['privacy']
                self.privacy_score = privacy_result.get('score', 0.0)
    
    def is_valid(self) -> bool:
        """Check if generation result is valid."""
        return (
            self.data is not None and
            len(self.errors) == 0 and
            (self.validation_report is None or self.validation_report.passed)
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        summary = {
            'task_id': self.task_id,
            'pipeline_id': self.pipeline_id,
            'timestamp': self.timestamp,
            'is_valid': self.is_valid(),
            'quality_score': self.quality_score,
            'privacy_score': self.privacy_score,
            'has_validation': self.validation_report is not None,
            'has_watermark': self.watermark_info is not None,
            'has_lineage': self.lineage_id is not None,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
        
        # Add data info if available
        if hasattr(self.data, '__len__'):
            summary['record_count'] = len(self.data)
        
        # Add metadata info
        if self.metadata:
            summary.update({
                'generation_time': self.metadata.get('generation_time'),
                'generator_type': self.metadata.get('generator'),
                'data_type': self.metadata.get('data_type')
            })
        
        return summary
    
    def save_data(
        self, 
        path: Union[str, Path], 
        format: str = 'json',
        include_metadata: bool = True
    ) -> None:
        """
        Save generated data to file.
        
        Args:
            path: Output file path
            format: Output format (json, csv, parquet)
            include_metadata: Whether to include metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            self._save_as_json(path, include_metadata)
        elif format.lower() == 'csv':
            self._save_as_csv(path, include_metadata)
        elif format.lower() == 'parquet':
            self._save_as_parquet(path, include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_as_json(self, path: Path, include_metadata: bool) -> None:
        """Save as JSON file."""
        output = {
            'data': self._serialize_data_for_json(),
        }
        
        if include_metadata:
            output.update({
                'metadata': self.metadata,
                'task_id': self.task_id,
                'pipeline_id': self.pipeline_id,
                'timestamp': self.timestamp,
                'quality_score': self.quality_score,
                'privacy_score': self.privacy_score,
                'lineage_id': self.lineage_id
            })
            
            if self.validation_report:
                output['validation_report'] = self.validation_report.to_dict()
            
            if self.watermark_info:
                output['watermark_info'] = self.watermark_info
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
    
    def _save_as_csv(self, path: Path, include_metadata: bool) -> None:
        """Save as CSV file."""
        try:
            import pandas as pd
            
            # Convert data to DataFrame if possible
            if isinstance(self.data, list):
                df = pd.DataFrame(self.data)
            elif hasattr(self.data, 'to_csv'):
                df = self.data
            else:
                raise ValueError("Data cannot be converted to CSV format")
            
            # Add metadata columns if requested
            if include_metadata:
                df['_task_id'] = self.task_id
                df['_pipeline_id'] = self.pipeline_id
                df['_timestamp'] = self.timestamp
                df['_quality_score'] = self.quality_score
                df['_privacy_score'] = self.privacy_score
            
            df.to_csv(path, index=False)
            
        except ImportError:
            raise RuntimeError("pandas is required for CSV export")
    
    def _save_as_parquet(self, path: Path, include_metadata: bool) -> None:
        """Save as Parquet file."""
        try:
            import pandas as pd
            
            # Convert data to DataFrame if possible
            if isinstance(self.data, list):
                df = pd.DataFrame(self.data)
            elif hasattr(self.data, 'to_parquet'):
                df = self.data
            else:
                raise ValueError("Data cannot be converted to Parquet format")
            
            # Add metadata as Parquet metadata
            metadata_dict = {}
            if include_metadata:
                metadata_dict = {
                    'task_id': str(self.task_id),
                    'pipeline_id': str(self.pipeline_id),
                    'timestamp': str(self.timestamp),
                    'quality_score': str(self.quality_score),
                    'privacy_score': str(self.privacy_score),
                    'lineage_id': str(self.lineage_id) if self.lineage_id else None
                }
            
            df.to_parquet(path, index=False, metadata=metadata_dict)
            
        except ImportError:
            raise RuntimeError("pandas and pyarrow are required for Parquet export")
    
    def _serialize_data_for_json(self) -> Any:
        """Serialize data for JSON output."""
        # Handle common data types
        if hasattr(self.data, 'to_dict'):
            return self.data.to_dict()
        elif hasattr(self.data, 'tolist'):
            return self.data.tolist()
        elif isinstance(self.data, (list, dict, str, int, float, bool)):
            return self.data
        else:
            # Try to convert to string as fallback
            return str(self.data)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'task_id': self.task_id,
            'pipeline_id': self.pipeline_id,
            'data': self._serialize_data_for_json(),
            'metadata': self.metadata,
            'quality_score': self.quality_score,
            'privacy_score': self.privacy_score,
            'lineage_id': self.lineage_id,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp,
            'is_valid': self.is_valid()
        }
        
        if self.validation_report:
            result['validation_report'] = self.validation_report.to_dict()
        
        if self.watermark_info:
            result['watermark_info'] = self.watermark_info
        
        return result