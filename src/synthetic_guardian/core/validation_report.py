"""
Validation Report - Comprehensive validation reporting for synthetic data
"""

import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from ..utils.logger import get_logger


class ValidationStatus(Enum):
    """Validation status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ValidationSeverity(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidatorResult:
    """Individual validator result."""
    name: str
    status: ValidationStatus
    score: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    severity: ValidationSeverity = ValidationSeverity.MEDIUM
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_passed(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED
    
    def is_failed(self) -> bool:
        """Check if validation failed."""
        return self.status == ValidationStatus.FAILED
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0 or self.status == ValidationStatus.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'score': self.score,
            'message': self.message,
            'details': self.details,
            'errors': self.errors,
            'warnings': self.warnings,
            'severity': self.severity.value,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'passed': self.is_passed()
        }


@dataclass
class ValidationMetrics:
    """Validation metrics summary."""
    total_validators: int = 0
    passed_validators: int = 0
    failed_validators: int = 0
    warning_validators: int = 0
    skipped_validators: int = 0
    overall_score: float = 0.0
    average_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    total_execution_time: float = 0.0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_validators == 0:
            return 0.0
        return self.passed_validators / self.total_validators
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_validators == 0:
            return 0.0
        return self.failed_validators / self.total_validators
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_validators': self.total_validators,
            'passed_validators': self.passed_validators,
            'failed_validators': self.failed_validators,
            'warning_validators': self.warning_validators,
            'skipped_validators': self.skipped_validators,
            'overall_score': self.overall_score,
            'average_score': self.average_score,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'pass_rate': self.pass_rate,
            'failure_rate': self.failure_rate,
            'total_execution_time': self.total_execution_time,
            'critical_issues': self.critical_issues,
            'high_issues': self.high_issues,
            'medium_issues': self.medium_issues,
            'low_issues': self.low_issues
        }


class ValidationReport:
    """
    Comprehensive validation report for synthetic data.
    
    Provides detailed reporting of all validation results including
    scores, errors, warnings, and comprehensive metrics.
    """
    
    def __init__(self, task_id: Optional[str] = None, logger=None):
        """Initialize validation report."""
        self.task_id = task_id
        self.timestamp = time.time()
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Results storage
        self.validator_results: Dict[str, ValidatorResult] = {}
        self.metrics = ValidationMetrics()
        
        # Report metadata
        self.report_id = f"validation_{int(self.timestamp)}"
        self.metadata: Dict[str, Any] = {}
        self.completed = False
        
    def add_validator_result(self, validator_name: str, result: Union[Dict, ValidatorResult]) -> None:
        """
        Add a validator result to the report.
        
        Args:
            validator_name: Name of the validator
            result: Validator result (dict or ValidatorResult)
        """
        if isinstance(result, dict):
            # Convert dict to ValidatorResult
            validator_result = ValidatorResult(
                name=validator_name,
                status=ValidationStatus(result.get('status', 'failed')),
                score=result.get('score', 0.0),
                message=result.get('message', ''),
                details=result.get('details', {}),
                errors=result.get('errors', []),
                warnings=result.get('warnings', []),
                severity=ValidationSeverity(result.get('severity', 'medium')),
                execution_time=result.get('execution_time', 0.0),
                metadata=result.get('metadata', {})
            )
        else:
            validator_result = result
        
        self.validator_results[validator_name] = validator_result
        self._update_metrics()
        
        self.logger.debug(f"Added validator result: {validator_name} -> {validator_result.status.value}")
    
    def _update_metrics(self) -> None:
        """Update validation metrics."""
        results = list(self.validator_results.values())
        
        if not results:
            return
        
        # Basic counts
        self.metrics.total_validators = len(results)
        self.metrics.passed_validators = sum(1 for r in results if r.is_passed())
        self.metrics.failed_validators = sum(1 for r in results if r.is_failed())
        self.metrics.warning_validators = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        self.metrics.skipped_validators = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        
        # Score calculations
        scores = [r.score for r in results if r.status != ValidationStatus.SKIPPED]
        if scores:
            self.metrics.average_score = sum(scores) / len(scores)
            self.metrics.min_score = min(scores)
            self.metrics.max_score = max(scores)
            
            # Overall score weighted by severity
            weighted_scores = []
            weights = []
            for r in results:
                if r.status != ValidationStatus.SKIPPED:
                    weight = {
                        ValidationSeverity.CRITICAL: 4,
                        ValidationSeverity.HIGH: 3,
                        ValidationSeverity.MEDIUM: 2,
                        ValidationSeverity.LOW: 1,
                        ValidationSeverity.INFO: 0.5
                    }[r.severity]
                    weighted_scores.append(r.score * weight)
                    weights.append(weight)
            
            if weights:
                self.metrics.overall_score = sum(weighted_scores) / sum(weights)
        
        # Execution time
        self.metrics.total_execution_time = sum(r.execution_time for r in results)
        
        # Issue counts by severity
        self.metrics.critical_issues = sum(
            1 for r in results 
            if r.is_failed() and r.severity == ValidationSeverity.CRITICAL
        )
        self.metrics.high_issues = sum(
            1 for r in results 
            if r.is_failed() and r.severity == ValidationSeverity.HIGH
        )
        self.metrics.medium_issues = sum(
            1 for r in results 
            if r.is_failed() and r.severity == ValidationSeverity.MEDIUM
        )
        self.metrics.low_issues = sum(
            1 for r in results 
            if r.is_failed() and r.severity == ValidationSeverity.LOW
        )
    
    def is_passed(self) -> bool:
        """Check if overall validation passed."""
        if not self.validator_results:
            return False
        
        # Must have no critical or high severity failures
        critical_failures = any(
            r.is_failed() and r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]
            for r in self.validator_results.values()
        )
        
        return not critical_failures and self.metrics.pass_rate >= 0.8
    
    def get_failed_validators(self) -> List[ValidatorResult]:
        """Get list of failed validators."""
        return [r for r in self.validator_results.values() if r.is_failed()]
    
    def get_warnings(self) -> List[ValidatorResult]:
        """Get list of validators with warnings."""
        return [r for r in self.validator_results.values() if r.has_warnings()]
    
    def get_critical_issues(self) -> List[ValidatorResult]:
        """Get critical validation issues."""
        return [
            r for r in self.validator_results.values()
            if r.is_failed() and r.severity == ValidationSeverity.CRITICAL
        ]
    
    def get_all_errors(self) -> List[str]:
        """Get all validation errors."""
        errors = []
        for validator_result in self.validator_results.values():
            for error in validator_result.errors:
                errors.append(f"{validator_result.name}: {error}")
        return errors
    
    def get_all_warnings(self) -> List[str]:
        """Get all validation warnings."""
        warnings = []
        for validator_result in self.validator_results.values():
            for warning in validator_result.warnings:
                warnings.append(f"{validator_result.name}: {warning}")
        return warnings
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'report_id': self.report_id,
            'task_id': self.task_id,
            'timestamp': self.timestamp,
            'completed': self.completed,
            'overall_passed': self.is_passed(),
            'metrics': self.metrics.to_dict(),
            'critical_issues_count': self.metrics.critical_issues,
            'total_errors': sum(len(r.errors) for r in self.validator_results.values()),
            'total_warnings': sum(len(r.warnings) for r in self.validator_results.values()),
            'validators': list(self.validator_results.keys())
        }
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append(f"Validation Report ({self.report_id})")
        lines.append("=" * 50)
        lines.append(f"Task ID: {self.task_id}")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}")
        lines.append(f"Overall Status: {'PASSED' if self.is_passed() else 'FAILED'}")
        lines.append("")
        
        # Metrics
        lines.append("METRICS")
        lines.append("-" * 20)
        lines.append(f"Total Validators: {self.metrics.total_validators}")
        lines.append(f"Passed: {self.metrics.passed_validators}")
        lines.append(f"Failed: {self.metrics.failed_validators}")
        lines.append(f"Warnings: {self.metrics.warning_validators}")
        lines.append(f"Pass Rate: {self.metrics.pass_rate:.1%}")
        lines.append(f"Overall Score: {self.metrics.overall_score:.2f}")
        lines.append(f"Execution Time: {self.metrics.total_execution_time:.2f}s")
        lines.append("")
        
        # Critical Issues
        critical_issues = self.get_critical_issues()
        if critical_issues:
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 20)
            for issue in critical_issues:
                lines.append(f"- {issue.name}: {issue.message}")
                for error in issue.errors:
                    lines.append(f"  * {error}")
            lines.append("")
        
        # Validator Results
        lines.append("VALIDATOR RESULTS")
        lines.append("-" * 20)
        for name, result in self.validator_results.items():
            status_symbol = "✓" if result.is_passed() else "✗"
            lines.append(f"{status_symbol} {name}: {result.score:.2f} ({result.status.value})")
            if result.message:
                lines.append(f"  Message: {result.message}")
            if result.errors:
                lines.append(f"  Errors: {', '.join(result.errors)}")
            if result.warnings:
                lines.append(f"  Warnings: {', '.join(result.warnings)}")
        
        return "\n".join(lines)
    
    def save_report(
        self, 
        path: Union[str, Path], 
        format: str = 'json',
        include_details: bool = True
    ) -> None:
        """
        Save validation report to file.
        
        Args:
            path: Output file path
            format: Output format (json, txt)
            include_details: Whether to include detailed results
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            self._save_as_json(path, include_details)
        elif format.lower() == 'txt':
            self._save_as_text(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_as_json(self, path: Path, include_details: bool) -> None:
        """Save as JSON file."""
        report_data = {
            'report_id': self.report_id,
            'task_id': self.task_id,
            'timestamp': self.timestamp,
            'completed': self.completed,
            'overall_passed': self.is_passed(),
            'metrics': self.metrics.to_dict(),
            'summary': self.get_summary(),
            'metadata': self.metadata
        }
        
        if include_details:
            report_data['validator_results'] = {
                name: result.to_dict()
                for name, result in self.validator_results.items()
            }
        
        with open(path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _save_as_text(self, path: Path) -> None:
        """Save as text file."""
        with open(path, 'w') as f:
            f.write(self.generate_text_report())
    
    def finalize(self) -> None:
        """Finalize the validation report."""
        self.completed = True
        self._update_metrics()
        self.logger.info(f"Validation report finalized: {self.report_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'task_id': self.task_id,
            'timestamp': self.timestamp,
            'completed': self.completed,
            'overall_passed': self.is_passed(),
            'metrics': self.metrics.to_dict(),
            'validator_results': {
                name: result.to_dict()
                for name, result in self.validator_results.items()
            },
            'metadata': self.metadata
        }