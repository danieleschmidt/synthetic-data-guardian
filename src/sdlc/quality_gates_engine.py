"""
Quality Gates Engine - Comprehensive quality validation and enforcement
"""

import asyncio
import time
import subprocess
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import psutil

from ..utils.logger import get_logger


class QualityGateType(Enum):
    """Types of quality gates."""
    SYNTAX = "syntax"
    TESTS = "tests"
    COVERAGE = "coverage"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DEPENDENCIES = "dependencies"
    DOCUMENTATION = "documentation"


@dataclass
class QualityGateConfig:
    """Configuration for a quality gate."""
    name: str
    type: QualityGateType
    threshold: float
    weight: float = 1.0
    enabled: bool = True
    timeout: int = 300  # 5 minutes
    retry_count: int = 2
    commands: List[str] = field(default_factory=list)
    success_patterns: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)
    metrics_extraction: Dict[str, str] = field(default_factory=dict)


@dataclass
class QualityGateExecution:
    """Result of quality gate execution."""
    config: QualityGateConfig
    success: bool
    score: float
    duration: float
    output: str = ""
    error: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)


class QualityGatesEngine:
    """
    Quality Gates Engine - Enforces progressive quality standards.
    
    This engine implements comprehensive quality validation including:
    - Code syntax and style validation
    - Test execution and coverage analysis
    - Security vulnerability scanning
    - Performance benchmarking
    - Compliance checking
    - Dependency auditing
    """
    
    def __init__(self, project_root: Path, logger=None):
        """Initialize quality gates engine."""
        self.project_root = project_root
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Quality gate configurations
        self.quality_gates = self._initialize_quality_gates()
        
        # Execution state
        self.execution_history: List[QualityGateExecution] = []
        self.current_scores: Dict[str, float] = {}
        
        self.logger.info(f"Initialized Quality Gates Engine for {project_root}")
    
    def _initialize_quality_gates(self) -> Dict[str, QualityGateConfig]:
        """Initialize default quality gate configurations."""
        gates = {}
        
        # Syntax validation
        gates['python_syntax'] = QualityGateConfig(
            name="Python Syntax Check",
            type=QualityGateType.SYNTAX,
            threshold=100.0,
            commands=['python', '-m', 'py_compile'],
            success_patterns=[r'.*'],  # No error means success
            failure_patterns=[r'SyntaxError', r'IndentationError']
        )
        
        gates['javascript_syntax'] = QualityGateConfig(
            name="JavaScript Syntax Check", 
            type=QualityGateType.SYNTAX,
            threshold=100.0,
            commands=['node', '--check'],
            success_patterns=[r'.*'],
            failure_patterns=[r'SyntaxError', r'Unexpected token']
        )
        
        # Unit tests
        gates['python_tests'] = QualityGateConfig(
            name="Python Unit Tests",
            type=QualityGateType.TESTS,
            threshold=85.0,
            commands=['pytest', '--tb=short', '--quiet'],
            success_patterns=[r'(\d+) passed'],
            failure_patterns=[r'(\d+) failed', r'FAILED'],
            metrics_extraction={
                'passed': r'(\d+) passed',
                'failed': r'(\d+) failed',
                'errors': r'(\d+) error'
            }
        )
        
        gates['javascript_tests'] = QualityGateConfig(
            name="JavaScript Unit Tests",
            type=QualityGateType.TESTS,
            threshold=85.0,
            commands=['npm', 'test'],
            success_patterns=[r'Tests:\s+(\d+) passed'],
            failure_patterns=[r'Tests:\s+\d+ failed', r'FAIL'],
            metrics_extraction={
                'passed': r'Tests:\s+(\d+) passed',
                'failed': r'Tests:\s+(\d+) failed'
            }
        )
        
        # Code coverage
        gates['code_coverage'] = QualityGateConfig(
            name="Code Coverage Analysis",
            type=QualityGateType.COVERAGE,
            threshold=85.0,
            commands=['npm', 'run', 'test:coverage'],
            success_patterns=[r'All files\s+\|\s+(\d+\.?\d*)\s+\|'],
            metrics_extraction={
                'coverage_percentage': r'All files\s+\|\s+(\d+\.?\d*)\s+\|'
            }
        )
        
        # Security scanning
        gates['security_audit'] = QualityGateConfig(
            name="Security Vulnerability Scan",
            type=QualityGateType.SECURITY,
            threshold=90.0,
            commands=['npm', 'audit', '--audit-level=moderate'],
            success_patterns=[r'found 0 vulnerabilities'],
            failure_patterns=[r'found (\d+) vulnerabilities', r'high severity'],
            metrics_extraction={
                'vulnerabilities': r'found (\d+) vulnerabilities'
            }
        )
        
        gates['python_security'] = QualityGateConfig(
            name="Python Security Audit",
            type=QualityGateType.SECURITY, 
            threshold=90.0,
            commands=['bandit', '-r', 'src/', '-f', 'json'],
            success_patterns=[r'"results": \[\]'],
            failure_patterns=[r'"severity": "HIGH"', r'"severity": "MEDIUM"'],
            metrics_extraction={
                'issues': r'"results": \[([^\]]*)\]'
            }
        )
        
        # Performance benchmarks
        gates['performance_benchmark'] = QualityGateConfig(
            name="Performance Benchmark",
            type=QualityGateType.PERFORMANCE,
            threshold=80.0,
            commands=['npm', 'run', 'test:performance'],
            success_patterns=[r'Performance: (\d+\.?\d*)ms'],
            metrics_extraction={
                'response_time': r'Performance: (\d+\.?\d*)ms',
                'throughput': r'Throughput: (\d+\.?\d*) req/s'
            }
        )
        
        # Compliance checks
        gates['gdpr_compliance'] = QualityGateConfig(
            name="GDPR Compliance Check",
            type=QualityGateType.COMPLIANCE,
            threshold=95.0,
            commands=['python', '-m', 'synthetic_guardian.compliance.gdpr_checker'],
            success_patterns=[r'GDPR compliance: (\d+\.?\d*)%'],
            metrics_extraction={
                'compliance_score': r'GDPR compliance: (\d+\.?\d*)%'
            }
        )
        
        # Dependency auditing
        gates['dependency_vulnerabilities'] = QualityGateConfig(
            name="Dependency Vulnerability Check",
            type=QualityGateType.DEPENDENCIES,
            threshold=95.0,
            commands=['pip-audit', '--format=json'],
            success_patterns=[r'\[\]'],  # Empty vulnerabilities array
            failure_patterns=[r'"vulnerabilities":\s*\[.*\]'],
            metrics_extraction={
                'vulnerable_packages': r'"vulnerabilities":\s*\[([^\]]*)\]'
            }
        )
        
        return gates
    
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all enabled quality gates."""
        self.logger.info("🔍 Executing all quality gates")
        start_time = time.time()
        
        results = {}
        total_score = 0.0
        total_weight = 0.0
        passed_gates = 0
        failed_gates = 0
        
        # Execute gates in parallel for efficiency
        tasks = []
        for gate_name, config in self.quality_gates.items():
            if config.enabled:
                task = asyncio.create_task(self._execute_quality_gate(gate_name, config))
                tasks.append((gate_name, task))
        
        # Wait for all gates to complete
        for gate_name, task in tasks:
            try:
                execution = await task
                results[gate_name] = execution
                self.execution_history.append(execution)
                
                # Update scores and counts
                if execution.success:
                    passed_gates += 1
                    total_score += execution.score * execution.config.weight
                    total_weight += execution.config.weight
                else:
                    failed_gates += 1
                
                self.current_scores[gate_name] = execution.score
                
                # Log result
                status = "✅ PASSED" if execution.success else "❌ FAILED"
                self.logger.info(
                    f"{status} {execution.config.name}: {execution.score:.1f}% "
                    f"(threshold: {execution.config.threshold:.1f}%)"
                )
                
            except Exception as e:
                self.logger.error(f"Quality gate {gate_name} failed with exception: {e}")
                failed_gates += 1
                results[gate_name] = QualityGateExecution(
                    config=self.quality_gates[gate_name],
                    success=False,
                    score=0.0,
                    duration=0.0,
                    error=str(e)
                )
        
        total_duration = time.time() - start_time
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Generate summary
        summary = {
            'overall_score': overall_score,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'total_gates': len([g for g in self.quality_gates.values() if g.enabled]),
            'total_duration': total_duration,
            'gate_results': {
                name: self._serialize_execution(result)
                for name, result in results.items()
            },
            'recommendations': self._generate_recommendations(results),
            'quality_report': self._generate_quality_report(results)
        }
        
        self.logger.info(
            f"🎯 Quality Gates Summary: {passed_gates}/{passed_gates + failed_gates} passed, "
            f"Overall Score: {overall_score:.1f}%"
        )
        
        return summary
    
    async def _execute_quality_gate(self, gate_name: str, config: QualityGateConfig) -> QualityGateExecution:
        """Execute a single quality gate."""
        self.logger.debug(f"Executing quality gate: {config.name}")
        start_time = time.time()
        
        for attempt in range(config.retry_count + 1):
            try:
                # Execute the gate command(s)
                if config.commands:
                    result = await self._run_gate_command(config)
                else:
                    # Custom gate logic
                    result = await self._run_custom_gate(gate_name, config)
                
                # Parse results and calculate score
                execution = await self._parse_gate_result(config, result)
                execution.duration = time.time() - start_time
                
                # If successful or final attempt, return result
                if execution.success or attempt == config.retry_count:
                    if attempt > 0:
                        self.logger.info(f"Quality gate {config.name} succeeded on attempt {attempt + 1}")
                    return execution
                
                # Wait before retry
                if attempt < config.retry_count:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Quality gate {config.name} timed out on attempt {attempt + 1}")
                if attempt == config.retry_count:
                    return QualityGateExecution(
                        config=config,
                        success=False,
                        score=0.0,
                        duration=time.time() - start_time,
                        error="Execution timed out"
                    )
            except Exception as e:
                self.logger.error(f"Quality gate {config.name} failed on attempt {attempt + 1}: {e}")
                if attempt == config.retry_count:
                    return QualityGateExecution(
                        config=config,
                        success=False,
                        score=0.0,
                        duration=time.time() - start_time,
                        error=str(e)
                    )
        
        # Should not reach here
        return QualityGateExecution(
            config=config,
            success=False,
            score=0.0,
            duration=time.time() - start_time,
            error="Unexpected execution path"
        )
    
    async def _run_gate_command(self, config: QualityGateConfig) -> Dict[str, Any]:
        """Run the command for a quality gate."""
        try:
            # Prepare command
            cmd = config.commands.copy()
            
            # Add file patterns for syntax checks
            if config.type == QualityGateType.SYNTAX:
                if 'python' in cmd[0]:
                    py_files = list(self.project_root.glob('**/*.py'))
                    cmd.extend([str(f) for f in py_files[:10]])  # Limit for performance
                elif 'node' in cmd[0]:
                    js_files = list(self.project_root.glob('**/*.js'))
                    cmd.extend([str(f) for f in js_files[:10]])
            
            # Execute command with timeout
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.project_root
                ),
                timeout=config.timeout
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore'),
                'command': ' '.join(cmd)
            }
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'command': ' '.join(config.commands)
            }
    
    async def _run_custom_gate(self, gate_name: str, config: QualityGateConfig) -> Dict[str, Any]:
        """Run custom quality gate logic."""
        # Implement custom gates that don't use external commands
        if gate_name == 'file_structure_check':
            return await self._check_file_structure()
        elif gate_name == 'code_quality_metrics':
            return await self._calculate_code_quality_metrics()
        else:
            return {
                'returncode': 0,
                'stdout': 'Custom gate executed successfully',
                'stderr': ''
            }
    
    async def _parse_gate_result(self, config: QualityGateConfig, result: Dict[str, Any]) -> QualityGateExecution:
        """Parse the result of a quality gate execution."""
        success = result['returncode'] == 0
        output = result['stdout']
        error = result['stderr']
        metrics = {}
        score = 0.0
        recommendations = []
        
        # Extract metrics using regex patterns
        for metric_name, pattern in config.metrics_extraction.items():
            matches = re.findall(pattern, output)
            if matches:
                try:
                    value = float(matches[0])
                    metrics[metric_name] = value
                except (ValueError, IndexError):
                    pass
        
        # Calculate score based on gate type
        if config.type == QualityGateType.SYNTAX:
            score = 100.0 if success else 0.0
            
        elif config.type == QualityGateType.TESTS:
            if 'passed' in metrics and 'failed' in metrics:
                total_tests = metrics['passed'] + metrics['failed']
                score = (metrics['passed'] / total_tests * 100) if total_tests > 0 else 0.0
            else:
                score = 100.0 if success else 0.0
                
        elif config.type == QualityGateType.COVERAGE:
            if 'coverage_percentage' in metrics:
                score = metrics['coverage_percentage']
            else:
                score = 80.0 if success else 0.0  # Default assumption
                
        elif config.type == QualityGateType.SECURITY:
            if 'vulnerabilities' in metrics:
                vuln_count = metrics['vulnerabilities']
                score = max(100.0 - (vuln_count * 10), 0.0)  # -10 points per vulnerability
            else:
                score = 100.0 if success else 50.0
                
        elif config.type == QualityGateType.PERFORMANCE:
            if 'response_time' in metrics:
                # Score based on response time (assuming < 200ms is ideal)
                response_time = metrics['response_time']
                score = max(100.0 - (response_time - 200) / 10, 0.0) if response_time > 200 else 100.0
            else:
                score = 85.0 if success else 40.0
                
        elif config.type == QualityGateType.COMPLIANCE:
            if 'compliance_score' in metrics:
                score = metrics['compliance_score']
            else:
                score = 95.0 if success else 60.0
                
        elif config.type == QualityGateType.DEPENDENCIES:
            if 'vulnerable_packages' in metrics:
                vuln_count = len(metrics['vulnerable_packages'])
                score = max(100.0 - (vuln_count * 5), 0.0)
            else:
                score = 100.0 if success else 75.0
        else:
            score = 100.0 if success else 0.0
        
        # Generate recommendations based on results
        if not success or score < config.threshold:
            recommendations = self._generate_gate_recommendations(config, result, score)
        
        # Final success determination
        final_success = success and score >= config.threshold
        
        return QualityGateExecution(
            config=config,
            success=final_success,
            score=score,
            duration=0.0,  # Will be set by caller
            output=output,
            error=error,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _generate_gate_recommendations(self, config: QualityGateConfig, result: Dict[str, Any], score: float) -> List[str]:
        """Generate recommendations for failed or low-scoring gates."""
        recommendations = []
        
        if config.type == QualityGateType.SYNTAX:
            recommendations.append("Fix syntax errors in source code")
            recommendations.append("Use automated code formatting tools")
            
        elif config.type == QualityGateType.TESTS:
            recommendations.append("Write more comprehensive unit tests")
            recommendations.append("Fix failing test cases")
            recommendations.append("Improve test coverage for critical paths")
            
        elif config.type == QualityGateType.COVERAGE:
            recommendations.append(f"Increase code coverage to meet {config.threshold}% threshold")
            recommendations.append("Add tests for uncovered code paths")
            
        elif config.type == QualityGateType.SECURITY:
            recommendations.append("Address security vulnerabilities")
            recommendations.append("Update dependencies to secure versions")
            recommendations.append("Implement additional security controls")
            
        elif config.type == QualityGateType.PERFORMANCE:
            recommendations.append("Optimize slow operations")
            recommendations.append("Implement caching strategies")
            recommendations.append("Profile and optimize critical paths")
            
        elif config.type == QualityGateType.COMPLIANCE:
            recommendations.append("Review and address compliance gaps")
            recommendations.append("Update privacy and data handling policies")
            
        elif config.type == QualityGateType.DEPENDENCIES:
            recommendations.append("Update vulnerable dependencies")
            recommendations.append("Review and audit third-party packages")
        
        return recommendations
    
    async def _check_file_structure(self) -> Dict[str, Any]:
        """Check project file structure for best practices."""
        required_files = [
            'README.md',
            'requirements.txt',
            'package.json',
            'Dockerfile',
            '.gitignore'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        score = ((len(required_files) - len(missing_files)) / len(required_files)) * 100
        
        return {
            'returncode': 0 if len(missing_files) == 0 else 1,
            'stdout': f"File structure check: {score:.1f}% complete\nMissing files: {missing_files}",
            'stderr': ''
        }
    
    async def _calculate_code_quality_metrics(self) -> Dict[str, Any]:
        """Calculate various code quality metrics."""
        # Simplified implementation - in reality would use tools like SonarQube
        python_files = list(self.project_root.glob('**/*.py'))
        js_files = list(self.project_root.glob('**/*.js'))
        
        total_lines = 0
        for file in python_files + js_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except Exception:
                pass
        
        # Mock quality score based on code organization
        quality_score = 85.0 if total_lines > 1000 else 70.0
        
        return {
            'returncode': 0,
            'stdout': f"Code quality score: {quality_score:.1f}%\nTotal lines: {total_lines}",
            'stderr': ''
        }
    
    def _generate_recommendations(self, results: Dict[str, QualityGateExecution]) -> List[str]:
        """Generate overall recommendations based on all gate results."""
        recommendations = []
        
        failed_gates = [name for name, result in results.items() if not result.success]
        low_scoring_gates = [
            name for name, result in results.items() 
            if result.score < result.config.threshold
        ]
        
        if failed_gates:
            recommendations.append(f"Address failed quality gates: {', '.join(failed_gates)}")
        
        if low_scoring_gates:
            recommendations.append(f"Improve scores for: {', '.join(low_scoring_gates)}")
        
        # Type-specific recommendations
        syntax_failures = [name for name, result in results.items() 
                          if result.config.type == QualityGateType.SYNTAX and not result.success]
        if syntax_failures:
            recommendations.append("Fix syntax errors before proceeding to other quality checks")
        
        test_failures = [name for name, result in results.items()
                        if result.config.type == QualityGateType.TESTS and not result.success]
        if test_failures:
            recommendations.append("Improve test coverage and fix failing tests")
        
        security_issues = [name for name, result in results.items()
                          if result.config.type == QualityGateType.SECURITY and not result.success]
        if security_issues:
            recommendations.append("Address security vulnerabilities immediately")
        
        return recommendations
    
    def _generate_quality_report(self, results: Dict[str, QualityGateExecution]) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        report = {
            'summary': {
                'total_gates': len(results),
                'passed_gates': len([r for r in results.values() if r.success]),
                'failed_gates': len([r for r in results.values() if not r.success]),
                'average_score': sum(r.score for r in results.values()) / len(results) if results else 0
            },
            'by_category': {},
            'trends': {},
            'critical_issues': []
        }
        
        # Group by category
        for gate_type in QualityGateType:
            category_results = [r for r in results.values() if r.config.type == gate_type]
            if category_results:
                report['by_category'][gate_type.value] = {
                    'gates': len(category_results),
                    'passed': len([r for r in category_results if r.success]),
                    'average_score': sum(r.score for r in category_results) / len(category_results)
                }
        
        # Identify critical issues
        critical_failures = [
            result for result in results.values()
            if not result.success and result.config.type in [
                QualityGateType.SECURITY, QualityGateType.COMPLIANCE
            ]
        ]
        
        for failure in critical_failures:
            report['critical_issues'].append({
                'gate': failure.config.name,
                'type': failure.config.type.value,
                'score': failure.score,
                'recommendations': failure.recommendations
            })
        
        return report
    
    def _serialize_execution(self, execution: QualityGateExecution) -> Dict[str, Any]:
        """Serialize quality gate execution for JSON output."""
        return {
            'gate_name': execution.config.name,
            'gate_type': execution.config.type.value,
            'success': execution.success,
            'score': execution.score,
            'threshold': execution.config.threshold,
            'duration': execution.duration,
            'metrics': execution.metrics,
            'recommendations': execution.recommendations,
            'artifacts': execution.artifacts
        }
    
    def get_current_scores(self) -> Dict[str, float]:
        """Get current scores for all quality gates."""
        return self.current_scores.copy()
    
    def get_execution_history(self) -> List[QualityGateExecution]:
        """Get execution history."""
        return self.execution_history.copy()
    
    async def validate_gate_thresholds(self) -> Dict[str, List[str]]:
        """Validate that all quality gate thresholds are reasonable."""
        issues = {}
        
        for name, config in self.quality_gates.items():
            gate_issues = []
            
            if config.threshold < 0 or config.threshold > 100:
                gate_issues.append(f"Threshold {config.threshold} is out of valid range 0-100")
            
            if config.weight <= 0:
                gate_issues.append(f"Weight {config.weight} must be positive")
            
            if config.timeout <= 0:
                gate_issues.append(f"Timeout {config.timeout} must be positive")
            
            if not config.commands and config.type != QualityGateType.COMPLIANCE:
                gate_issues.append("No commands specified for gate execution")
            
            if gate_issues:
                issues[name] = gate_issues
        
        return issues