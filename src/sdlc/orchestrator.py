"""
SDLC Orchestrator - Main coordination engine for autonomous software development lifecycle
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .autonomous_executor import AutonomousSDLCExecutor, SDLCConfig, SDLCPhase
from .progressive_enhancer import ProgressiveEnhancer, GenerationPhase
from .quality_gates_engine import QualityGatesEngine, QualityGateType
from .security_engine import SecurityEngine
from .monitoring_engine import MonitoringEngine
from .performance_engine import PerformanceEngine
from .deployment_engine import DeploymentEngine, DeploymentEnvironment

from ..utils.logger import get_logger


class OrchestrationMode(Enum):
    """Orchestration execution modes."""
    AUTONOMOUS = "autonomous"  # Full autonomous execution
    SUPERVISED = "supervised"  # Human oversight at key points
    MANUAL = "manual"  # Step-by-step manual execution


@dataclass
class OrchestrationConfig:
    """Configuration for SDLC orchestration."""
    project_root: Path
    mode: OrchestrationMode = OrchestrationMode.AUTONOMOUS
    enable_progressive_enhancement: bool = True
    enable_quality_gates: bool = True
    enable_security_validation: bool = True
    enable_performance_optimization: bool = True
    enable_monitoring: bool = True
    enable_deployment: bool = True
    target_environment: DeploymentEnvironment = DeploymentEnvironment.STAGING
    quality_threshold: float = 85.0
    security_threshold: float = 90.0
    performance_threshold: float = 80.0


@dataclass
class OrchestrationResult:
    """Result of orchestrated SDLC execution."""
    success: bool
    total_duration: float
    phases_completed: List[str]
    overall_score: float
    generation_results: Dict[str, Any] = field(default_factory=dict)
    quality_gate_results: Dict[str, Any] = field(default_factory=dict)
    security_results: Dict[str, Any] = field(default_factory=dict)
    performance_results: Dict[str, Any] = field(default_factory=dict)
    deployment_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)


class SDLCOrchestrator:
    """
    SDLC Orchestrator - Main coordination engine for autonomous software development.
    
    This orchestrator coordinates all aspects of the autonomous SDLC including:
    - Progressive enhancement execution (Gen 1 → Gen 2 → Gen 3)
    - Comprehensive quality gate validation
    - Security validation and enforcement
    - Performance optimization and monitoring
    - Automated deployment and infrastructure management
    - Real-time monitoring and alerting
    """
    
    def __init__(self, config: OrchestrationConfig, logger=None):
        """Initialize SDLC orchestrator."""
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Initialize all engines
        self.sdlc_executor = AutonomousSDLCExecutor(
            SDLCConfig(
                project_root=config.project_root,
                enable_progressive_enhancement=config.enable_progressive_enhancement,
                enable_autonomous_execution=config.mode == OrchestrationMode.AUTONOMOUS
            ),
            logger=self.logger
        )
        
        self.progressive_enhancer = ProgressiveEnhancer(
            config.project_root,
            logger=self.logger
        ) if config.enable_progressive_enhancement else None
        
        self.quality_gates_engine = QualityGatesEngine(
            config.project_root,
            logger=self.logger
        ) if config.enable_quality_gates else None
        
        self.security_engine = SecurityEngine(
            logger=self.logger
        ) if config.enable_security_validation else None
        
        self.monitoring_engine = MonitoringEngine(
            logger=self.logger
        ) if config.enable_monitoring else None
        
        self.performance_engine = PerformanceEngine(
            logger=self.logger
        ) if config.enable_performance_optimization else None
        
        self.deployment_engine = DeploymentEngine(
            logger=self.logger
        ) if config.enable_deployment else None
        
        # Orchestration state
        self.execution_history: List[OrchestrationResult] = []
        self.current_execution: Optional[OrchestrationResult] = None
        
        self.logger.info(f"SDLC Orchestrator initialized for {config.project_root}")
    
    async def execute_full_sdlc(self) -> OrchestrationResult:
        """Execute the complete autonomous SDLC."""
        self.logger.info("🚀 Starting Full Autonomous SDLC Execution")
        start_time = time.time()
        
        result = OrchestrationResult(
            success=False,
            total_duration=0.0,
            phases_completed=[],
            overall_score=0.0
        )
        
        self.current_execution = result
        
        try:
            # Phase 1: Progressive Enhancement
            if self.progressive_enhancer and self.config.enable_progressive_enhancement:
                self.logger.info("📈 Executing Progressive Enhancement")
                enhancement_result = await self.progressive_enhancer.execute_all_generations()
                result.generation_results = enhancement_result
                result.phases_completed.append("progressive_enhancement")
                
                if enhancement_result['status'] != 'completed':
                    self.logger.warning("Progressive enhancement did not complete fully")
            
            # Phase 2: Quality Gates Validation
            if self.quality_gates_engine and self.config.enable_quality_gates:
                self.logger.info("🔍 Executing Quality Gates")
                quality_result = await self.quality_gates_engine.execute_all_gates()
                result.quality_gate_results = quality_result
                result.phases_completed.append("quality_gates")
                
                if quality_result['overall_score'] < self.config.quality_threshold:
                    self.logger.warning(f"Quality score {quality_result['overall_score']:.1f}% below threshold {self.config.quality_threshold}%")
            
            # Phase 3: Security Validation
            if self.security_engine and self.config.enable_security_validation:
                self.logger.info("🛡️ Executing Security Validation")
                security_result = await self._execute_security_validation()
                result.security_results = security_result
                result.phases_completed.append("security_validation")
                
                if security_result['security_score'] < self.config.security_threshold:
                    self.logger.warning(f"Security score {security_result['security_score']:.1f}% below threshold {self.config.security_threshold}%")
            
            # Phase 4: Performance Optimization
            if self.performance_engine and self.config.enable_performance_optimization:
                self.logger.info("⚡ Executing Performance Optimization")
                performance_result = await self._execute_performance_optimization()
                result.performance_results = performance_result
                result.phases_completed.append("performance_optimization")
                
                if performance_result['performance_score'] < self.config.performance_threshold:
                    self.logger.warning(f"Performance score {performance_result['performance_score']:.1f}% below threshold {self.config.performance_threshold}%")
            
            # Phase 5: Monitoring Setup
            if self.monitoring_engine and self.config.enable_monitoring:
                self.logger.info("📊 Setting up Monitoring")
                await self.monitoring_engine.start_monitoring()
                monitoring_result = await self._setup_monitoring()
                result.phases_completed.append("monitoring_setup")
            
            # Phase 6: Deployment Preparation
            if self.deployment_engine and self.config.enable_deployment:
                self.logger.info("🚀 Preparing Deployment")
                deployment_result = await self._prepare_deployment()
                result.deployment_results = deployment_result
                result.phases_completed.append("deployment_preparation")
            
            # Calculate overall results
            result.total_duration = time.time() - start_time
            result.overall_score = self._calculate_overall_score(result)
            result.success = self._determine_overall_success(result)
            result.recommendations = self._generate_recommendations(result)
            result.artifacts = self._collect_artifacts(result)
            
            # Store execution result
            self.execution_history.append(result)
            
            status = "✅ SUCCESS" if result.success else "⚠️ PARTIAL SUCCESS"
            self.logger.info(f"{status} Full SDLC execution completed in {result.total_duration:.2f}s")
            self.logger.info(f"📊 Overall Score: {result.overall_score:.1f}%")
            
            return result
            
        except Exception as e:
            result.total_duration = time.time() - start_time
            result.success = False
            result.overall_score = 0.0
            self.logger.error(f"❌ SDLC execution failed: {str(e)}")
            
            self.execution_history.append(result)
            return result
        
        finally:
            self.current_execution = None
    
    async def _execute_security_validation(self) -> Dict[str, Any]:
        """Execute comprehensive security validation."""
        # Generate security report
        security_report = await self.security_engine.generate_security_report()
        
        # Test various security scenarios
        test_requests = [
            {
                'source_ip': '127.0.0.1',
                'path': '/api/generate',
                'payload': {'schema': {'name': 'string', 'age': 'integer'}},
                'headers': {'authorization': 'Bearer valid_token'}
            },
            {
                'source_ip': '192.168.1.100',
                'path': '/api/validate',
                'payload': {'data': 'SELECT * FROM users; DROP TABLE users;'},  # SQL injection attempt
                'headers': {}
            },
            {
                'source_ip': '10.0.0.1',
                'path': '/health',
                'payload': {},
                'headers': {}
            }
        ]
        
        validation_results = []
        for request in test_requests:
            validation_result = await self.security_engine.validate_request(request)
            validation_results.append({
                'request': request,
                'result': validation_result
            })
        
        # Calculate security score
        blocked_malicious = len([r for r in validation_results if not r['result']['allowed'] and 'injection' in str(r['request']['payload'])])
        allowed_legitimate = len([r for r in validation_results if r['result']['allowed'] and 'health' in r['request']['path']])
        
        security_score = security_report['metrics']['security_score']
        
        return {
            'security_score': security_score,
            'security_report': security_report,
            'validation_tests': validation_results,
            'blocked_malicious_requests': blocked_malicious,
            'allowed_legitimate_requests': allowed_legitimate
        }
    
    async def _execute_performance_optimization(self) -> Dict[str, Any]:
        """Execute performance optimization and benchmarking."""
        # Generate performance report
        performance_report = await self.performance_engine.generate_performance_report()
        
        # Run optimization tests
        async def sample_operation():
            await asyncio.sleep(0.1)  # Simulate work
            return "operation_result"
        
        # Benchmark different optimization strategies
        benchmark_results = {}
        
        from .performance_engine import OptimizationStrategy
        for strategy in OptimizationStrategy:
            benchmark_result = await self.performance_engine.benchmark_operation(
                sample_operation,
                iterations=5
            )
            benchmark_results[strategy.value] = benchmark_result
        
        # Test caching performance
        cache_tests = []
        for i in range(100):
            key = f"test_key_{i}"
            value = f"test_value_{i}"
            self.performance_engine.cache_result(key, value)
        
        # Test cache hit rate
        cache_hits = 0
        for i in range(50):  # Test first 50 keys
            key = f"test_key_{i}"
            cached_value = self.performance_engine.get_cached_result(key)
            if cached_value is not None:
                cache_hits += 1
        
        cache_hit_rate = cache_hits / 50
        
        # Detect bottlenecks
        bottlenecks = self.performance_engine.detect_bottlenecks()
        
        return {
            'performance_score': performance_report['performance_score'],
            'performance_report': performance_report,
            'benchmark_results': benchmark_results,
            'cache_hit_rate': cache_hit_rate,
            'bottlenecks': bottlenecks,
            'optimization_opportunities': performance_report['optimization_opportunities']
        }
    
    async def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup comprehensive monitoring."""
        # Record some sample metrics
        await self.monitoring_engine.record_counter('sdlc_executions_total', 1)
        await self.monitoring_engine.record_gauge('sdlc_health_score', 95.0)
        
        # Get health status
        health_status = self.monitoring_engine.get_health_status()
        
        # Get metrics summary
        metrics_summary = self.monitoring_engine.get_metrics_summary()
        
        # Generate monitoring report
        monitoring_report = await self.monitoring_engine.generate_monitoring_report()
        
        return {
            'health_status': health_status,
            'metrics_summary': metrics_summary,
            'monitoring_report': monitoring_report
        }
    
    async def _prepare_deployment(self) -> Dict[str, Any]:
        """Prepare deployment configuration and infrastructure."""
        from .deployment_engine import DeploymentConfig, DeploymentStrategy, InfrastructureProvider
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            name="synthetic-data-guardian",
            environment=self.config.target_environment,
            strategy=DeploymentStrategy.ROLLING,
            provider=InfrastructureProvider.KUBERNETES,
            image_tag="latest",
            replicas=3 if self.config.target_environment == DeploymentEnvironment.PRODUCTION else 2,
            environment_vars={
                'NODE_ENV': self.config.target_environment.value,
                'LOG_LEVEL': 'INFO',
                'ENABLE_METRICS': 'true'
            },
            health_checks={
                'liveness': {'path': '/health', 'port': 8080},
                'readiness': {'path': '/ready', 'port': 8080}
            }
        )
        
        # Generate infrastructure code
        infra_result = await self.deployment_engine.generate_infrastructure_code(
            deployment_config,
            self.config.project_root / 'infrastructure'
        )
        
        # Generate deployment report
        deployment_report = await self.deployment_engine.generate_deployment_report()
        
        return {
            'deployment_config': deployment_config.__dict__,
            'infrastructure_generation': infra_result,
            'deployment_report': deployment_report,
            'ready_for_deployment': infra_result['success']
        }
    
    def _calculate_overall_score(self, result: OrchestrationResult) -> float:
        """Calculate overall SDLC execution score."""
        scores = []
        weights = []
        
        # Progressive enhancement score
        if result.generation_results:
            overall_metrics = result.generation_results.get('overall_metrics', {})
            enhancement_score = overall_metrics.get('progressive_enhancement_score', 0)
            scores.append(enhancement_score)
            weights.append(0.25)
        
        # Quality gates score
        if result.quality_gate_results:
            quality_score = result.quality_gate_results.get('overall_score', 0)
            scores.append(quality_score)
            weights.append(0.30)
        
        # Security score
        if result.security_results:
            security_score = result.security_results.get('security_score', 0)
            scores.append(security_score)
            weights.append(0.20)
        
        # Performance score
        if result.performance_results:
            performance_score = result.performance_results.get('performance_score', 0)
            scores.append(performance_score)
            weights.append(0.15)
        
        # Deployment readiness score
        if result.deployment_results:
            deployment_score = 100.0 if result.deployment_results.get('ready_for_deployment') else 70.0
            scores.append(deployment_score)
            weights.append(0.10)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            return weighted_sum / total_weight
        
        return 0.0
    
    def _determine_overall_success(self, result: OrchestrationResult) -> bool:
        """Determine if overall SDLC execution was successful."""
        # Check minimum phase completion
        required_phases = ["progressive_enhancement", "quality_gates"]
        missing_phases = [phase for phase in required_phases if phase not in result.phases_completed]
        
        if missing_phases:
            return False
        
        # Check minimum score threshold
        if result.overall_score < 70.0:
            return False
        
        # Check specific thresholds
        if result.quality_gate_results:
            quality_score = result.quality_gate_results.get('overall_score', 0)
            if quality_score < self.config.quality_threshold:
                return False
        
        if result.security_results:
            security_score = result.security_results.get('security_score', 0)
            if security_score < self.config.security_threshold:
                return False
        
        return True
    
    def _generate_recommendations(self, result: OrchestrationResult) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []
        
        # Check overall score
        if result.overall_score < 85.0:
            recommendations.append(f"Overall SDLC score is {result.overall_score:.1f}% - aim for 85%+")
        
        # Progressive enhancement recommendations
        if result.generation_results:
            gen_recommendations = result.generation_results.get('recommendations', [])
            recommendations.extend(gen_recommendations)
        
        # Quality gates recommendations
        if result.quality_gate_results:
            quality_recommendations = result.quality_gate_results.get('recommendations', [])
            recommendations.extend(quality_recommendations)
        
        # Security recommendations
        if result.security_results and 'security_report' in result.security_results:
            security_recommendations = result.security_results['security_report'].get('recommendations', [])
            recommendations.extend(security_recommendations)
        
        # Performance recommendations
        if result.performance_results and 'performance_report' in result.performance_results:
            perf_recommendations = result.performance_results['performance_report'].get('recommendations', [])
            recommendations.extend(perf_recommendations)
        
        # Deployment recommendations
        if result.deployment_results and 'deployment_report' in result.deployment_results:
            deploy_recommendations = result.deployment_results['deployment_report'].get('recommendations', [])
            recommendations.extend(deploy_recommendations)
        
        # Add general recommendations
        if not result.success:
            recommendations.insert(0, "Address critical issues before proceeding to production")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _collect_artifacts(self, result: OrchestrationResult) -> List[str]:
        """Collect all artifacts generated during SDLC execution."""
        artifacts = []
        
        # Progressive enhancement artifacts
        if result.generation_results and 'generations_executed' in result.generation_results:
            generations = result.generation_results['generation_results']
            for gen_name, gen_result in generations.items():
                if 'tasks_completed' in gen_result:
                    for task in gen_result['tasks_completed']:
                        artifacts.extend(task.get('artifacts_created', []))
        
        # Quality gates artifacts
        artifacts.extend([
            'quality_gate_report.json',
            'test_results.xml',
            'coverage_report.html'
        ])
        
        # Security artifacts
        artifacts.extend([
            'security_report.json',
            'vulnerability_scan.json'
        ])
        
        # Performance artifacts
        artifacts.extend([
            'performance_report.json',
            'benchmark_results.json'
        ])
        
        # Deployment artifacts
        if result.deployment_results and 'infrastructure_generation' in result.deployment_results:
            infra_files = result.deployment_results['infrastructure_generation'].get('files_created', [])
            artifacts.extend(infra_files)
        
        return artifacts
    
    async def execute_single_phase(self, phase: str) -> Dict[str, Any]:
        """Execute a single SDLC phase."""
        self.logger.info(f"Executing single phase: {phase}")
        
        if phase == "progressive_enhancement" and self.progressive_enhancer:
            return await self.progressive_enhancer.execute_all_generations()
        elif phase == "quality_gates" and self.quality_gates_engine:
            return await self.quality_gates_engine.execute_all_gates()
        elif phase == "security_validation" and self.security_engine:
            return await self._execute_security_validation()
        elif phase == "performance_optimization" and self.performance_engine:
            return await self._execute_performance_optimization()
        elif phase == "monitoring_setup" and self.monitoring_engine:
            await self.monitoring_engine.start_monitoring()
            return await self._setup_monitoring()
        elif phase == "deployment_preparation" and self.deployment_engine:
            return await self._prepare_deployment()
        else:
            raise ValueError(f"Unknown or disabled phase: {phase}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        if self.current_execution is None:
            return {'status': 'idle', 'message': 'No active execution'}
        
        return {
            'status': 'running',
            'phases_completed': self.current_execution.phases_completed,
            'current_score': self.current_execution.overall_score,
            'elapsed_time': time.time() - (time.time() - self.current_execution.total_duration)
        }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history summary."""
        return [
            {
                'timestamp': time.time(),  # Simplified - would store actual timestamp
                'success': result.success,
                'overall_score': result.overall_score,
                'duration': result.total_duration,
                'phases_completed': result.phases_completed
            }
            for result in self.execution_history
        ]
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive SDLC report."""
        if not self.execution_history:
            return {'error': 'No execution history available'}
        
        latest_execution = self.execution_history[-1]
        
        # Executive summary
        executive_summary = {
            'overall_success': latest_execution.success,
            'overall_score': latest_execution.overall_score,
            'execution_duration': latest_execution.total_duration,
            'phases_completed': len(latest_execution.phases_completed),
            'artifacts_generated': len(latest_execution.artifacts),
            'recommendations_count': len(latest_execution.recommendations)
        }
        
        # Detailed analysis
        detailed_analysis = {
            'progressive_enhancement': latest_execution.generation_results,
            'quality_gates': latest_execution.quality_gate_results,
            'security_validation': latest_execution.security_results,
            'performance_optimization': latest_execution.performance_results,
            'deployment_preparation': latest_execution.deployment_results
        }
        
        # Trend analysis
        if len(self.execution_history) > 1:
            trend_analysis = self._analyze_trends()
        else:
            trend_analysis = {'message': 'Insufficient data for trend analysis'}
        
        return {
            'report_timestamp': time.time(),
            'executive_summary': executive_summary,
            'detailed_analysis': detailed_analysis,
            'trend_analysis': trend_analysis,
            'recommendations': latest_execution.recommendations,
            'artifacts': latest_execution.artifacts,
            'next_steps': self._generate_next_steps(latest_execution)
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends across multiple executions."""
        if len(self.execution_history) < 2:
            return {}
        
        scores = [result.overall_score for result in self.execution_history]
        durations = [result.total_duration for result in self.execution_history]
        
        # Calculate trends
        score_trend = "improving" if scores[-1] > scores[0] else "declining"
        duration_trend = "faster" if durations[-1] < durations[0] else "slower"
        
        return {
            'score_trend': score_trend,
            'duration_trend': duration_trend,
            'average_score': sum(scores) / len(scores),
            'average_duration': sum(durations) / len(durations),
            'total_executions': len(self.execution_history)
        }
    
    def _generate_next_steps(self, result: OrchestrationResult) -> List[str]:
        """Generate next steps based on execution results."""
        next_steps = []
        
        if result.success:
            next_steps.extend([
                "Proceed with deployment to target environment",
                "Set up continuous monitoring and alerting",
                "Schedule regular SDLC health checks",
                "Plan next iteration improvements"
            ])
        else:
            next_steps.extend([
                "Address critical issues identified in recommendations",
                "Re-run failed phases after fixes",
                "Review and adjust quality thresholds if needed",
                "Consider additional testing before deployment"
            ])
        
        # Add specific next steps based on missing phases
        all_phases = [
            "progressive_enhancement", "quality_gates", "security_validation",
            "performance_optimization", "monitoring_setup", "deployment_preparation"
        ]
        
        missing_phases = [phase for phase in all_phases if phase not in result.phases_completed]
        if missing_phases:
            next_steps.append(f"Complete missing phases: {', '.join(missing_phases)}")
        
        return next_steps
    
    async def cleanup(self) -> None:
        """Clean up orchestrator resources."""
        self.logger.info("Cleaning up SDLC Orchestrator")
        
        # Cleanup engines
        if self.monitoring_engine:
            await self.monitoring_engine.stop_monitoring()
        
        if self.performance_engine:
            await self.performance_engine.cleanup()
        
        if self.sdlc_executor:
            await self.sdlc_executor.cleanup()
        
        self.logger.info("SDLC Orchestrator cleanup completed")