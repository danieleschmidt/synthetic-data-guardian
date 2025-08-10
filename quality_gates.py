#!/usr/bin/env python3
"""
Quality Gates for Synthetic Data Guardian

Automated quality checks that must pass before deployment.
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class QualityGate:
    """Individual quality gate check."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = 0
        self.end_time = 0
        self.passed = False
        self.score = 0.0
        self.details = {}
        self.errors = []
    
    def start(self):
        """Start timing the quality gate."""
        self.start_time = time.time()
        print(f"🔍 {self.name}: {self.description}")
    
    def finish(self, passed: bool, score: float = 0.0, details: Dict = None, errors: List = None):
        """Finish the quality gate with results."""
        self.end_time = time.time()
        self.passed = passed
        self.score = score
        self.details = details or {}
        self.errors = errors or []
        
        duration = self.end_time - self.start_time
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} ({duration:.2f}s) - Score: {score:.2f}")
        
        if not passed and errors:
            for error in errors:
                print(f"    ❌ {error}")


class QualityGateRunner:
    """Runs all quality gates and reports results."""
    
    def __init__(self):
        self.gates: List[QualityGate] = []
        self.overall_start_time = 0
        self.overall_end_time = 0
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate to the runner."""
        self.gates.append(gate)
    
    async def run_all_gates(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all quality gates and return overall pass/fail status."""
        print("🚀 Running Quality Gates for Synthetic Data Guardian")
        print("=" * 70)
        
        self.overall_start_time = time.time()
        
        # Run each gate
        for gate in self.gates:
            gate.start()
            
            try:
                if gate.name == "Unit Tests":
                    await self._run_unit_tests(gate)
                elif gate.name == "Integration Tests":
                    await self._run_integration_tests(gate)
                elif gate.name == "Performance Tests":
                    await self._run_performance_tests(gate)
                elif gate.name == "Security Scan":
                    await self._run_security_scan(gate)
                elif gate.name == "Code Quality":
                    await self._run_code_quality(gate)
                elif gate.name == "Memory Leak Detection":
                    await self._run_memory_leak_detection(gate)
                elif gate.name == "Dependency Check":
                    await self._run_dependency_check(gate)
                elif gate.name == "Documentation Check":
                    await self._run_documentation_check(gate)
                else:
                    gate.finish(False, 0.0, errors=[f"Unknown gate: {gate.name}"])
                    
            except Exception as e:
                gate.finish(False, 0.0, errors=[f"Gate execution failed: {str(e)}"])
        
        self.overall_end_time = time.time()
        
        # Generate report
        return self._generate_report()
    
    async def _run_unit_tests(self, gate: QualityGate):
        """Run unit tests."""
        try:
            # Run our comprehensive test suite
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/test_suite.py', 
                '-v', '--tb=short', '--maxfail=10'
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            passed = result.returncode == 0
            
            # Parse test results
            output_lines = result.stdout.split('\n')
            test_summary = [line for line in output_lines if 'passed' in line or 'failed' in line]
            
            details = {
                'exit_code': result.returncode,
                'test_summary': test_summary,
                'stdout': result.stdout[:1000],  # Truncate for readability
                'stderr': result.stderr[:1000]
            }
            
            errors = []
            if not passed:
                errors.append("Unit tests failed")
                errors.extend(result.stderr.split('\n')[:5])  # First 5 error lines
            
            # Calculate score based on test results
            score = 100.0 if passed else 0.0
            
            gate.finish(passed, score, details, errors)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Failed to run unit tests: {str(e)}"])
    
    async def _run_integration_tests(self, gate: QualityGate):
        """Run integration tests."""
        try:
            # Run our basic functionality tests
            test_files = [
                'test_minimal.py',
                'test_basic_guardian.py',
                'test_robustness.py',
                'test_optimization.py'
            ]
            
            passed_tests = 0
            total_tests = len(test_files)
            errors = []
            
            for test_file in test_files:
                test_path = Path(__file__).parent / test_file
                if test_path.exists():
                    result = subprocess.run([
                        sys.executable, str(test_path)
                    ], capture_output=True, text=True, cwd=Path(__file__).parent)
                    
                    if result.returncode == 0:
                        passed_tests += 1
                    else:
                        errors.append(f"{test_file}: {result.stderr.split(chr(10))[0] if result.stderr else 'Failed'}")
                else:
                    errors.append(f"{test_file}: File not found")
            
            passed = passed_tests == total_tests
            score = (passed_tests / total_tests) * 100.0
            
            details = {
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'pass_rate': score
            }
            
            gate.finish(passed, score, details, errors)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Failed to run integration tests: {str(e)}"])
    
    async def _run_performance_tests(self, gate: QualityGate):
        """Run performance tests."""
        try:
            # Test basic generation performance
            from synthetic_guardian.core.guardian import Guardian
            
            start_time = time.time()
            
            async with Guardian() as guardian:
                pipeline_config = {
                    "name": "performance_test",
                    "generator_type": "tabular",
                    "data_type": "tabular",
                    "schema": {
                        "id": "integer",
                        "value": "float"
                    }
                }
                
                # Generate data and measure performance
                generation_times = []
                for i in range(3):
                    gen_start = time.time()
                    result = await guardian.generate(
                        pipeline_config=pipeline_config,
                        num_records=100,
                        seed=i,
                        validate=False,
                        watermark=False
                    )
                    gen_time = time.time() - gen_start
                    generation_times.append(gen_time)
                    
                    if len(result.data) != 100:
                        raise Exception(f"Expected 100 records, got {len(result.data)}")
            
            total_time = time.time() - start_time
            avg_generation_time = sum(generation_times) / len(generation_times)
            
            # Performance thresholds
            max_total_time = 30.0  # 30 seconds max
            max_avg_generation = 5.0  # 5 seconds per generation max
            
            passed = total_time < max_total_time and avg_generation_time < max_avg_generation
            score = min(100.0, (max_total_time - total_time) / max_total_time * 100.0)
            
            details = {
                'total_time': total_time,
                'avg_generation_time': avg_generation_time,
                'generation_times': generation_times,
                'max_total_time': max_total_time,
                'max_avg_generation': max_avg_generation
            }
            
            errors = []
            if not passed:
                if total_time >= max_total_time:
                    errors.append(f"Total time {total_time:.2f}s exceeds {max_total_time}s")
                if avg_generation_time >= max_avg_generation:
                    errors.append(f"Avg generation time {avg_generation_time:.2f}s exceeds {max_avg_generation}s")
            
            gate.finish(passed, score, details, errors)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Performance test failed: {str(e)}"])
    
    async def _run_security_scan(self, gate: QualityGate):
        """Run security scans."""
        try:
            # Basic security checks
            security_issues = []
            
            # Check for common security anti-patterns in code
            src_path = Path(__file__).parent / 'src'
            if src_path.exists():
                dangerous_patterns = [
                    ('eval(', 'Use of eval() function'),
                    ('exec(', 'Use of exec() function'),
                    ('subprocess.call', 'Direct subprocess call'),
                    ('os.system', 'Use of os.system()'),
                    ('input(', 'Use of input() without validation'),
                ]
                
                for py_file in src_path.rglob('*.py'):
                    try:
                        content = py_file.read_text()
                        for pattern, description in dangerous_patterns:
                            if pattern in content:
                                security_issues.append(f"{py_file.name}: {description}")
                    except Exception:
                        continue
            
            # Test input validation
            from synthetic_guardian.middleware.input_validator import InputValidator
            validator = InputValidator()
            
            # Test XSS prevention
            try:
                validator.validate_input("<script>alert('xss')</script>", ['safe_string'])
                security_issues.append("XSS prevention not working")
            except:
                pass  # Good, should raise exception
            
            # Test SQL injection prevention
            try:
                validator.validate_input("'; DROP TABLE users; --", ['safe_string'])
                security_issues.append("SQL injection prevention not working")
            except:
                pass  # Good, should raise exception
            
            passed = len(security_issues) == 0
            score = max(0.0, 100.0 - len(security_issues) * 10.0)
            
            details = {
                'issues_found': len(security_issues),
                'security_score': score
            }
            
            gate.finish(passed, score, details, security_issues)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Security scan failed: {str(e)}"])
    
    async def _run_code_quality(self, gate: QualityGate):
        """Run code quality checks."""
        try:
            src_path = Path(__file__).parent / 'src'
            quality_score = 100.0
            issues = []
            
            # Basic code quality metrics
            total_lines = 0
            total_files = 0
            
            if src_path.exists():
                for py_file in src_path.rglob('*.py'):
                    try:
                        content = py_file.read_text()
                        lines = content.split('\n')
                        total_lines += len(lines)
                        total_files += 1
                        
                        # Check for very long functions (basic heuristic)
                        in_function = False
                        function_lines = 0
                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith('def ') or stripped.startswith('async def '):
                                in_function = True
                                function_lines = 0
                            elif in_function:
                                function_lines += 1
                                if function_lines > 100:
                                    issues.append(f"{py_file.name}: Very long function detected")
                                    quality_score -= 5
                                    in_function = False
                                elif stripped == '' or not stripped:
                                    continue
                                elif not stripped.startswith(' ') and not stripped.startswith('\t'):
                                    in_function = False
                    except Exception:
                        continue
            
            # Check for basic documentation
            readme_exists = (Path(__file__).parent / 'README.md').exists()
            if not readme_exists:
                issues.append("README.md not found")
                quality_score -= 10
            
            passed = quality_score >= 70.0  # Minimum 70% quality score
            
            details = {
                'total_files': total_files,
                'total_lines': total_lines,
                'quality_score': quality_score,
                'readme_exists': readme_exists
            }
            
            gate.finish(passed, quality_score, details, issues)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Code quality check failed: {str(e)}"])
    
    async def _run_memory_leak_detection(self, gate: QualityGate):
        """Run memory leak detection."""
        try:
            import psutil
            import gc
            
            # Monitor memory usage during Guardian operations
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run multiple Guardian operations
            from synthetic_guardian.core.guardian import Guardian
            
            for i in range(5):
                async with Guardian() as guardian:
                    pipeline_config = {
                        "name": f"memory_test_{i}",
                        "generator_type": "tabular",
                        "data_type": "tabular",
                        "schema": {"id": "integer", "value": "float"}
                    }
                    
                    await guardian.generate(
                        pipeline_config=pipeline_config,
                        num_records=50,
                        validate=False,
                        watermark=False
                    )
                
                # Force garbage collection
                gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Allow some memory increase, but not too much
            max_increase = 100  # MB
            passed = memory_increase < max_increase
            
            score = max(0.0, 100.0 - (memory_increase / max_increase) * 100.0) if memory_increase > 0 else 100.0
            
            details = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'max_allowed_increase_mb': max_increase
            }
            
            errors = []
            if not passed:
                errors.append(f"Memory increased by {memory_increase:.1f}MB (max: {max_increase}MB)")
            
            gate.finish(passed, score, details, errors)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Memory leak detection failed: {str(e)}"])
    
    async def _run_dependency_check(self, gate: QualityGate):
        """Check dependencies and imports."""
        try:
            issues = []
            
            # Test critical imports
            critical_imports = [
                'synthetic_guardian.core.guardian',
                'synthetic_guardian.generators.tabular',
                'synthetic_guardian.middleware.error_handler',
                'synthetic_guardian.middleware.input_validator',
                'synthetic_guardian.monitoring.health_monitor',
                'synthetic_guardian.optimization.performance_optimizer',
                'synthetic_guardian.optimization.caching'
            ]
            
            failed_imports = []
            for module in critical_imports:
                try:
                    __import__(module)
                except ImportError as e:
                    failed_imports.append(f"{module}: {str(e)}")
            
            # Check if required system packages are available
            system_packages = ['numpy', 'pandas', 'psutil']
            missing_packages = []
            
            for package in system_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            issues.extend(failed_imports)
            issues.extend([f"Missing system package: {pkg}" for pkg in missing_packages])
            
            passed = len(issues) == 0
            score = max(0.0, 100.0 - len(issues) * 20.0)
            
            details = {
                'critical_imports_tested': len(critical_imports),
                'failed_imports': len(failed_imports),
                'missing_packages': len(missing_packages)
            }
            
            gate.finish(passed, score, details, issues)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Dependency check failed: {str(e)}"])
    
    async def _run_documentation_check(self, gate: QualityGate):
        """Check documentation completeness."""
        try:
            issues = []
            score = 100.0
            
            # Check for key documentation files
            doc_files = [
                ('README.md', 'Main documentation'),
                ('CLAUDE.md', 'Claude Code configuration'),
            ]
            
            missing_docs = []
            for filename, description in doc_files:
                doc_path = Path(__file__).parent / filename
                if not doc_path.exists():
                    missing_docs.append(f"{filename}: {description}")
                    score -= 25
            
            # Check for basic docstrings in core modules
            src_path = Path(__file__).parent / 'src'
            if src_path.exists():
                core_modules = [
                    'synthetic_guardian/core/guardian.py',
                    'synthetic_guardian/generators/base.py',
                    'synthetic_guardian/generators/tabular.py'
                ]
                
                undocumented_modules = []
                for module_path in core_modules:
                    full_path = src_path / module_path
                    if full_path.exists():
                        try:
                            content = full_path.read_text()
                            # Basic check for module docstring
                            if not ('"""' in content[:500] or "'''" in content[:500]):
                                undocumented_modules.append(module_path)
                                score -= 10
                        except Exception:
                            continue
                
                issues.extend(undocumented_modules)
            
            issues.extend(missing_docs)
            passed = score >= 50.0  # Minimum 50% documentation score
            
            details = {
                'documentation_score': score,
                'missing_docs': len(missing_docs),
                'undocumented_modules': len(undocumented_modules) if 'undocumented_modules' in locals() else 0
            }
            
            gate.finish(passed, score, details, issues)
            
        except Exception as e:
            gate.finish(False, 0.0, errors=[f"Documentation check failed: {str(e)}"])
    
    def _generate_report(self) -> Tuple[bool, Dict[str, Any]]:
        """Generate comprehensive quality gate report."""
        total_duration = self.overall_end_time - self.overall_start_time
        
        passed_gates = [gate for gate in self.gates if gate.passed]
        failed_gates = [gate for gate in self.gates if not gate.passed]
        
        overall_passed = len(failed_gates) == 0
        overall_score = sum(gate.score for gate in self.gates) / len(self.gates) if self.gates else 0.0
        
        print("=" * 70)
        print(f"🎯 Quality Gate Results ({total_duration:.2f}s)")
        print(f"Overall Score: {overall_score:.1f}/100.0")
        print(f"Gates Passed: {len(passed_gates)}/{len(self.gates)}")
        print("")
        
        if overall_passed:
            print("🎉 ALL QUALITY GATES PASSED! ✨")
            print("Ready for deployment! 🚀")
        else:
            print("❌ QUALITY GATES FAILED")
            print(f"Failed gates: {', '.join(gate.name for gate in failed_gates)}")
            print("")
            print("🔧 Issues to fix:")
            for gate in failed_gates:
                print(f"  • {gate.name}:")
                for error in gate.errors:
                    print(f"    - {error}")
        
        report = {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'total_duration': total_duration,
            'gates_passed': len(passed_gates),
            'gates_total': len(self.gates),
            'gates': {
                gate.name: {
                    'passed': gate.passed,
                    'score': gate.score,
                    'duration': gate.end_time - gate.start_time,
                    'details': gate.details,
                    'errors': gate.errors
                }
                for gate in self.gates
            }
        }
        
        return overall_passed, report


async def main():
    """Main entry point for quality gate execution."""
    runner = QualityGateRunner()
    
    # Define quality gates
    gates = [
        QualityGate("Unit Tests", "Run comprehensive unit test suite"),
        QualityGate("Integration Tests", "Run integration and functionality tests"),
        QualityGate("Performance Tests", "Verify performance benchmarks"),
        QualityGate("Security Scan", "Check for security vulnerabilities"),
        QualityGate("Code Quality", "Analyze code quality metrics"),
        QualityGate("Memory Leak Detection", "Detect potential memory leaks"),
        QualityGate("Dependency Check", "Verify all dependencies are available"),
        QualityGate("Documentation Check", "Check documentation completeness"),
    ]
    
    for gate in gates:
        runner.add_gate(gate)
    
    # Run all quality gates
    passed, report = await runner.run_all_gates()
    
    # Save report
    report_path = Path(__file__).parent / 'quality_gate_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 Detailed report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())