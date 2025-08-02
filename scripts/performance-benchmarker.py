#!/usr/bin/env python3
"""
Synthetic Data Guardian - Performance Benchmarker

Automated performance testing and benchmarking for the Synthetic Data Guardian project.
Measures build times, test execution times, API response times, and resource usage.
"""

import json
import os
import psutil
import subprocess
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import argparse
import logging
import requests
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitors system resource usage during benchmarks."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.thread = None
    
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)
        
        if not self.data:
            return {}
        
        # Calculate statistics
        cpu_values = [d["cpu_percent"] for d in self.data]
        memory_values = [d["memory_mb"] for d in self.data]
        
        return {
            "duration_seconds": len(self.data) * self.interval,
            "samples": len(self.data),
            "cpu": {
                "avg_percent": sum(cpu_values) / len(cpu_values),
                "max_percent": max(cpu_values),
                "min_percent": min(cpu_values)
            },
            "memory": {
                "avg_mb": sum(memory_values) / len(memory_values),
                "max_mb": max(memory_values),
                "min_mb": min(memory_values)
            },
            "raw_data": self.data
        }
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                self.data.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.used / (1024 * 1024),
                    "memory_percent": memory_info.percent
                })
                
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking for the project."""
    
    def __init__(self, repo_path: str = ".", config: Optional[Dict[str, Any]] = None):
        self.repo_path = Path(repo_path)
        self.config = config or self._default_config()
        self.results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repository_path": str(self.repo_path.absolute()),
                "system_info": self._get_system_info()
            },
            "benchmarks": {}
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default benchmarking configuration."""
        return {
            "build": {
                "enabled": True,
                "commands": {
                    "npm": ["npm", "run", "build"],
                    "python": ["python", "setup.py", "build"]
                },
                "timeout": 300
            },
            "tests": {
                "enabled": True,
                "commands": {
                    "npm": ["npm", "test"],
                    "python": ["python", "-m", "pytest"],
                    "python_coverage": ["python", "-m", "pytest", "--cov"]
                },
                "timeout": 600
            },
            "api": {
                "enabled": False,
                "base_url": "http://localhost:8080",
                "endpoints": [
                    {"path": "/health", "method": "GET"},
                    {"path": "/api/v1/generate", "method": "POST", "payload": {"test": "data"}},
                    {"path": "/api/v1/validate", "method": "POST", "payload": {"test": "data"}}
                ],
                "concurrent_users": [1, 5, 10],
                "duration_seconds": 30
            },
            "load": {
                "enabled": False,
                "scenarios": [
                    {"name": "light", "users": 10, "duration": 60},
                    {"name": "medium", "users": 50, "duration": 120},
                    {"name": "heavy", "users": 100, "duration": 180}
                ]
            },
            "memory": {
                "enabled": True,
                "heap_profile": True
            }
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
            "platform": sys.platform,
            "python_version": sys.version
        }
    
    @contextmanager
    def _monitor_resources(self):
        """Context manager for resource monitoring."""
        monitor = ResourceMonitor()
        monitor.start()
        try:
            yield monitor
        finally:
            resource_data = monitor.stop()
            return resource_data
    
    def run_command_benchmark(self, command: List[str], name: str, timeout: int = 300) -> Dict[str, Any]:
        """Benchmark a command execution."""
        logger.info(f"Benchmarking command: {' '.join(command)}")
        
        with self._monitor_resources() as monitor:
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    command,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Get resource usage
                resource_data = monitor.stop()
                
                return {
                    "name": name,
                    "command": command,
                    "success": result.returncode == 0,
                    "duration_seconds": duration,
                    "return_code": result.returncode,
                    "stdout_lines": len(result.stdout.split('\n')) if result.stdout else 0,
                    "stderr_lines": len(result.stderr.split('\n')) if result.stderr else 0,
                    "resource_usage": resource_data,
                    "stdout": result.stdout[:1000] if result.stdout else "",  # Truncate
                    "stderr": result.stderr[:1000] if result.stderr else ""   # Truncate
                }
                
            except subprocess.TimeoutExpired:
                end_time = time.time()
                resource_data = monitor.stop()
                
                return {
                    "name": name,
                    "command": command,
                    "success": False,
                    "duration_seconds": end_time - start_time,
                    "return_code": -1,
                    "error": "Command timed out",
                    "resource_usage": resource_data
                }
                
            except Exception as e:
                end_time = time.time()
                resource_data = monitor.stop()
                
                return {
                    "name": name,
                    "command": command,
                    "success": False,
                    "duration_seconds": end_time - start_time,
                    "return_code": -1,
                    "error": str(e),
                    "resource_usage": resource_data
                }
    
    def benchmark_build_performance(self) -> Dict[str, Any]:
        """Benchmark build performance."""
        logger.info("Benchmarking build performance...")
        
        if not self.config["build"]["enabled"]:
            return {"enabled": False}
        
        build_results = {}
        
        # Check for different build systems
        build_commands = self.config["build"]["commands"]
        timeout = self.config["build"]["timeout"]
        
        # NPM build
        if (self.repo_path / "package.json").exists() and "npm" in build_commands:
            build_results["npm"] = self.run_command_benchmark(
                build_commands["npm"], "npm_build", timeout
            )
        
        # Python build
        if (self.repo_path / "setup.py").exists() and "python" in build_commands:
            build_results["python"] = self.run_command_benchmark(
                build_commands["python"], "python_build", timeout
            )
        
        # Docker build (if Dockerfile exists)
        if (self.repo_path / "Dockerfile").exists():
            docker_build_cmd = ["docker", "build", "-t", "sdg-benchmark", "."]
            build_results["docker"] = self.run_command_benchmark(
                docker_build_cmd, "docker_build", timeout * 2
            )
        
        return build_results
    
    def benchmark_test_performance(self) -> Dict[str, Any]:
        """Benchmark test execution performance."""
        logger.info("Benchmarking test performance...")
        
        if not self.config["tests"]["enabled"]:
            return {"enabled": False}
        
        test_results = {}
        test_commands = self.config["tests"]["commands"]
        timeout = self.config["tests"]["timeout"]
        
        # NPM tests
        if (self.repo_path / "package.json").exists() and "npm" in test_commands:
            test_results["npm"] = self.run_command_benchmark(
                test_commands["npm"], "npm_test", timeout
            )
        
        # Python tests
        if ((self.repo_path / "pytest.ini").exists() or 
            (self.repo_path / "pyproject.toml").exists()) and "python" in test_commands:
            test_results["python"] = self.run_command_benchmark(
                test_commands["python"], "python_test", timeout
            )
            
            # Python tests with coverage
            if "python_coverage" in test_commands:
                test_results["python_coverage"] = self.run_command_benchmark(
                    test_commands["python_coverage"], "python_test_coverage", timeout
                )
        
        return test_results
    
    def benchmark_api_performance(self) -> Dict[str, Any]:
        """Benchmark API endpoint performance."""
        logger.info("Benchmarking API performance...")
        
        if not self.config["api"]["enabled"]:
            return {"enabled": False}
        
        api_results = {
            "endpoints": {},
            "load_tests": {}
        }
        
        base_url = self.config["api"]["base_url"]
        endpoints = self.config["api"]["endpoints"]
        
        # Test individual endpoints
        for endpoint in endpoints:
            endpoint_name = f"{endpoint['method']}_{endpoint['path'].replace('/', '_')}"
            api_results["endpoints"][endpoint_name] = self._benchmark_endpoint(
                base_url, endpoint
            )
        
        # Load testing
        concurrent_users = self.config["api"]["concurrent_users"]
        duration = self.config["api"]["duration_seconds"]
        
        for user_count in concurrent_users:
            load_test_name = f"load_test_{user_count}_users"
            api_results["load_tests"][load_test_name] = self._benchmark_load(
                base_url, endpoints[0] if endpoints else {"path": "/health", "method": "GET"},
                user_count, duration
            )
        
        return api_results
    
    def _benchmark_endpoint(self, base_url: str, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single API endpoint."""
        url = f"{base_url}{endpoint['path']}"
        method = endpoint.get('method', 'GET').upper()
        payload = endpoint.get('payload')
        
        # Warmup requests
        for _ in range(3):
            try:
                if method == 'GET':
                    requests.get(url, timeout=5)
                elif method == 'POST':
                    requests.post(url, json=payload, timeout=5)
            except:
                pass
        
        # Benchmark requests
        durations = []
        status_codes = []
        errors = 0
        
        for _ in range(10):  # 10 requests per endpoint
            start_time = time.time()
            
            try:
                if method == 'GET':
                    response = requests.get(url, timeout=5)
                elif method == 'POST':
                    response = requests.post(url, json=payload, timeout=5)
                else:
                    continue
                
                end_time = time.time()
                durations.append(end_time - start_time)
                status_codes.append(response.status_code)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Request failed: {e}")
        
        if durations:
            return {
                "url": url,
                "method": method,
                "request_count": len(durations),
                "error_count": errors,
                "avg_response_time_ms": (sum(durations) / len(durations)) * 1000,
                "min_response_time_ms": min(durations) * 1000,
                "max_response_time_ms": max(durations) * 1000,
                "status_codes": status_codes,
                "success_rate": (len(durations) / (len(durations) + errors)) * 100
            }
        else:
            return {
                "url": url,
                "method": method,
                "error": "All requests failed",
                "error_count": errors
            }
    
    def _benchmark_load(self, base_url: str, endpoint: Dict[str, Any], 
                       user_count: int, duration: int) -> Dict[str, Any]:
        """Benchmark API under load."""
        url = f"{base_url}{endpoint['path']}"
        method = endpoint.get('method', 'GET').upper()
        payload = endpoint.get('payload')
        
        results = {
            "user_count": user_count,
            "duration_seconds": duration,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "requests_per_second": 0
        }
        
        def worker():
            """Worker function for load testing."""
            session = requests.Session()
            while time.time() < end_time:
                start_request = time.time()
                
                try:
                    if method == 'GET':
                        response = session.get(url, timeout=10)
                    elif method == 'POST':
                        response = session.post(url, json=payload, timeout=10)
                    else:
                        continue
                    
                    end_request = time.time()
                    
                    with lock:
                        results["total_requests"] += 1
                        if response.status_code < 400:
                            results["successful_requests"] += 1
                        else:
                            results["failed_requests"] += 1
                        results["response_times"].append(end_request - start_request)
                        
                except Exception:
                    with lock:
                        results["total_requests"] += 1
                        results["failed_requests"] += 1
        
        # Start load test
        import threading
        lock = threading.Lock()
        end_time = time.time() + duration
        threads = []
        
        logger.info(f"Starting load test: {user_count} users for {duration} seconds")
        
        for _ in range(user_count):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Calculate statistics
        if results["response_times"]:
            response_times = results["response_times"]
            results["avg_response_time_ms"] = (sum(response_times) / len(response_times)) * 1000
            results["min_response_time_ms"] = min(response_times) * 1000
            results["max_response_time_ms"] = max(response_times) * 1000
            results["requests_per_second"] = results["total_requests"] / duration
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            length = len(sorted_times)
            results["p50_response_time_ms"] = sorted_times[int(length * 0.5)] * 1000
            results["p95_response_time_ms"] = sorted_times[int(length * 0.95)] * 1000
            results["p99_response_time_ms"] = sorted_times[int(length * 0.99)] * 1000
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("Benchmarking memory usage...")
        
        if not self.config["memory"]["enabled"]:
            return {"enabled": False}
        
        memory_results = {}
        
        # Baseline memory usage
        baseline_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_results["baseline_mb"] = baseline_memory
        
        # Memory usage during build (if possible)
        if (self.repo_path / "package.json").exists():
            build_cmd = ["npm", "run", "build"]
            build_memory = self._measure_memory_during_command(build_cmd)
            memory_results["build_memory"] = build_memory
        
        # Memory usage during tests
        test_commands = []
        if (self.repo_path / "package.json").exists():
            test_commands.append(["npm", "test"])
        if (self.repo_path / "pytest.ini").exists():
            test_commands.append(["python", "-m", "pytest"])
        
        for i, cmd in enumerate(test_commands):
            test_memory = self._measure_memory_during_command(cmd)
            memory_results[f"test_memory_{i}"] = test_memory
        
        return memory_results
    
    def _measure_memory_during_command(self, command: List[str]) -> Dict[str, Any]:
        """Measure memory usage during command execution."""
        monitor = ResourceMonitor(interval=0.5)
        monitor.start()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                timeout=300
            )
            
            resource_data = monitor.stop()
            
            return {
                "command": command,
                "success": result.returncode == 0,
                "peak_memory_mb": resource_data.get("memory", {}).get("max_mb", 0),
                "avg_memory_mb": resource_data.get("memory", {}).get("avg_mb", 0),
                "duration_seconds": resource_data.get("duration_seconds", 0)
            }
            
        except Exception as e:
            monitor.stop()
            return {
                "command": command,
                "success": False,
                "error": str(e)
            }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all enabled benchmarks."""
        logger.info("Starting comprehensive performance benchmark...")
        
        # Build performance
        if self.config["build"]["enabled"]:
            self.results["benchmarks"]["build"] = self.benchmark_build_performance()
        
        # Test performance
        if self.config["tests"]["enabled"]:
            self.results["benchmarks"]["tests"] = self.benchmark_test_performance()
        
        # API performance
        if self.config["api"]["enabled"]:
            self.results["benchmarks"]["api"] = self.benchmark_api_performance()
        
        # Memory usage
        if self.config["memory"]["enabled"]:
            self.results["benchmarks"]["memory"] = self.benchmark_memory_usage()
        
        # Calculate overall scores
        self.results["performance_scores"] = self._calculate_performance_scores()
        
        logger.info("Comprehensive benchmark completed!")
        return self.results
    
    def _calculate_performance_scores(self) -> Dict[str, Any]:
        """Calculate performance scores based on benchmark results."""
        scores = {
            "build_score": 0,
            "test_score": 0,
            "api_score": 0,
            "memory_score": 0,
            "overall_score": 0
        }
        
        benchmarks = self.results["benchmarks"]
        
        # Build score (based on build time)
        if "build" in benchmarks:
            build_times = []
            for build_type, result in benchmarks["build"].items():
                if isinstance(result, dict) and "duration_seconds" in result:
                    build_times.append(result["duration_seconds"])
            
            if build_times:
                avg_build_time = sum(build_times) / len(build_times)
                # Score: 100 for < 30s, 50 for 60s, 0 for > 300s
                scores["build_score"] = max(0, min(100, 100 - (avg_build_time - 30) * 2))
        
        # Test score (based on test time)
        if "tests" in benchmarks:
            test_times = []
            for test_type, result in benchmarks["tests"].items():
                if isinstance(result, dict) and "duration_seconds" in result:
                    test_times.append(result["duration_seconds"])
            
            if test_times:
                avg_test_time = sum(test_times) / len(test_times)
                # Score: 100 for < 60s, 50 for 120s, 0 for > 600s
                scores["test_score"] = max(0, min(100, 100 - (avg_test_time - 60) / 5.4))
        
        # API score (based on response time)
        if "api" in benchmarks and "endpoints" in benchmarks["api"]:
            response_times = []
            for endpoint, result in benchmarks["api"]["endpoints"].items():
                if isinstance(result, dict) and "avg_response_time_ms" in result:
                    response_times.append(result["avg_response_time_ms"])
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                # Score: 100 for < 100ms, 50 for 500ms, 0 for > 2000ms
                scores["api_score"] = max(0, min(100, 100 - (avg_response_time - 100) / 19))
        
        # Memory score (based on memory efficiency)
        if "memory" in benchmarks:
            memory_usage = benchmarks["memory"].get("baseline_mb", 0)
            # Score: 100 for < 500MB, 50 for 1GB, 0 for > 4GB
            scores["memory_score"] = max(0, min(100, 100 - (memory_usage - 500) / 35))
        
        # Overall score (weighted average)
        valid_scores = [s for s in scores.values() if s > 0]
        if valid_scores:
            scores["overall_score"] = sum(valid_scores) / len(valid_scores)
        
        return scores
    
    def generate_report(self, format: str = "summary") -> str:
        """Generate a performance benchmark report."""
        if format == "json":
            return json.dumps(self.results, indent=2, default=str)
        elif format == "summary":
            return self._generate_summary_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SYNTHETIC DATA GUARDIAN - PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {self.results['metadata']['timestamp']}")
        lines.append(f"Repository: {self.results['metadata']['repository_path']}")
        lines.append("")
        
        # System info
        system_info = self.results["metadata"]["system_info"]
        lines.append("💻 SYSTEM INFORMATION")
        lines.append("-" * 30)
        lines.append(f"CPU Cores: {system_info.get('cpu_count', 'N/A')}")
        lines.append(f"Memory: {system_info.get('memory_total_gb', 0):.1f} GB")
        lines.append(f"Platform: {system_info.get('platform', 'N/A')}")
        lines.append("")
        
        # Performance scores
        if "performance_scores" in self.results:
            lines.append("🏆 PERFORMANCE SCORES")
            lines.append("-" * 30)
            scores = self.results["performance_scores"]
            
            for score_name, score_value in scores.items():
                if score_value > 0:
                    score_str = f"{score_value:.1f}/100"
                    status = "🟢" if score_value >= 80 else "🟡" if score_value >= 60 else "🔴"
                    lines.append(f"{status} {score_name.replace('_', ' ').title()}: {score_str}")
            lines.append("")
        
        # Build performance
        if "build" in self.results["benchmarks"]:
            lines.append("🔨 BUILD PERFORMANCE")
            lines.append("-" * 30)
            for build_type, result in self.results["benchmarks"]["build"].items():
                if isinstance(result, dict):
                    success = "✅" if result.get("success", False) else "❌"
                    duration = result.get("duration_seconds", 0)
                    lines.append(f"{success} {build_type.upper()}: {duration:.1f}s")
            lines.append("")
        
        # Test performance
        if "tests" in self.results["benchmarks"]:
            lines.append("🧪 TEST PERFORMANCE")
            lines.append("-" * 30)
            for test_type, result in self.results["benchmarks"]["tests"].items():
                if isinstance(result, dict):
                    success = "✅" if result.get("success", False) else "❌"
                    duration = result.get("duration_seconds", 0)
                    lines.append(f"{success} {test_type.upper()}: {duration:.1f}s")
            lines.append("")
        
        # API performance
        if "api" in self.results["benchmarks"] and "endpoints" in self.results["benchmarks"]["api"]:
            lines.append("🌐 API PERFORMANCE")
            lines.append("-" * 30)
            for endpoint, result in self.results["benchmarks"]["api"]["endpoints"].items():
                if isinstance(result, dict) and "avg_response_time_ms" in result:
                    avg_time = result["avg_response_time_ms"]
                    success_rate = result.get("success_rate", 0)
                    lines.append(f"{endpoint}: {avg_time:.1f}ms (success: {success_rate:.1f}%)")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("End of Report")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, filename: Optional[str] = None, format: str = "json"):
        """Save the benchmark report to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}.{format}"
        
        report_content = self.generate_report(format)
        
        with open(filename, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Performance benchmark report saved to: {filename}")
        return filename


def main():
    """Main entry point for the performance benchmarker."""
    parser = argparse.ArgumentParser(
        description="Benchmark performance of Synthetic Data Guardian project"
    )
    parser.add_argument(
        "--repo-path", "-r",
        default=".",
        help="Path to the repository (default: current directory)"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "summary"],
        default="summary",
        help="Output format (default: summary)"
    )
    parser.add_argument(
        "--build-only", "-b",
        action="store_true",
        help="Only benchmark build performance"
    )
    parser.add_argument(
        "--test-only", "-t",
        action="store_true",
        help="Only benchmark test performance"
    )
    parser.add_argument(
        "--api-only", "-a",
        action="store_true",
        help="Only benchmark API performance"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Create benchmarker
        benchmarker = PerformanceBenchmarker(args.repo_path, config)
        
        # Adjust config based on command line options
        if args.build_only:
            benchmarker.config["tests"]["enabled"] = False
            benchmarker.config["api"]["enabled"] = False
            benchmarker.config["memory"]["enabled"] = False
        elif args.test_only:
            benchmarker.config["build"]["enabled"] = False
            benchmarker.config["api"]["enabled"] = False
            benchmarker.config["memory"]["enabled"] = False
        elif args.api_only:
            benchmarker.config["build"]["enabled"] = False
            benchmarker.config["tests"]["enabled"] = False
            benchmarker.config["memory"]["enabled"] = False
        
        # Run benchmarks
        results = benchmarker.run_comprehensive_benchmark()
        
        # Save report
        if args.output:
            benchmarker.save_report(args.output, args.format)
        else:
            output_file = benchmarker.save_report(format=args.format)
            print(f"Benchmark report saved to: {output_file}")
        
        # Print summary
        if args.format == "summary":
            print("\n" + benchmarker.generate_report("summary"))
        
        # Performance summary
        if "performance_scores" in results:
            overall_score = results["performance_scores"]["overall_score"]
            print(f"\n🎯 Overall Performance Score: {overall_score:.1f}/100")
            
            if overall_score >= 80:
                print("✅ Excellent performance!")
            elif overall_score >= 60:
                print("⚠️  Good performance with room for improvement")
            else:
                print("🚨 Performance needs optimization")
        
    except Exception as e:
        logger.error(f"Performance benchmarking failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()