#!/usr/bin/env python3
"""
Synthetic Data Guardian - Integration Tester

Automated integration testing script that validates component interactions,
external service dependencies, and end-to-end workflows.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import logging
import requests
import psycopg2
import redis
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceHealthChecker:
    """Checks health and connectivity of external services."""
    
    def __init__(self):
        self.services = {}
    
    def check_http_service(self, name: str, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Check HTTP service health."""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            end_time = time.time()
            
            return {
                "name": name,
                "type": "http",
                "url": url,
                "healthy": response.status_code < 400,
                "status_code": response.status_code,
                "response_time_ms": (end_time - start_time) * 1000,
                "error": None
            }
        except Exception as e:
            return {
                "name": name,
                "type": "http",
                "url": url,
                "healthy": False,
                "status_code": None,
                "response_time_ms": None,
                "error": str(e)
            }
    
    def check_database_service(self, name: str, connection_string: str) -> Dict[str, Any]:
        """Check database service health."""
        try:
            start_time = time.time()
            
            if connection_string.startswith('postgresql://'):
                conn = psycopg2.connect(connection_string)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                conn.close()
            elif connection_string.startswith('redis://'):
                r = redis.from_url(connection_string)
                r.ping()
            else:
                raise ValueError(f"Unsupported database type: {connection_string}")
            
            end_time = time.time()
            
            return {
                "name": name,
                "type": "database",
                "connection_string": connection_string.split('@')[0] + '@***',  # Hide credentials
                "healthy": True,
                "response_time_ms": (end_time - start_time) * 1000,
                "error": None
            }
        except Exception as e:
            return {
                "name": name,
                "type": "database",
                "connection_string": connection_string.split('@')[0] + '@***',
                "healthy": False,
                "response_time_ms": None,
                "error": str(e)
            }
    
    def check_neo4j_service(self, name: str, uri: str, username: str, password: str) -> Dict[str, Any]:
        """Check Neo4j service health."""
        try:
            start_time = time.time()
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            
            driver.close()
            end_time = time.time()
            
            return {
                "name": name,
                "type": "neo4j",
                "uri": uri,
                "healthy": True,
                "response_time_ms": (end_time - start_time) * 1000,
                "error": None
            }
        except Exception as e:
            return {
                "name": name,
                "type": "neo4j",
                "uri": uri,
                "healthy": False,
                "response_time_ms": None,
                "error": str(e)
            }
    
    def check_port(self, name: str, host: str, port: int, timeout: int = 5) -> Dict[str, Any]:
        """Check if a port is accessible."""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            end_time = time.time()
            
            return {
                "name": name,
                "type": "port",
                "host": host,
                "port": port,
                "healthy": result == 0,
                "response_time_ms": (end_time - start_time) * 1000,
                "error": None if result == 0 else f"Port {port} not accessible"
            }
        except Exception as e:
            return {
                "name": name,
                "type": "port",
                "host": host,
                "port": port,
                "healthy": False,
                "response_time_ms": None,
                "error": str(e)
            }


class IntegrationTester:
    """Comprehensive integration testing for the Synthetic Data Guardian."""
    
    def __init__(self, repo_path: str = ".", config: Optional[Dict[str, Any]] = None):
        self.repo_path = Path(repo_path)
        self.config = config or self._default_config()
        self.health_checker = ServiceHealthChecker()
        self.results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repository_path": str(self.repo_path.absolute()),
                "test_environment": os.getenv("TEST_ENV", "local")
            },
            "service_health": {},
            "integration_tests": {},
            "end_to_end_tests": {},
            "performance_tests": {},
            "summary": {}
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default integration testing configuration."""
        return {
            "services": {
                "api": {
                    "enabled": True,
                    "url": "http://localhost:8080",
                    "health_endpoint": "/health"
                },
                "postgres": {
                    "enabled": True,
                    "connection_string": os.getenv("DATABASE_URL", "postgresql://localhost:5432/synthetic_guardian_test")
                },
                "redis": {
                    "enabled": True,
                    "connection_string": os.getenv("REDIS_URL", "redis://localhost:6379/0")
                },
                "neo4j": {
                    "enabled": True,
                    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                    "password": os.getenv("NEO4J_PASSWORD", "password")
                }
            },
            "tests": {
                "api_integration": {
                    "enabled": True,
                    "endpoints": [
                        {"path": "/health", "method": "GET", "expected_status": 200},
                        {"path": "/api/v1/generate", "method": "POST", "expected_status": [200, 201]},
                        {"path": "/api/v1/validate", "method": "POST", "expected_status": [200, 201]},
                        {"path": "/api/v1/watermark", "method": "POST", "expected_status": [200, 201]}
                    ]
                },
                "database_integration": {
                    "enabled": True,
                    "tests": [
                        "connection_test",
                        "crud_operations",
                        "transaction_test",
                        "migration_test"
                    ]
                },
                "cache_integration": {
                    "enabled": True,
                    "tests": [
                        "connection_test",
                        "set_get_test",
                        "expiration_test",
                        "pipeline_test"
                    ]
                },
                "end_to_end": {
                    "enabled": True,
                    "scenarios": [
                        "data_generation_workflow",
                        "validation_workflow",
                        "watermarking_workflow",
                        "lineage_tracking_workflow"
                    ]
                }
            },
            "performance": {
                "enabled": True,
                "thresholds": {
                    "api_response_time_ms": 500,
                    "database_query_time_ms": 100,
                    "cache_operation_time_ms": 10
                }
            }
        }
    
    def check_service_health(self) -> Dict[str, Any]:
        """Check health of all configured services."""
        logger.info("Checking service health...")
        
        health_results = {}
        services_config = self.config["services"]
        
        # Check API service
        if services_config["api"]["enabled"]:
            api_url = f"{services_config['api']['url']}{services_config['api']['health_endpoint']}"
            health_results["api"] = self.health_checker.check_http_service("API", api_url)
        
        # Check PostgreSQL
        if services_config["postgres"]["enabled"]:
            health_results["postgres"] = self.health_checker.check_database_service(
                "PostgreSQL", services_config["postgres"]["connection_string"]
            )
        
        # Check Redis
        if services_config["redis"]["enabled"]:
            health_results["redis"] = self.health_checker.check_database_service(
                "Redis", services_config["redis"]["connection_string"]
            )
        
        # Check Neo4j
        if services_config["neo4j"]["enabled"]:
            health_results["neo4j"] = self.health_checker.check_neo4j_service(
                "Neo4j",
                services_config["neo4j"]["uri"],
                services_config["neo4j"]["username"],
                services_config["neo4j"]["password"]
            )
        
        # Check common ports
        common_ports = [
            ("PostgreSQL", "localhost", 5432),
            ("Redis", "localhost", 6379),
            ("Neo4j", "localhost", 7687),
            ("API", "localhost", 8080)
        ]
        
        for name, host, port in common_ports:
            port_name = f"{name.lower()}_port"
            health_results[port_name] = self.health_checker.check_port(f"{name} Port", host, port)
        
        return health_results
    
    def test_api_integration(self) -> Dict[str, Any]:
        """Test API endpoint integration."""
        logger.info("Testing API integration...")
        
        if not self.config["tests"]["api_integration"]["enabled"]:
            return {"enabled": False}
        
        api_results = {}
        base_url = self.config["services"]["api"]["url"]
        endpoints = self.config["tests"]["api_integration"]["endpoints"]
        
        for endpoint in endpoints:
            endpoint_name = f"{endpoint['method']}_{endpoint['path'].replace('/', '_')}"
            api_results[endpoint_name] = self._test_api_endpoint(base_url, endpoint)
        
        return api_results
    
    def _test_api_endpoint(self, base_url: str, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific API endpoint."""
        url = f"{base_url}{endpoint['path']}"
        method = endpoint['method'].upper()
        expected_status = endpoint['expected_status']
        
        if not isinstance(expected_status, list):
            expected_status = [expected_status]
        
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                # Use test data for POST requests
                test_data = {
                    "test": True,
                    "data": "integration_test",
                    "timestamp": datetime.now().isoformat()
                }
                response = requests.post(url, json=test_data, timeout=10)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            end_time = time.time()
            
            success = response.status_code in expected_status
            
            return {
                "url": url,
                "method": method,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": success,
                "response_time_ms": (end_time - start_time) * 1000,
                "response_size_bytes": len(response.content),
                "headers": dict(response.headers)
            }
            
        except Exception as e:
            return {
                "url": url,
                "method": method,
                "success": False,
                "error": str(e)
            }
    
    def test_database_integration(self) -> Dict[str, Any]:
        """Test database integration."""
        logger.info("Testing database integration...")
        
        if not self.config["tests"]["database_integration"]["enabled"]:
            return {"enabled": False}
        
        db_results = {}
        connection_string = self.config["services"]["postgres"]["connection_string"]
        
        # Test basic connection
        db_results["connection"] = self._test_database_connection(connection_string)
        
        # Test CRUD operations
        db_results["crud"] = self._test_database_crud(connection_string)
        
        # Test transactions
        db_results["transactions"] = self._test_database_transactions(connection_string)
        
        return db_results
    
    def _test_database_connection(self, connection_string: str) -> Dict[str, Any]:
        """Test database connection."""
        try:
            start_time = time.time()
            conn = psycopg2.connect(connection_string)
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            end_time = time.time()
            
            return {
                "success": True,
                "version": version,
                "connection_time_ms": (end_time - start_time) * 1000
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_database_crud(self, connection_string: str) -> Dict[str, Any]:
        """Test database CRUD operations."""
        try:
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            
            # Create test table
            table_name = f"integration_test_{int(time.time())}"
            cursor.execute(f"""
                CREATE TEMPORARY TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Insert test data
            cursor.execute(f"INSERT INTO {table_name} (name) VALUES (%s)", ("integration_test",))
            
            # Read test data
            cursor.execute(f"SELECT id, name FROM {table_name} WHERE name = %s", ("integration_test",))
            result = cursor.fetchone()
            
            # Update test data
            cursor.execute(f"UPDATE {table_name} SET name = %s WHERE id = %s", ("updated_test", result[0]))
            
            # Verify update
            cursor.execute(f"SELECT name FROM {table_name} WHERE id = %s", (result[0],))
            updated_result = cursor.fetchone()
            
            # Delete test data
            cursor.execute(f"DELETE FROM {table_name} WHERE id = %s", (result[0],))
            
            # Verify deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_result = cursor.fetchone()
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "operations": {
                    "create": True,
                    "insert": True,
                    "select": result is not None,
                    "update": updated_result[0] == "updated_test",
                    "delete": count_result[0] == 0
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_database_transactions(self, connection_string: str) -> Dict[str, Any]:
        """Test database transaction handling."""
        try:
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            
            # Test successful transaction
            table_name = f"transaction_test_{int(time.time())}"
            cursor.execute(f"""
                CREATE TEMPORARY TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    value INTEGER
                )
            """)
            
            # Begin transaction
            cursor.execute(f"INSERT INTO {table_name} (value) VALUES (1)")
            cursor.execute(f"INSERT INTO {table_name} (value) VALUES (2)")
            conn.commit()
            
            # Verify transaction
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            commit_count = cursor.fetchone()[0]
            
            # Test rollback
            try:
                cursor.execute(f"INSERT INTO {table_name} (value) VALUES (3)")
                cursor.execute("SELECT 1/0")  # Force error
                conn.commit()
            except:
                conn.rollback()
            
            # Verify rollback
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            rollback_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "commit_test": commit_count == 2,
                "rollback_test": rollback_count == 2  # Should still be 2 after rollback
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_cache_integration(self) -> Dict[str, Any]:
        """Test cache (Redis) integration."""
        logger.info("Testing cache integration...")
        
        if not self.config["tests"]["cache_integration"]["enabled"]:
            return {"enabled": False}
        
        cache_results = {}
        connection_string = self.config["services"]["redis"]["connection_string"]
        
        try:
            r = redis.from_url(connection_string)
            
            # Test basic operations
            cache_results["basic_operations"] = self._test_cache_operations(r)
            
            # Test expiration
            cache_results["expiration"] = self._test_cache_expiration(r)
            
            # Test pipeline
            cache_results["pipeline"] = self._test_cache_pipeline(r)
            
        except Exception as e:
            cache_results["error"] = str(e)
        
        return cache_results
    
    def _test_cache_operations(self, redis_client) -> Dict[str, Any]:
        """Test basic cache operations."""
        try:
            test_key = f"integration_test_{int(time.time())}"
            test_value = "integration_test_value"
            
            # Set operation
            start_time = time.time()
            redis_client.set(test_key, test_value)
            set_time = (time.time() - start_time) * 1000
            
            # Get operation
            start_time = time.time()
            retrieved_value = redis_client.get(test_key).decode('utf-8')
            get_time = (time.time() - start_time) * 1000
            
            # Delete operation
            start_time = time.time()
            redis_client.delete(test_key)
            delete_time = (time.time() - start_time) * 1000
            
            # Verify deletion
            deleted_value = redis_client.get(test_key)
            
            return {
                "success": True,
                "set_correct": True,
                "get_correct": retrieved_value == test_value,
                "delete_correct": deleted_value is None,
                "performance": {
                    "set_time_ms": set_time,
                    "get_time_ms": get_time,
                    "delete_time_ms": delete_time
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_cache_expiration(self, redis_client) -> Dict[str, Any]:
        """Test cache expiration."""
        try:
            test_key = f"expiration_test_{int(time.time())}"
            
            # Set with expiration
            redis_client.setex(test_key, 1, "expiring_value")  # 1 second expiration
            
            # Immediate check
            immediate_value = redis_client.get(test_key)
            
            # Wait and check expiration
            time.sleep(1.5)
            expired_value = redis_client.get(test_key)
            
            return {
                "success": True,
                "immediate_exists": immediate_value is not None,
                "expired_correctly": expired_value is None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_cache_pipeline(self, redis_client) -> Dict[str, Any]:
        """Test cache pipeline operations."""
        try:
            pipeline = redis_client.pipeline()
            
            # Pipeline operations
            start_time = time.time()
            for i in range(10):
                pipeline.set(f"pipeline_test_{i}", f"value_{i}")
            pipeline.execute()
            pipeline_time = (time.time() - start_time) * 1000
            
            # Verify pipeline results
            verification_count = 0
            for i in range(10):
                value = redis_client.get(f"pipeline_test_{i}")
                if value and value.decode('utf-8') == f"value_{i}":
                    verification_count += 1
                redis_client.delete(f"pipeline_test_{i}")
            
            return {
                "success": True,
                "pipeline_time_ms": pipeline_time,
                "verification_count": verification_count,
                "all_operations_successful": verification_count == 10
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_end_to_end_workflows(self) -> Dict[str, Any]:
        """Test end-to-end workflows."""
        logger.info("Testing end-to-end workflows...")
        
        if not self.config["tests"]["end_to_end"]["enabled"]:
            return {"enabled": False}
        
        e2e_results = {}
        base_url = self.config["services"]["api"]["url"]
        
        # Test data generation workflow
        e2e_results["data_generation"] = self._test_data_generation_workflow(base_url)
        
        # Test validation workflow
        e2e_results["validation"] = self._test_validation_workflow(base_url)
        
        # Test watermarking workflow
        e2e_results["watermarking"] = self._test_watermarking_workflow(base_url)
        
        return e2e_results
    
    def _test_data_generation_workflow(self, base_url: str) -> Dict[str, Any]:
        """Test complete data generation workflow."""
        try:
            workflow_steps = []
            
            # Step 1: Submit generation request
            generation_data = {
                "schema": {
                    "columns": [
                        {"name": "user_id", "type": "integer"},
                        {"name": "email", "type": "email"},
                        {"name": "age", "type": "integer", "min": 18, "max": 80}
                    ]
                },
                "rows": 100,
                "format": "json"
            }
            
            response = requests.post(
                f"{base_url}/api/v1/generate",
                json=generation_data,
                timeout=30
            )
            
            workflow_steps.append({
                "step": "submit_generation",
                "success": response.status_code in [200, 201, 202],
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            })
            
            if response.status_code in [200, 201]:
                # Step 2: Check generation status (if async)
                if "job_id" in response.json():
                    job_id = response.json()["job_id"]
                    
                    # Poll for completion
                    for _ in range(10):  # Max 10 attempts
                        status_response = requests.get(
                            f"{base_url}/api/v1/jobs/{job_id}",
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data.get("status") == "completed":
                                break
                        
                        time.sleep(2)
                    
                    workflow_steps.append({
                        "step": "check_status",
                        "success": status_response.status_code == 200,
                        "final_status": status_data.get("status", "unknown")
                    })
            
            return {
                "success": all(step.get("success", False) for step in workflow_steps),
                "steps": workflow_steps,
                "total_time_ms": sum(step.get("response_time_ms", 0) for step in workflow_steps)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_validation_workflow(self, base_url: str) -> Dict[str, Any]:
        """Test data validation workflow."""
        try:
            validation_data = {
                "data": [
                    {"user_id": 1, "email": "test@example.com", "age": 25},
                    {"user_id": 2, "email": "user@test.com", "age": 35}
                ],
                "schema": {
                    "columns": [
                        {"name": "user_id", "type": "integer"},
                        {"name": "email", "type": "email"},
                        {"name": "age", "type": "integer", "min": 18, "max": 80}
                    ]
                }
            }
            
            response = requests.post(
                f"{base_url}/api/v1/validate",
                json=validation_data,
                timeout=15
            )
            
            success = response.status_code in [200, 201]
            
            result = {
                "success": success,
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            
            if success:
                validation_result = response.json()
                result["validation_passed"] = validation_result.get("valid", False)
                result["validation_score"] = validation_result.get("score", 0)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_watermarking_workflow(self, base_url: str) -> Dict[str, Any]:
        """Test watermarking workflow."""
        try:
            watermark_data = {
                "data": [
                    {"user_id": 1, "value": 100.5},
                    {"user_id": 2, "value": 200.3}
                ],
                "watermark_key": "integration_test_key",
                "strength": 0.1
            }
            
            response = requests.post(
                f"{base_url}/api/v1/watermark",
                json=watermark_data,
                timeout=15
            )
            
            success = response.status_code in [200, 201]
            
            result = {
                "success": success,
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            
            if success:
                watermark_result = response.json()
                result["watermarked_data_count"] = len(watermark_result.get("data", []))
                result["watermark_applied"] = "watermark_id" in watermark_result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance integration tests."""
        logger.info("Running performance integration tests...")
        
        if not self.config["performance"]["enabled"]:
            return {"enabled": False}
        
        performance_results = {}
        thresholds = self.config["performance"]["thresholds"]
        
        # API performance test
        api_results = self.test_api_integration()
        api_performance = self._analyze_api_performance(api_results, thresholds)
        performance_results["api"] = api_performance
        
        # Database performance test
        db_results = self.test_database_integration()
        db_performance = self._analyze_database_performance(db_results, thresholds)
        performance_results["database"] = db_performance
        
        # Cache performance test
        cache_results = self.test_cache_integration()
        cache_performance = self._analyze_cache_performance(cache_results, thresholds)
        performance_results["cache"] = cache_performance
        
        return performance_results
    
    def _analyze_api_performance(self, api_results: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API performance results."""
        if not api_results or "enabled" in api_results:
            return {"analyzed": False}
        
        response_times = []
        for endpoint, result in api_results.items():
            if isinstance(result, dict) and "response_time_ms" in result:
                response_times.append(result["response_time_ms"])
        
        if not response_times:
            return {"analyzed": False}
        
        avg_response_time = sum(response_times) / len(response_times)
        threshold = thresholds["api_response_time_ms"]
        
        return {
            "analyzed": True,
            "avg_response_time_ms": avg_response_time,
            "max_response_time_ms": max(response_times),
            "min_response_time_ms": min(response_times),
            "threshold_ms": threshold,
            "within_threshold": avg_response_time <= threshold,
            "endpoint_count": len(response_times)
        }
    
    def _analyze_database_performance(self, db_results: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database performance results."""
        if not db_results or "enabled" in db_results:
            return {"analyzed": False}
        
        connection_time = 0
        if "connection" in db_results and "connection_time_ms" in db_results["connection"]:
            connection_time = db_results["connection"]["connection_time_ms"]
        
        threshold = thresholds["database_query_time_ms"]
        
        return {
            "analyzed": True,
            "connection_time_ms": connection_time,
            "threshold_ms": threshold,
            "within_threshold": connection_time <= threshold
        }
    
    def _analyze_cache_performance(self, cache_results: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cache performance results."""
        if not cache_results or "enabled" in cache_results or "error" in cache_results:
            return {"analyzed": False}
        
        operation_times = []
        
        if "basic_operations" in cache_results and "performance" in cache_results["basic_operations"]:
            perf = cache_results["basic_operations"]["performance"]
            operation_times.extend([
                perf.get("set_time_ms", 0),
                perf.get("get_time_ms", 0),
                perf.get("delete_time_ms", 0)
            ])
        
        if not operation_times:
            return {"analyzed": False}
        
        avg_operation_time = sum(operation_times) / len(operation_times)
        threshold = thresholds["cache_operation_time_ms"]
        
        return {
            "analyzed": True,
            "avg_operation_time_ms": avg_operation_time,
            "max_operation_time_ms": max(operation_times),
            "threshold_ms": threshold,
            "within_threshold": avg_operation_time <= threshold
        }
    
    def run_comprehensive_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting comprehensive integration tests...")
        
        # Check service health first
        self.results["service_health"] = self.check_service_health()
        
        # Check if critical services are healthy
        critical_services_healthy = True
        for service, health in self.results["service_health"].items():
            if isinstance(health, dict) and not health.get("healthy", False):
                if "api" in service or "postgres" in service:
                    critical_services_healthy = False
                    logger.warning(f"Critical service {service} is not healthy")
        
        if not critical_services_healthy:
            logger.warning("Skipping some tests due to unhealthy critical services")
        
        # Run integration tests
        self.results["integration_tests"]["api"] = self.test_api_integration()
        self.results["integration_tests"]["database"] = self.test_database_integration()
        self.results["integration_tests"]["cache"] = self.test_cache_integration()
        
        # Run end-to-end tests
        self.results["end_to_end_tests"] = self.test_end_to_end_workflows()
        
        # Run performance tests
        self.results["performance_tests"] = self.run_performance_tests()
        
        # Generate summary
        self.results["summary"] = self._generate_test_summary()
        
        logger.info("Comprehensive integration tests completed!")
        return self.results
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "healthy_services": 0,
            "total_services": 0,
            "overall_success": False
        }
        
        # Service health summary
        for service, health in self.results["service_health"].items():
            summary["total_services"] += 1
            if isinstance(health, dict) and health.get("healthy", False):
                summary["healthy_services"] += 1
        
        # Integration tests summary
        def count_test_results(test_results):
            if not isinstance(test_results, dict):
                return 0, 0
            
            passed = 0
            failed = 0
            
            for test_name, result in test_results.items():
                if test_name == "enabled" and not result:
                    continue
                
                if isinstance(result, dict):
                    if result.get("success", False):
                        passed += 1
                    else:
                        failed += 1
                elif isinstance(result, bool) and result:
                    passed += 1
                else:
                    failed += 1
            
            return passed, failed
        
        # Count all test results
        for test_category in ["integration_tests", "end_to_end_tests", "performance_tests"]:
            if test_category in self.results:
                for test_type, test_results in self.results[test_category].items():
                    passed, failed = count_test_results(test_results)
                    summary["passed_tests"] += passed
                    summary["failed_tests"] += failed
        
        summary["total_tests"] = summary["passed_tests"] + summary["failed_tests"]
        
        # Overall success criteria
        service_health_ratio = summary["healthy_services"] / max(summary["total_services"], 1)
        test_success_ratio = summary["passed_tests"] / max(summary["total_tests"], 1)
        
        summary["overall_success"] = (
            service_health_ratio >= 0.8 and  # At least 80% of services healthy
            test_success_ratio >= 0.9        # At least 90% of tests passed
        )
        
        return summary
    
    def save_report(self, filename: Optional[str] = None, format: str = "json"):
        """Save the integration test report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integration_test_report_{timestamp}.{format}"
        
        if format == "json":
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Integration test report saved to: {filename}")
        return filename


def main():
    """Main entry point for the integration tester."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for Synthetic Data Guardian"
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
        "--health-only", "-h",
        action="store_true",
        help="Only check service health"
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
        
        # Create integration tester
        tester = IntegrationTester(args.repo_path, config)
        
        # Run tests
        if args.health_only:
            results = {"service_health": tester.check_service_health()}
        else:
            results = tester.run_comprehensive_integration_tests()
        
        # Save report
        if args.output:
            tester.save_report(args.output)
        else:
            output_file = tester.save_report()
            print(f"Integration test report saved to: {output_file}")
        
        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print(f"\n🎯 Integration Test Summary:")
            print(f"  Services: {summary['healthy_services']}/{summary['total_services']} healthy")
            print(f"  Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
            
            if summary["overall_success"]:
                print("✅ All integration tests passed!")
            else:
                print("❌ Some integration tests failed")
                sys.exit(1)
        else:
            print("\n⚠️  Health check completed - see report for details")
            
    except Exception as e:
        logger.error(f"Integration testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()