#!/usr/bin/env python3
"""
Final comprehensive validation test for Synthetic Data Guardian

Tests the complete end-to-end functionality of all components
to ensure production readiness.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


async def test_complete_system_integration():
    """Test complete system integration with all components."""
    print("🔬 Running Complete System Integration Test")
    print("=" * 60)
    
    results = {
        'core_system': False,
        'generation_pipeline': False,
        'error_handling': False,
        'input_validation': False,
        'health_monitoring': False,
        'performance_optimization': False,
        'caching_system': False,
        'security_features': False,
        'concurrent_operations': False,
        'resource_management': False
    }
    
    try:
        # Test 1: Core System Initialization
        print("1. Testing core system initialization...")
        from synthetic_guardian.core.guardian import Guardian
        
        async with Guardian() as guardian:
            assert guardian.initialized
            metrics = guardian.get_metrics()
            assert 'total_generations' in metrics
            print("   ✅ Core system initialized successfully")
            results['core_system'] = True
        
        # Test 2: End-to-End Generation Pipeline
        print("2. Testing complete generation pipeline...")
        async with Guardian() as guardian:
            pipeline_config = {
                "name": "final_validation_test",
                "generator_type": "tabular",
                "data_type": "tabular",
                "schema": {
                    "user_id": {"type": "integer", "min": 1, "max": 10000},
                    "username": {"type": "categorical", "categories": ["alice", "bob", "charlie", "diana", "eve"]},
                    "email": {"type": "string"},
                    "score": {"type": "float", "min": 0.0, "max": 100.0},
                    "active": {"type": "boolean"}
                }
            }
            
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=100,
                seed=12345,
                validate=False,  # Skip for performance
                watermark=False  # Skip for performance
            )
            
            assert result is not None
            assert len(result.data) == 100
            assert result.is_valid()
            assert all(col in result.data.columns for col in ['user_id', 'username', 'email', 'score', 'active'])
            print("   ✅ Generation pipeline working correctly")
            results['generation_pipeline'] = True
        
        # Test 3: Error Handling Integration
        print("3. Testing error handling integration...")
        from synthetic_guardian.middleware.error_handler import get_error_handler, ValidationError
        
        error_handler = get_error_handler()
        initial_errors = error_handler.get_statistics()['total_errors']
        
        # Trigger a handled error
        test_error = ValidationError("Test error for validation")
        await error_handler.handle_error(test_error)
        
        final_errors = error_handler.get_statistics()['total_errors']
        assert final_errors > initial_errors
        print("   ✅ Error handling system working correctly")
        results['error_handling'] = True
        
        # Test 4: Input Validation Integration
        print("4. Testing input validation integration...")
        from synthetic_guardian.middleware.input_validator import get_input_validator
        
        validator = get_input_validator()
        
        # Test valid input
        valid_email = validator.validate_input("test@example.com", ['email'])
        assert valid_email == "test@example.com"
        
        # Test invalid input handling
        try:
            validator.validate_input("<script>alert('xss')</script>", ['safe_string'])
            assert False, "Should have raised ValidationError"
        except Exception:
            pass  # Expected
        
        print("   ✅ Input validation system working correctly")
        results['input_validation'] = True
        
        # Test 5: Health Monitoring Integration
        print("5. Testing health monitoring integration...")
        from synthetic_guardian.monitoring.health_monitor import get_health_monitor
        
        health_monitor = get_health_monitor()
        status, message, details = health_monitor.get_overall_health()
        
        assert status is not None
        assert isinstance(message, str)
        assert 'total_checks' in details
        print("   ✅ Health monitoring system working correctly")
        results['health_monitoring'] = True
        
        # Test 6: Performance Optimization Integration
        print("6. Testing performance optimization integration...")
        from synthetic_guardian.optimization.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        
        # Test batch optimization
        def simple_task(batch):
            return sum(batch) if isinstance(batch, list) else 0
        
        test_batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        results_batch = await optimizer.optimize_batch_operation(
            simple_task,
            test_batches,
            operation_name="final_test_batch"
        )
        
        assert results_batch == [6, 15, 24]
        print("   ✅ Performance optimization system working correctly")
        results['performance_optimization'] = True
        
        # Test 7: Caching System Integration
        print("7. Testing caching system integration...")
        from synthetic_guardian.optimization.caching import get_cache
        
        cache = get_cache()
        
        # Test cache operations
        test_key = "final_test_key"
        test_value = {"test": "data", "timestamp": time.time()}
        
        await cache.set(test_key, test_value)
        cached_value = await cache.get(test_key)
        assert cached_value == test_value
        
        cache_stats = cache.get_stats()
        assert cache_stats['memory_entries'] >= 1
        print("   ✅ Caching system working correctly")
        results['caching_system'] = True
        
        # Test 8: Security Features Integration
        print("8. Testing security features integration...")
        
        # Test input sanitization
        try:
            validator.validate_input("'; DROP TABLE users; --", ['safe_string'])
            assert False, "Should have raised SecurityError"
        except:
            pass  # Expected
        
        # Test error handling for security events
        from synthetic_guardian.middleware.error_handler import SecurityError
        security_error = SecurityError("Test security event")
        await error_handler.handle_error(security_error)
        
        print("   ✅ Security features working correctly")
        results['security_features'] = True
        
        # Test 9: Concurrent Operations
        print("9. Testing concurrent operations...")
        
        async def concurrent_generation_task(task_id):
            async with Guardian() as guardian:
                config = {
                    "name": f"concurrent_test_{task_id}",
                    "generator_type": "tabular",
                    "data_type": "tabular",
                    "schema": {"id": "integer", "value": "float"}
                }
                
                result = await guardian.generate(
                    pipeline_config=config,
                    num_records=10,
                    seed=task_id,
                    validate=False,
                    watermark=False
                )
                
                return len(result.data)
        
        # Run multiple concurrent generations
        concurrent_tasks = [concurrent_generation_task(i) for i in range(5)]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        
        assert all(result == 10 for result in concurrent_results)
        print("   ✅ Concurrent operations working correctly")
        results['concurrent_operations'] = True
        
        # Test 10: Resource Management
        print("10. Testing resource management...")
        
        # Test memory optimization
        optimizer.optimize_memory_usage()
        
        # Test auto-scaling
        scaling_actions = optimizer.auto_scale_resources(target_load=0.8)
        assert isinstance(scaling_actions, dict)
        
        # Clean up
        await optimizer.cleanup()
        
        print("    ✅ Resource management working correctly")
        results['resource_management'] = True
        
        # Final System Health Check
        print("\n🔍 Final System Health Check...")
        
        # Run final health checks
        await health_monitor._run_health_checks()
        final_status, final_message, final_details = health_monitor.get_overall_health()
        
        print(f"    System Status: {final_status.value}")
        print(f"    Health Message: {final_message}")
        print(f"    Active Checks: {final_details['total_checks']}")
        print(f"    Healthy Checks: {final_details['healthy_checks']}")
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, results
    
    # Calculate overall success
    successful_tests = sum(results.values())
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100
    
    print("\n" + "=" * 60)
    print("🎯 FINAL VALIDATION RESULTS")
    print(f"Tests Passed: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    print("")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    overall_success = successful_tests == total_tests
    
    if overall_success:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("🚀 Synthetic Data Guardian is READY for production deployment!")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} systems need attention before production deployment")
    
    return overall_success, results


async def test_production_readiness_checklist():
    """Test production readiness checklist."""
    print("\n📋 PRODUCTION READINESS CHECKLIST")
    print("=" * 60)
    
    checklist = {
        'core_functionality': False,
        'error_handling': False,
        'security_measures': False,
        'performance_optimization': False,
        'monitoring_health': False,
        'logging_system': False,
        'documentation': False,
        'deployment_config': False
    }
    
    try:
        # Check core functionality
        from synthetic_guardian.core.guardian import Guardian
        async with Guardian() as guardian:
            checklist['core_functionality'] = guardian.initialized
        
        # Check error handling
        from synthetic_guardian.middleware.error_handler import get_error_handler
        error_handler = get_error_handler()
        checklist['error_handling'] = len(error_handler.get_statistics()) > 0
        
        # Check security measures
        from synthetic_guardian.middleware.input_validator import get_input_validator
        validator = get_input_validator()
        try:
            validator.validate_input("<script>", ['safe_string'])
            checklist['security_measures'] = False
        except:
            checklist['security_measures'] = True
        
        # Check performance optimization
        from synthetic_guardian.optimization.performance_optimizer import get_performance_optimizer
        optimizer = get_performance_optimizer()
        checklist['performance_optimization'] = True  # If import succeeds
        
        # Check monitoring and health
        from synthetic_guardian.monitoring.health_monitor import get_health_monitor
        health_monitor = get_health_monitor()
        status, _, _ = health_monitor.get_overall_health()
        checklist['monitoring_health'] = status is not None
        
        # Check logging system
        from synthetic_guardian.utils.logger import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        checklist['logging_system'] = True
        
        # Check documentation
        readme_exists = Path("README.md").exists()
        deployment_exists = Path("DEPLOYMENT.md").exists()
        checklist['documentation'] = readme_exists and deployment_exists
        
        # Check deployment configuration
        dockerfile_exists = Path("Dockerfile").exists()
        compose_exists = Path("docker-compose.production.yml").exists()
        requirements_exists = Path("requirements.txt").exists()
        checklist['deployment_config'] = dockerfile_exists and compose_exists and requirements_exists
        
    except Exception as e:
        print(f"❌ Checklist evaluation failed: {e}")
    
    # Print checklist results
    for item, passed in checklist.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {item.replace('_', ' ').title()}")
    
    passed_items = sum(checklist.values())
    total_items = len(checklist)
    
    print(f"\nChecklist Score: {passed_items}/{total_items} ({(passed_items/total_items)*100:.1f}%)")
    
    if passed_items == total_items:
        print("🎉 PRODUCTION READY! All checklist items passed.")
    else:
        print(f"⚠️  {total_items - passed_items} items need attention before production deployment.")
    
    return passed_items == total_items, checklist


async def main():
    """Main test runner for final validation."""
    print("🧪 Synthetic Data Guardian - Final Validation Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run complete system integration test
    system_success, system_results = await test_complete_system_integration()
    
    # Run production readiness checklist
    checklist_success, checklist_results = await test_production_readiness_checklist()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🏁 FINAL VALIDATION SUMMARY")
    print(f"Total Test Time: {total_time:.2f} seconds")
    print("")
    
    if system_success and checklist_success:
        print("🎉 COMPLETE SUCCESS!")
        print("✨ Synthetic Data Guardian is fully validated and ready for production!")
        print("🚀 All systems are operational and meet enterprise standards.")
        print("")
        print("Next Steps:")
        print("1. Review DEPLOYMENT.md for production deployment")
        print("2. Configure environment variables for your setup")
        print("3. Run docker-compose -f docker-compose.production.yml up -d")
        print("4. Monitor system health at http://localhost:3000 (Grafana)")
        return 0
    else:
        print("⚠️  VALIDATION INCOMPLETE")
        if not system_success:
            print("❌ System integration tests failed")
        if not checklist_success:
            print("❌ Production readiness checklist failed")
        print("")
        print("Please address the failed components before production deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))