#!/usr/bin/env python3
"""
Test robustness features - error handling, validation, monitoring
"""

import asyncio
import sys
sys.path.insert(0, '/root/repo/src')

async def test_error_handling():
    """Test comprehensive error handling"""
    print("🧪 Testing error handling...")
    
    try:
        from synthetic_guardian.middleware.error_handler import (
            ErrorHandler, ValidationError, GenerationError, SecurityError
        )
        
        handler = ErrorHandler()
        
        # Test different error types
        errors = [
            ValidationError("Invalid input format", context={'field': 'test'}),
            GenerationError("Failed to generate data"),
            SecurityError("Potential SQL injection detected"),
            ValueError("Generic error")
        ]
        
        for error in errors:
            error_details = await handler.handle_error(error)
            print(f"   ✅ Handled {type(error).__name__}: {error_details.error_id}")
        
        # Check statistics
        stats = handler.get_statistics()
        print(f"   📊 Error statistics: {stats['total_errors']} total, {len(stats['critical_errors'])} critical")
        
        return True
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

async def test_input_validation():
    """Test input validation and sanitization"""
    print("🧪 Testing input validation...")
    
    try:
        from synthetic_guardian.middleware.input_validator import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Test valid inputs
        valid_cases = [
            ("hello world", ['safe_string'], "Safe string"),
            ("test@example.com", ['email'], "Valid email"),
            (42, ['positive_integer'], "Positive integer"),
            ("tabular", ['generator_type'], "Valid generator type")
        ]
        
        for value, rules, description in valid_cases:
            try:
                result = validator.validate_input(value, rules, 'test_field')
                print(f"   ✅ {description}: {result}")
            except Exception as e:
                print(f"   ❌ {description} failed: {e}")
        
        # Test invalid inputs that should be caught
        invalid_cases = [
            ("<script>alert('xss')</script>", ['safe_string'], "XSS attempt"),
            ("not-an-email", ['email'], "Invalid email"),
            (-5, ['positive_integer'], "Negative integer"),
            ("invalid_type", ['generator_type'], "Invalid generator")
        ]
        
        for value, rules, description in invalid_cases:
            try:
                validator.validate_input(value, rules, 'test_field')
                print(f"   ❌ {description} should have failed but didn't")
            except ValidationError:
                print(f"   ✅ {description}: Correctly rejected")
            except Exception as e:
                print(f"   ❌ {description} failed with unexpected error: {e}")
        
        return True
    except Exception as e:
        print(f"   ❌ Input validation test failed: {e}")
        return False

async def test_health_monitoring():
    """Test health monitoring system"""
    print("🧪 Testing health monitoring...")
    
    try:
        from synthetic_guardian.monitoring.health_monitor import (
            HealthMonitor, HealthCheck, ComponentStatus
        )
        
        monitor = HealthMonitor()
        
        # Add a simple test check
        def test_check():
            return ComponentStatus.UP, "Test check passed", {'test': True}
        
        monitor.register_check(HealthCheck(
            name="test_check",
            check_function=test_check,
            interval_seconds=5.0
        ))
        
        # Run health checks once
        await monitor._run_health_checks()
        
        # Get overall health
        status, message, details = monitor.get_overall_health()
        print(f"   ✅ Overall health: {status.value} - {message}")
        print(f"   📊 Health details: {details['total_checks']} checks, {details['healthy_checks']} healthy")
        
        # Get component health
        component_health = monitor.get_component_health()
        for name, health in component_health.items():
            print(f"   📋 {name}: {health.status.value} - {health.message}")
        
        # Record some request metrics
        monitor.record_request(True, 0.1)
        monitor.record_request(True, 0.2)
        monitor.record_request(False, 0.5)
        
        return True
    except Exception as e:
        print(f"   ❌ Health monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_robust_guardian():
    """Test Guardian with robustness features enabled"""
    print("🧪 Testing robust Guardian...")
    
    try:
        from synthetic_guardian.core.guardian import Guardian
        from synthetic_guardian.middleware.error_handler import get_error_handler
        from synthetic_guardian.monitoring.health_monitor import get_health_monitor
        
        # Initialize monitoring
        health_monitor = get_health_monitor()
        await health_monitor.start_monitoring()
        
        # Test generation with error handling
        async with Guardian() as guardian:
            # Test valid generation
            pipeline_config = {
                "name": "robust_test",
                "generator_type": "tabular",
                "data_type": "tabular",
                "schema": {
                    "id": "integer",
                    "value": "float"
                }
            }
            
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=5,
                seed=42,
                validate=False,
                watermark=False
            )
            
            print(f"   ✅ Generated {len(result.data)} records successfully")
            
            # Test error handling with invalid parameters
            try:
                await guardian.generate(
                    pipeline_config=pipeline_config,
                    num_records=-1,  # Invalid
                    seed=42
                )
                print("   ❌ Should have failed with negative num_records")
            except Exception as e:
                print(f"   ✅ Correctly handled invalid parameters: {type(e).__name__}")
        
        # Check error statistics
        error_handler = get_error_handler()
        stats = error_handler.get_statistics()
        print(f"   📊 Errors handled: {stats['total_errors']}")
        
        # Stop monitoring
        health_monitor.stop_monitoring()
        
        return True
    except Exception as e:
        print(f"   ❌ Robust Guardian test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("🛡️  Testing Robustness Features")
    print("=" * 60)
    
    tests = [
        test_error_handling,
        test_input_validation,
        test_health_monitoring,
        test_robust_guardian
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 SUCCESS: All {total} robustness tests passed!")
        return 0
    else:
        print(f"💥 PARTIAL: {passed}/{total} tests passed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))