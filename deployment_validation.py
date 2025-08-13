#!/usr/bin/env python3
"""
Deployment Readiness Validation Script
Validates the Synthetic Data Guardian system for production deployment
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_guardian import Guardian, PipelineBuilder


async def validate_production_readiness():
    """
    Comprehensive production readiness validation
    """
    print("🚀 Production Deployment Readiness Validation")
    print("=" * 60)
    
    validation_results = {
        "core_system": False,
        "data_generation": False, 
        "security": False,
        "performance": False,
        "monitoring": False,
        "configuration": False
    }
    
    try:
        # 1. Core System Validation
        print("1. Validating core system components...")
        guardian = Guardian()
        await guardian.initialize()
        print("   ✅ Core system initialization successful")
        validation_results["core_system"] = True
        
        # 2. Data Generation Validation  
        print("2. Validating data generation capabilities...")
        pipeline_config = {
            'name': 'production_test',
            'data_type': 'tabular',
            'generator_type': 'tabular',
            'schema': {
                'id': {'type': 'integer', 'range': [1, 1000]},
                'value': {'type': 'float', 'range': [0.0, 100.0]}
            }
        }
        
        result = await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=1000,
            seed=42
        )
        
        if result.data is not None and len(result.data) == 1000:
            print("   ✅ Data generation validation successful")
            validation_results["data_generation"] = True
        else:
            print("   ❌ Data generation validation failed")
            
        # 3. Security Validation
        print("3. Validating security features...")
        # Test watermarking
        watermark_result = await guardian.watermark(
            data=result.data,
            method='statistical',
            message='production_test'
        )
        
        if watermark_result:
            print("   ✅ Security validation successful")
            validation_results["security"] = True
        else:
            print("   ❌ Security validation failed")
            
        # 4. Performance Validation
        print("4. Validating performance benchmarks...")
        start_time = time.time()
        
        perf_result = await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=10000,
            seed=42
        )
        
        generation_time = time.time() - start_time
        records_per_second = 10000 / generation_time
        
        if records_per_second > 1000:  # Minimum 1K records/sec
            print(f"   ✅ Performance validation successful ({records_per_second:.0f} rec/sec)")
            validation_results["performance"] = True
        else:
            print(f"   ❌ Performance validation failed ({records_per_second:.0f} rec/sec)")
            
        # 5. Monitoring Validation
        print("5. Validating monitoring capabilities...")
        metrics = guardian.get_metrics()
        tasks = guardian.get_active_tasks()
        
        if metrics is not None and tasks is not None:
            print("   ✅ Monitoring validation successful")
            validation_results["monitoring"] = True
        else:
            print("   ❌ Monitoring validation failed")
            
        # 6. Configuration Validation
        print("6. Validating configuration management...")
        config_valid = True
        
        # Check required files exist
        required_files = [
            'pyproject.toml',
            'src/synthetic_guardian/__init__.py',
            'src/synthetic_guardian/core/guardian.py'
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                print(f"   ❌ Missing required file: {file_path}")
                config_valid = False
                
        if config_valid:
            print("   ✅ Configuration validation successful")
            validation_results["configuration"] = True
            
        await guardian.cleanup()
        
    except Exception as e:
        print(f"   ❌ Validation error: {str(e)}")
        
    # Final Report
    print("\n" + "=" * 60)
    print("🎯 PRODUCTION READINESS REPORT")
    print("=" * 60)
    
    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    
    for component, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component.replace('_', ' ').title()}: {status}")
        
    print(f"\nOverall Score: {passed_checks}/{total_checks} ({100*passed_checks/total_checks:.1f}%)")
    
    if passed_checks == total_checks:
        print("\n🎉 SYSTEM IS PRODUCTION READY! 🎉")
        return True
    else:
        print(f"\n⚠️  System requires attention before production deployment")
        return False


if __name__ == "__main__":
    result = asyncio.run(validate_production_readiness())
    sys.exit(0 if result else 1)