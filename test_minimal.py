#!/usr/bin/env python3
"""
Minimal test for Synthetic Data Guardian - no external dependencies
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_minimal_imports():
    """Test that we can import core modules."""
    print("🧪 Testing minimal imports...")
    
    try:
        # Test core module imports
        from synthetic_guardian.utils.logger import get_logger
        from synthetic_guardian.generators.base import BaseGenerator
        from synthetic_guardian.validators.base import BaseValidator
        from synthetic_guardian.watermarks.base import BaseWatermarker
        
        print("✅ Core base classes imported successfully")
        
        # Test logger functionality
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logger working")
        
        print("🎉 Minimal functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_class_creation():
    """Test creating basic instances without external dependencies."""
    print("\n🧪 Testing basic class instantiation...")
    
    try:
        from synthetic_guardian.generators.base import GeneratorConfig
        from synthetic_guardian.core.validation_report import ValidationReport, ValidationMetrics
        
        # Test config creation
        config = GeneratorConfig(
            name="test_generator",
            type="test",
            version="1.0.0"
        )
        print(f"✅ GeneratorConfig created: {config.name}")
        
        # Test validation report
        report = ValidationReport(task_id="test_task")
        metrics = ValidationMetrics()
        print(f"✅ ValidationReport created: {report.report_id}")
        print(f"✅ ValidationMetrics created: {metrics.total_validators}")
        
        print("🎉 Basic class creation test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Class creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner."""
    print("🔬 Synthetic Data Guardian - Minimal Functionality Test")
    print("=" * 60)
    print("Testing core functionality without external dependencies...")
    
    success = True
    
    # Test minimal imports
    if not test_minimal_imports():
        success = False
    
    # Test basic class creation
    if not test_basic_class_creation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎯 SUCCESS: All minimal tests passed!")
        print("The core Synthetic Data Guardian infrastructure is working!")
        return 0
    else:
        print("💥 FAILURE: Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)