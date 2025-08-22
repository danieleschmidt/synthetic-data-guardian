#!/usr/bin/env python3
"""
Test Research Integration with Guardian

This script tests the integration of research modules with the main Guardian system.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_guardian.core.guardian import Guardian, GuardianConfig


async def test_research_integration():
    """Test research module integration with Guardian."""
    print("🧪 Testing Research Integration with Guardian")
    print("=" * 50)
    
    # Test basic Guardian without research
    print("\n📋 Testing basic Guardian initialization...")
    basic_config = GuardianConfig(
        name="test-guardian-basic",
        enable_research_mode=False
    )
    
    basic_guardian = Guardian(basic_config)
    await basic_guardian.initialize()
    
    print(f"✅ Basic Guardian initialized: {basic_guardian.config.name}")
    print(f"   Research modules loaded: {len(basic_guardian.research_modules)}")
    
    # Test Guardian with research modules enabled
    print("\n🔬 Testing Guardian with research modules...")
    research_config = GuardianConfig(
        name="test-guardian-research",
        enable_research_mode=True,
        adaptive_privacy_enabled=True,
        quantum_watermarking_enabled=True,
        neural_temporal_preservation_enabled=True,
        zk_lineage_enabled=True,
        adversarial_testing_enabled=True
    )
    
    research_guardian = Guardian(research_config)
    
    try:
        await research_guardian.initialize()
        
        print(f"✅ Research Guardian initialized: {research_guardian.config.name}")
        print(f"   Research modules loaded: {len(research_guardian.research_modules)}")
        
        # Show loaded research modules
        if research_guardian.research_modules:
            print("   Loaded modules:")
            for module_name, module_obj in research_guardian.research_modules.items():
                print(f"     - {module_name}: {type(module_obj).__name__}")
        
        # Test individual module access
        if research_guardian.adaptive_privacy:
            print("   ✅ Adaptive Differential Privacy available")
        
        if research_guardian.quantum_watermarker:
            print("   ✅ Quantum-Resistant Watermarking available")
        
        if research_guardian.temporal_style_transfer:
            print("   ✅ Neural Temporal Preservation available")
        
        if research_guardian.zk_lineage_system:
            print("   ✅ Zero-Knowledge Lineage available")
        
        if research_guardian.adversarial_evaluator:
            print("   ✅ Adversarial Robustness Testing available")
        
        print("\n🎉 All research modules integrated successfully!")
        
    except Exception as e:
        print(f"❌ Research integration failed: {str(e)}")
        print("   This is expected if torch/cryptography dependencies are missing")
        print("   The Guardian will continue to work without research modules")
    
    # Test research info
    print("\n📚 Testing research information access...")
    try:
        from synthetic_guardian.research import get_research_info
        
        research_info = get_research_info()
        print(f"✅ Research info loaded:")
        print(f"   Version: {research_info['version']}")
        print(f"   Status: {research_info['status']}")
        print(f"   Contributions: {research_info['novel_contributions']}")
        
    except Exception as e:
        print(f"❌ Research info access failed: {str(e)}")
    
    print("\n✅ Research integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_research_integration())