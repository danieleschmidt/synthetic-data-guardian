#!/usr/bin/env python3
"""
Autonomous SDLC Execution Script
Execute the complete autonomous software development lifecycle with progressive quality gates.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.sdlc.orchestrator import (
    SDLCOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMode
)
from src.sdlc.deployment_engine import DeploymentEnvironment
from src.utils.logger import get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute Autonomous SDLC with Progressive Quality Gates"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["autonomous", "supervised", "manual"],
        default="autonomous",
        help="Execution mode (default: autonomous)"
    )
    
    parser.add_argument(
        "--target-env",
        choices=["development", "staging", "production", "test"],
        default="staging",
        help="Target deployment environment (default: staging)"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=85.0,
        help="Quality gate threshold percentage (default: 85.0)"
    )
    
    parser.add_argument(
        "--security-threshold",
        type=float,
        default=90.0,
        help="Security validation threshold percentage (default: 90.0)"
    )
    
    parser.add_argument(
        "--performance-threshold",
        type=float,
        default=80.0,
        help="Performance optimization threshold percentage (default: 80.0)"
    )
    
    parser.add_argument(
        "--disable-progressive",
        action="store_true",
        help="Disable progressive enhancement"
    )
    
    parser.add_argument(
        "--disable-quality-gates",
        action="store_true",
        help="Disable quality gates validation"
    )
    
    parser.add_argument(
        "--disable-security",
        action="store_true",
        help="Disable security validation"
    )
    
    parser.add_argument(
        "--disable-performance",
        action="store_true",
        help="Disable performance optimization"
    )
    
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Disable monitoring setup"
    )
    
    parser.add_argument(
        "--disable-deployment",
        action="store_true",
        help="Disable deployment preparation"
    )
    
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Output file for comprehensive report (JSON format)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def create_orchestration_config(args) -> OrchestrationConfig:
    """Create orchestration configuration from arguments."""
    # Map string values to enums
    mode_mapping = {
        "autonomous": OrchestrationMode.AUTONOMOUS,
        "supervised": OrchestrationMode.SUPERVISED,
        "manual": OrchestrationMode.MANUAL
    }
    
    env_mapping = {
        "development": DeploymentEnvironment.DEVELOPMENT,
        "staging": DeploymentEnvironment.STAGING,
        "production": DeploymentEnvironment.PRODUCTION,
        "test": DeploymentEnvironment.TEST
    }
    
    return OrchestrationConfig(
        project_root=args.project_root.resolve(),
        mode=mode_mapping[args.mode],
        enable_progressive_enhancement=not args.disable_progressive,
        enable_quality_gates=not args.disable_quality_gates,
        enable_security_validation=not args.disable_security,
        enable_performance_optimization=not args.disable_performance,
        enable_monitoring=not args.disable_monitoring,
        enable_deployment=not args.disable_deployment,
        target_environment=env_mapping[args.target_env],
        quality_threshold=args.quality_threshold,
        security_threshold=args.security_threshold,
        performance_threshold=args.performance_threshold
    )


def print_banner():
    """Print execution banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                 TERRAGON AUTONOMOUS SDLC EXECUTOR                ║
║              Progressive Quality Gates & Execution               ║
╠══════════════════════════════════════════════════════════════════╣
║  🧠 Intelligent Analysis    📈 Progressive Enhancement          ║
║  🔍 Quality Gates          🛡️  Security Validation             ║
║  ⚡ Performance Optimization 📊 Monitoring Setup               ║
║  🚀 Deployment Preparation  🎯 Autonomous Execution            ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_execution_summary(result, duration: float):
    """Print execution summary."""
    status_icon = "✅" if result.success else "⚠️"
    status_text = "SUCCESS" if result.success else "PARTIAL SUCCESS"
    
    print(f"\n{status_icon} SDLC EXECUTION {status_text}")
    print("=" * 60)
    print(f"📊 Overall Score:      {result.overall_score:.1f}%")
    print(f"⏱️  Total Duration:     {duration:.2f} seconds")
    print(f"🏗️  Phases Completed:   {len(result.phases_completed)}")
    print(f"📦 Artifacts Generated: {len(result.artifacts)}")
    print(f"💡 Recommendations:    {len(result.recommendations)}")
    
    print("\n🏗️ PHASES COMPLETED:")
    for phase in result.phases_completed:
        print(f"   ✓ {phase.replace('_', ' ').title()}")
    
    if result.recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in result.recommendations[:5]:  # Show first 5
            print(f"   • {rec}")
        if len(result.recommendations) > 5:
            print(f"   ... and {len(result.recommendations) - 5} more")
    
    print("\n" + "=" * 60)


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("autonomous_sdlc", level=log_level)
    
    # Print banner
    print_banner()
    
    # Create configuration
    config = create_orchestration_config(args)
    
    print(f"🎯 Project Root: {config.project_root}")
    print(f"🔧 Execution Mode: {config.mode.value}")
    print(f"🌍 Target Environment: {config.target_environment.value}")
    print(f"📊 Quality Threshold: {config.quality_threshold}%")
    print(f"🛡️ Security Threshold: {config.security_threshold}%")
    print(f"⚡ Performance Threshold: {config.performance_threshold}%")
    print()
    
    try:
        # Initialize orchestrator
        orchestrator = SDLCOrchestrator(config, logger=logger)
        
        # Execute autonomous SDLC
        logger.info("Starting autonomous SDLC execution...")
        start_time = asyncio.get_event_loop().time()
        
        result = await orchestrator.execute_full_sdlc()
        
        end_time = asyncio.get_event_loop().time()
        total_duration = end_time - start_time
        
        # Print summary
        print_execution_summary(result, total_duration)
        
        # Generate comprehensive report if requested
        if args.output_report:
            logger.info(f"Generating comprehensive report: {args.output_report}")
            
            comprehensive_report = await orchestrator.generate_comprehensive_report()
            
            # Add execution metadata
            comprehensive_report['execution_metadata'] = {
                'command_line_args': vars(args),
                'config': {
                    'project_root': str(config.project_root),
                    'mode': config.mode.value,
                    'target_environment': config.target_environment.value,
                    'quality_threshold': config.quality_threshold,
                    'security_threshold': config.security_threshold,
                    'performance_threshold': config.performance_threshold
                },
                'total_execution_time': total_duration
            }
            
            # Write report to file
            args.output_report.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_report, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            print(f"📋 Comprehensive report saved to: {args.output_report}")
        
        # Cleanup
        await orchestrator.cleanup()
        
        # Exit with appropriate code
        exit_code = 0 if result.success else 1
        
        if exit_code == 0:
            print(f"\n🎉 Autonomous SDLC execution completed successfully!")
        else:
            print(f"\n⚠️ Autonomous SDLC execution completed with issues.")
            print("   Review recommendations and re-run after addressing issues.")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️ Execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        print(f"\n❌ Execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
        sys.exit(130)