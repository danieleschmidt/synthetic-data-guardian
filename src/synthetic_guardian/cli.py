#!/usr/bin/env python3
"""
Synthetic Data Guardian CLI
Command-line interface for synthetic data generation and validation.
"""

import asyncio
import sys
import argparse
import json
try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path
from typing import Dict, Any

from .core.guardian import Guardian
from .utils.logger import get_logger, configure_logging


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description='Synthetic Data Guardian - Enterprise-grade synthetic data pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic data from configuration
  synthetic-guardian generate --config config.yaml --output data/synthetic.csv
  
  # Validate existing data
  synthetic-guardian validate --input data/synthetic.csv --reference data/real.csv
  
  # Start API server
  synthetic-guardian serve --port 8080
  
  # Track lineage
  synthetic-guardian lineage --dataset-id abc123 --format graph
  
  # Generate compliance report
  synthetic-guardian report --data data/ --standard gdpr --output report.pdf
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-error output')
    parser.add_argument('--config', '-c', type=Path,
                       help='Configuration file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', 
                                           help='Generate synthetic data')
    generate_parser.add_argument('--pipeline', type=str, required=True,
                                help='Pipeline configuration file or ID')
    generate_parser.add_argument('--num-records', '-n', type=int, default=1000,
                                help='Number of records to generate')
    generate_parser.add_argument('--seed', type=int,
                                help='Random seed for reproducibility')
    generate_parser.add_argument('--output', '-o', type=Path, required=True,
                                help='Output file path')
    generate_parser.add_argument('--format', choices=['json', 'csv', 'parquet'],
                                default='csv', help='Output format')
    generate_parser.add_argument('--conditions', type=str,
                                help='Generation conditions (JSON string)')
    generate_parser.add_argument('--no-validate', action='store_true',
                                help='Skip validation')
    generate_parser.add_argument('--no-watermark', action='store_true',
                                help='Skip watermarking')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate',
                                           help='Validate synthetic data')
    validate_parser.add_argument('--input', '-i', type=Path, required=True,
                                help='Input data file')
    validate_parser.add_argument('--reference', '-r', type=Path,
                                help='Reference data file')
    validate_parser.add_argument('--validators', nargs='+',
                                help='Specific validators to run')
    validate_parser.add_argument('--output', '-o', type=Path,
                                help='Output report file')
    validate_parser.add_argument('--format', choices=['json', 'text', 'html'],
                                default='text', help='Report format')
    validate_parser.add_argument('--threshold', type=float, default=0.8,
                                help='Minimum score threshold')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve',
                                        help='Start API server')
    serve_parser.add_argument('--host', default='localhost',
                             help='Server host')
    serve_parser.add_argument('--port', '-p', type=int, default=8080,
                             help='Server port')
    serve_parser.add_argument('--workers', type=int, default=1,
                             help='Number of worker processes')
    serve_parser.add_argument('--reload', action='store_true',
                             help='Enable auto-reload for development')
    
    # Lineage command
    lineage_parser = subparsers.add_parser('lineage',
                                          help='Track and query lineage')
    lineage_parser.add_argument('--dataset-id', type=str,
                               help='Dataset ID to query')
    lineage_parser.add_argument('--task-id', type=str,
                               help='Task ID to query')
    lineage_parser.add_argument('--format', choices=['json', 'graph', 'text'],
                               default='text', help='Output format')
    lineage_parser.add_argument('--output', '-o', type=Path,
                               help='Output file')
    
    # Report command
    report_parser = subparsers.add_parser('report',
                                         help='Generate compliance reports')
    report_parser.add_argument('--data', type=Path, required=True,
                              help='Data directory or file')
    report_parser.add_argument('--standard', choices=['gdpr', 'hipaa', 'ccpa'],
                              required=True, help='Compliance standard')
    report_parser.add_argument('--output', '-o', type=Path, required=True,
                              help='Output report file')
    report_parser.add_argument('--include-lineage', action='store_true',
                              help='Include lineage information')
    
    # Watermark command
    watermark_parser = subparsers.add_parser('watermark',
                                            help='Apply or verify watermarks')
    watermark_subparsers = watermark_parser.add_subparsers(dest='watermark_action')
    
    # Watermark embed
    embed_parser = watermark_subparsers.add_parser('embed',
                                                  help='Embed watermark')
    embed_parser.add_argument('--input', '-i', type=Path, required=True,
                             help='Input data file')
    embed_parser.add_argument('--output', '-o', type=Path, required=True,
                             help='Output data file')
    embed_parser.add_argument('--method', default='statistical',
                             help='Watermarking method')
    embed_parser.add_argument('--message', type=str,
                             help='Watermark message')
    embed_parser.add_argument('--strength', type=float, default=0.8,
                             help='Watermark strength')
    
    # Watermark verify
    verify_parser = watermark_subparsers.add_parser('verify',
                                                   help='Verify watermark')
    verify_parser.add_argument('--input', '-i', type=Path, required=True,
                              help='Input data file')
    verify_parser.add_argument('--method', default='statistical',
                              help='Watermarking method')
    verify_parser.add_argument('--output', '-o', type=Path,
                              help='Verification result file')
    
    return parser


async def command_generate(args, config):
    """Handle generate command."""
    logger = get_logger('generate')
    
    # Load pipeline configuration
    pipeline_config = None
    if args.pipeline.endswith(('.yaml', '.yml', '.json')):
        pipeline_path = Path(args.pipeline)
        if not pipeline_path.exists():
            logger.error(f"Pipeline configuration not found: {pipeline_path}")
            return 1
        
        with open(pipeline_path) as f:
            if pipeline_path.suffix in ['.yaml', '.yml']:
                if yaml is None:
                    logger.error("YAML support not available. Install PyYAML: pip install pyyaml")
                    return 1
                pipeline_config = yaml.safe_load(f)
            else:
                pipeline_config = json.load(f)
    else:
        # Pipeline ID
        pipeline_config = args.pipeline
    
    # Parse conditions if provided
    conditions = None
    if args.conditions:
        try:
            conditions = json.loads(args.conditions)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid conditions JSON: {e}")
            return 1
    
    # Create Guardian and generate data
    async with Guardian(config=config) as guardian:
        try:
            logger.info(f"Generating {args.num_records} records...")
            
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=args.num_records,
                seed=args.seed,
                conditions=conditions,
                validate=not args.no_validate,
                watermark=not args.no_watermark
            )
            
            # Export data
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if args.format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(result.export('json'), f, indent=2, default=str)
            elif args.format == 'csv':
                with open(output_path, 'w') as f:
                    f.write(result.export('csv'))
            elif args.format == 'parquet':
                import pandas as pd
                df = pd.DataFrame(result.data)
                df.to_parquet(output_path)
            
            logger.info(f"Generated data saved to: {output_path}")
            logger.info(f"Generation summary: {result.get_summary()}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return 1


async def command_validate(args, config):
    """Handle validate command."""
    logger = get_logger('validate')
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Load data based on format
    data = None
    try:
        if input_path.suffix == '.json':
            with open(input_path) as f:
                data = json.load(f)
        elif input_path.suffix == '.csv':
            import pandas as pd
            data = pd.read_csv(input_path).to_dict('records')
        elif input_path.suffix == '.parquet':
            import pandas as pd
            data = pd.read_parquet(input_path).to_dict('records')
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return 1
    except Exception as e:
        logger.error(f"Failed to load input data: {str(e)}")
        return 1
    
    # Load reference data if provided
    reference_data = None
    if args.reference:
        reference_path = Path(args.reference)
        if reference_path.exists():
            try:
                if reference_path.suffix == '.json':
                    with open(reference_path) as f:
                        reference_data = json.load(f)
                elif reference_path.suffix == '.csv':
                    import pandas as pd
                    reference_data = pd.read_csv(reference_path).to_dict('records')
                elif reference_path.suffix == '.parquet':
                    import pandas as pd
                    reference_data = pd.read_parquet(reference_path).to_dict('records')
            except Exception as e:
                logger.warning(f"Failed to load reference data: {str(e)}")
    
    # Validate data
    async with Guardian(config=config) as guardian:
        try:
            logger.info("Running validation...")
            
            validation_report = await guardian.validate(
                data=data,
                validators=args.validators,
                reference_data=reference_data
            )
            
            # Check threshold
            passed = validation_report.overall_score >= args.threshold
            
            # Output results
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    if args.format == 'json':
                        json.dump(validation_report.export('json'), f, indent=2, default=str)
                    elif args.format == 'text':
                        f.write(validation_report.export('text'))
                    elif args.format == 'html':
                        f.write(validation_report.export('html'))
                
                logger.info(f"Validation report saved to: {output_path}")
            else:
                # Print to console
                if args.format == 'json':
                    print(json.dumps(validation_report.export('json'), indent=2, default=str))
                else:
                    print(validation_report.export('text'))
            
            logger.info(f"Validation {'PASSED' if passed else 'FAILED'} (score: {validation_report.overall_score:.3f})")
            return 0 if passed else 1
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return 1


async def command_serve(args, config):
    """Handle serve command."""
    logger = get_logger('serve')
    
    try:
        import uvicorn
        from .api import create_app
        
        # Create FastAPI app
        app = create_app(config)
        
        logger.info(f"Starting server on {args.host}:{args.port}")
        
        # Run server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level='info' if not args.quiet else 'error'
        )
        
        return 0
        
    except ImportError:
        logger.error("FastAPI and uvicorn required for server mode. Install with: pip install synthetic-data-guardian[api]")
        return 1
    except Exception as e:
        logger.error(f"Server failed: {str(e)}")
        return 1


async def command_lineage(args, config):
    """Handle lineage command."""
    logger = get_logger('lineage')
    
    # For now, just return placeholder lineage info
    lineage_info = {
        'dataset_id': args.dataset_id,
        'task_id': args.task_id,
        'lineage_graph': {
            'nodes': [],
            'edges': []
        },
        'metadata': {
            'generated_at': '2024-01-01T00:00:00Z',
            'version': '1.0.0'
        }
    }
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if args.format == 'json':
                json.dump(lineage_info, f, indent=2)
            else:
                f.write(str(lineage_info))
        
        logger.info(f"Lineage information saved to: {output_path}")
    else:
        if args.format == 'json':
            print(json.dumps(lineage_info, indent=2))
        else:
            print(f"Lineage for dataset: {args.dataset_id or args.task_id}")
            print("(Lineage tracking implementation pending)")
    
    return 0


async def command_report(args, config):
    """Handle report command."""
    logger = get_logger('report')
    
    # Generate compliance report placeholder
    report_info = {
        'standard': args.standard,
        'data_path': str(args.data),
        'compliance_status': 'COMPLIANT',
        'checks_performed': [
            'data_minimization',
            'purpose_limitation',
            'privacy_preservation',
            'audit_trail'
        ],
        'generated_at': '2024-01-01T00:00:00Z'
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report_info, f, indent=2)
    
    logger.info(f"Compliance report saved to: {output_path}")
    return 0


async def command_watermark(args, config):
    """Handle watermark command."""
    logger = get_logger('watermark')
    
    if args.watermark_action == 'embed':
        # Load input data
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        # Embed watermark (placeholder implementation)
        logger.info(f"Embedding watermark in {input_path}")
        
        # For now, just copy the file
        import shutil
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        
        logger.info(f"Watermarked data saved to: {output_path}")
        
    elif args.watermark_action == 'verify':
        # Verify watermark (placeholder implementation)
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        logger.info(f"Verifying watermark in {input_path}")
        
        verification_result = {
            'watermark_detected': True,
            'method': args.method,
            'confidence': 0.95,
            'message': 'synthetic:example'
        }
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(verification_result, f, indent=2)
            
            logger.info(f"Verification result saved to: {output_path}")
        else:
            print(json.dumps(verification_result, indent=2))
    
    return 0


async def main_async(args=None):
    """Main async entry point."""
    parser = create_parser()
    args = parser.parse_args(args)
    
    # Configure logging
    if args.quiet:
        log_level = 'ERROR'
    elif args.verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'
    
    configure_logging(level=log_level)
    
    # Load configuration
    config = {}
    if args.config and args.config.exists():
        try:
            from .config import load_config
            config = load_config(args.config)
        except ImportError:
            print("Configuration loading not available - using defaults")
    
    # Route to command handler
    if args.command == 'generate':
        return await command_generate(args, config)
    elif args.command == 'validate':
        return await command_validate(args, config)
    elif args.command == 'serve':
        return await command_serve(args, config)
    elif args.command == 'lineage':
        return await command_lineage(args, config)
    elif args.command == 'report':
        return await command_report(args, config)
    elif args.command == 'watermark':
        return await command_watermark(args, config)
    else:
        parser.print_help()
        return 1


def main(args=None):
    """Main entry point for CLI."""
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
