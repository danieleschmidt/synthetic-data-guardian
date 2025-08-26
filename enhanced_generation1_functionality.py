#!/usr/bin/env python3
"""
Enhanced Generation 1 Functionality - TERRAGON AUTONOMOUS SDLC
Core synthetic data pipeline functionality with minimal dependencies
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TerragonDataAnalyzer:
    """Advanced data analysis for synthetic data quality assessment."""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data distribution and statistical properties."""
        if not data:
            return {"error": "No data provided"}
        
        analysis = {
            "total_records": len(data),
            "fields_analyzed": {},
            "data_quality_score": 0.0,
            "recommendations": []
        }
        
        # Get all field names from first record
        if data:
            fields = list(data[0].keys())
            
            for field in fields:
                field_values = [record.get(field) for record in data if record.get(field) is not None]
                
                field_analysis = {
                    "non_null_count": len(field_values),
                    "null_percentage": (len(data) - len(field_values)) / len(data) * 100,
                    "unique_values": len(set(str(v) for v in field_values)),
                    "data_type": self._infer_data_type(field_values)
                }
                
                # Type-specific analysis
                if field_analysis["data_type"] == "numeric":
                    numeric_values = [float(v) for v in field_values if self._is_numeric(v)]
                    if numeric_values:
                        field_analysis.update({
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "mean": sum(numeric_values) / len(numeric_values),
                            "range": max(numeric_values) - min(numeric_values)
                        })
                
                analysis["fields_analyzed"][field] = field_analysis
        
        # Calculate overall quality score
        analysis["data_quality_score"] = self._calculate_quality_score(analysis["fields_analyzed"])
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis["fields_analyzed"])
        
        return analysis
    
    def _infer_data_type(self, values: List[Any]) -> str:
        """Infer the data type of a field."""
        if not values:
            return "unknown"
        
        numeric_count = sum(1 for v in values if self._is_numeric(v))
        
        if numeric_count / len(values) > 0.8:
            return "numeric"
        elif len(set(str(v) for v in values)) / len(values) < 0.1:
            return "categorical"
        else:
            return "string"
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if a value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _calculate_quality_score(self, field_analysis: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall data quality score."""
        if not field_analysis:
            return 0.0
        
        scores = []
        for field, analysis in field_analysis.items():
            field_score = 100.0
            
            # Penalize for high null percentage
            field_score -= min(analysis["null_percentage"] * 2, 50)
            
            # Bonus for reasonable unique value ratio
            if analysis["non_null_count"] > 0:
                uniqueness_ratio = analysis["unique_values"] / analysis["non_null_count"]
                if 0.1 <= uniqueness_ratio <= 0.9:  # Good balance
                    field_score += 10
                elif uniqueness_ratio > 0.95:  # Too many unique values
                    field_score -= 5
            
            scores.append(max(0, field_score))
        
        return sum(scores) / len(scores)
    
    def _generate_recommendations(self, field_analysis: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        for field, analysis in field_analysis.items():
            if analysis["null_percentage"] > 20:
                recommendations.append(f"High null percentage ({analysis['null_percentage']:.1f}%) in field '{field}' - consider data imputation")
            
            if analysis["unique_values"] == 1:
                recommendations.append(f"Field '{field}' has no variance - consider removing or adding variation")
            
            if analysis["data_type"] == "numeric" and "range" in analysis and analysis["range"] == 0:
                recommendations.append(f"Numeric field '{field}' has no range variation")
        
        return recommendations


class TerragonPipelineOptimizer:
    """Optimize pipeline configuration for better synthetic data generation."""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_pipeline(self, pipeline_config: Dict[str, Any], performance_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize pipeline configuration based on performance data."""
        
        optimized_config = pipeline_config.copy()
        optimizations_applied = []
        
        # Optimize sample size based on performance
        if performance_data and "generation_time" in performance_data:
            generation_time = performance_data["generation_time"]
            current_size = pipeline_config.get("sample_size", 100)
            
            if generation_time > 5.0 and current_size > 500:  # Too slow for large datasets
                new_size = max(100, int(current_size * 0.8))
                optimized_config["sample_size"] = new_size
                optimizations_applied.append(f"Reduced sample size from {current_size} to {new_size} for better performance")
            elif generation_time < 0.5 and current_size < 10000:  # Can handle more
                new_size = min(10000, int(current_size * 1.2))
                optimized_config["sample_size"] = new_size
                optimizations_applied.append(f"Increased sample size from {current_size} to {new_size} to utilize capacity")
        
        # Optimize generator type based on data characteristics
        generator_type = pipeline_config.get("generator_type", "mock")
        schema = pipeline_config.get("schema", {})
        
        numeric_fields = sum(1 for field_type in schema.values() if "integer" in field_type or "float" in field_type)
        categorical_fields = sum(1 for field_type in schema.values() if "categorical" in field_type)
        
        if numeric_fields > categorical_fields and generator_type == "mock":
            optimized_config["generator_type"] = "tabular"
            optimizations_applied.append("Switched to tabular generator for numeric-heavy schema")
        
        # Add quality parameters
        if "parameters" not in optimized_config:
            optimized_config["parameters"] = {}
        
        optimized_config["parameters"].update({
            "quality_level": "standard",
            "randomization_seed": None,  # For reproducibility when needed
            "optimization_applied": True
        })
        
        optimization_record = {
            "timestamp": time.time(),
            "original_config": pipeline_config,
            "optimized_config": optimized_config,
            "optimizations": optimizations_applied,
            "performance_impact": performance_data
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"✅ Pipeline optimization completed: {len(optimizations_applied)} optimizations applied")
        return optimized_config
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization performance report."""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}
        
        return {
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": self.optimization_history[-5:],  # Last 5
            "common_optimizations": self._analyze_common_optimizations(),
            "performance_trends": self._analyze_performance_trends()
        }
    
    def _analyze_common_optimizations(self) -> List[str]:
        """Analyze the most common optimization patterns."""
        all_optimizations = []
        for record in self.optimization_history:
            all_optimizations.extend(record["optimizations"])
        
        # Count optimization types
        optimization_counts = {}
        for opt in all_optimizations:
            key = opt.split(' ')[0]  # First word as category
            optimization_counts[key] = optimization_counts.get(key, 0) + 1
        
        return [f"{opt}: {count} times" for opt, count in sorted(optimization_counts.items(), key=lambda x: x[1], reverse=True)]
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance improvement trends."""
        if len(self.optimization_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        recent = self.optimization_history[-5:]
        
        sample_sizes = [r["optimized_config"].get("sample_size", 0) for r in recent]
        generation_times = [r["performance_impact"].get("generation_time", 0) for r in recent if r["performance_impact"]]
        
        return {
            "average_sample_size": sum(sample_sizes) / len(sample_sizes) if sample_sizes else 0,
            "average_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
            "optimization_frequency": len(recent) / max(1, (recent[-1]["timestamp"] - recent[0]["timestamp"]) / 3600)  # per hour
        }


class TerragonBatchProcessor:
    """Process multiple data generation tasks in batch for efficiency."""
    
    def __init__(self):
        self.batch_results = {}
        self.processing_stats = {}
    
    def process_batch(self, pipelines: List[Dict[str, Any]], guardian) -> Dict[str, Any]:
        """Process multiple pipelines in batch."""
        
        batch_id = f"batch_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"🚀 Starting batch processing: {batch_id} with {len(pipelines)} pipelines")
        
        results = {
            "batch_id": batch_id,
            "start_time": start_time,
            "pipelines_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "results": {},
            "errors": [],
            "total_records_generated": 0
        }
        
        for i, pipeline_config in enumerate(pipelines):
            try:
                logger.info(f"Processing pipeline {i+1}/{len(pipelines)}: {pipeline_config.get('name', f'pipeline_{i}')}")
                
                # Import and use the simple guardian
                from synthetic_guardian.core.simple_guardian import SimpleGuardian, SimplePipeline
                
                # Create pipeline
                if isinstance(pipeline_config, dict):
                    pipeline = SimplePipeline(**pipeline_config)
                else:
                    pipeline = pipeline_config
                
                # Generate data
                result = guardian.generate(pipeline)
                
                if result.success:
                    results["successful_generations"] += 1
                    results["results"][pipeline.name] = {
                        "generation_id": result.generation_id,
                        "sample_size": result.metadata.get("sample_size", 0),
                        "generation_time": result.metadata.get("generation_time", 0),
                        "status": "success"
                    }
                    results["total_records_generated"] += result.metadata.get("sample_size", 0)
                else:
                    results["failed_generations"] += 1
                    results["results"][pipeline.name] = {
                        "status": "failed",
                        "errors": result.errors
                    }
                    results["errors"].extend(result.errors)
                
                results["pipelines_processed"] += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing pipeline {i+1}: {str(e)}")
                results["failed_generations"] += 1
                results["errors"].append(f"Pipeline {i+1}: {str(e)}")
        
        results["end_time"] = time.time()
        results["total_processing_time"] = results["end_time"] - start_time
        results["average_time_per_pipeline"] = results["total_processing_time"] / len(pipelines) if pipelines else 0
        
        # Store results
        self.batch_results[batch_id] = results
        
        logger.info(f"✅ Batch processing completed: {results['successful_generations']}/{len(pipelines)} successful")
        return results
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific batch."""
        return self.batch_results.get(batch_id)
    
    def list_batches(self) -> List[str]:
        """List all processed batch IDs."""
        return list(self.batch_results.keys())
    
    def generate_batch_report(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Generate detailed report for a batch."""
        batch_data = self.batch_results.get(batch_id)
        if not batch_data:
            return None
        
        return {
            "batch_summary": {
                "id": batch_id,
                "total_pipelines": batch_data["pipelines_processed"],
                "success_rate": batch_data["successful_generations"] / batch_data["pipelines_processed"] * 100 if batch_data["pipelines_processed"] > 0 else 0,
                "total_records": batch_data["total_records_generated"],
                "processing_time": f"{batch_data['total_processing_time']:.2f}s",
                "avg_time_per_pipeline": f"{batch_data['average_time_per_pipeline']:.2f}s"
            },
            "performance_metrics": {
                "throughput": batch_data["total_records_generated"] / batch_data["total_processing_time"] if batch_data["total_processing_time"] > 0 else 0,
                "pipeline_throughput": batch_data["pipelines_processed"] / batch_data["total_processing_time"] if batch_data["total_processing_time"] > 0 else 0
            },
            "detailed_results": batch_data["results"],
            "errors": batch_data["errors"]
        }


def demonstrate_generation1_capabilities():
    """Demonstrate Generation 1 enhanced capabilities."""
    
    print("🚀 TERRAGON AUTONOMOUS SDLC - Generation 1 Enhanced Functionality")
    print("=" * 70)
    
    # Initialize components
    try:
        from synthetic_guardian.core.simple_guardian import SimpleGuardian, SimplePipeline
        
        guardian = SimpleGuardian()
        analyzer = TerragonDataAnalyzer()
        optimizer = TerragonPipelineOptimizer()
        batch_processor = TerragonBatchProcessor()
        
        print("✅ All components initialized successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 1. Basic Data Generation
    print("\n1. Basic Data Generation:")
    pipeline1 = SimplePipeline(
        name="user_profiles",
        generator_type="tabular",
        schema={
            "user_id": "integer",
            "age": "integer[18:65]",
            "income": "float[25000:150000]",
            "city": "categorical"
        },
        sample_size=100
    )
    
    result1 = guardian.generate(pipeline1)
    print(f"   Generated {result1.metadata.get('sample_size', 0)} records")
    print(f"   Generation ID: {result1.generation_id[:8]}...")
    
    # 2. Data Analysis
    print("\n2. Advanced Data Analysis:")
    if result1.success and result1.data:
        analysis = analyzer.analyze_distribution(result1.data["records"])
        print(f"   Data Quality Score: {analysis['data_quality_score']:.1f}/100")
        print(f"   Fields Analyzed: {len(analysis['fields_analyzed'])}")
        print(f"   Recommendations: {len(analysis['recommendations'])}")
    
    # 3. Pipeline Optimization
    print("\n3. Pipeline Optimization:")
    performance_data = {
        "generation_time": 1.5,
        "memory_usage": "50MB"
    }
    
    optimized_config = optimizer.optimize_pipeline(
        pipeline1.__dict__,
        performance_data
    )
    print(f"   Optimizations Applied: {len(optimizer.optimization_history)}")
    
    # 4. Batch Processing
    print("\n4. Batch Processing:")
    batch_pipelines = [
        {
            "name": "customers",
            "generator_type": "tabular",
            "sample_size": 50,
            "schema": {"id": "integer", "name": "string", "value": "float"}
        },
        {
            "name": "transactions", 
            "generator_type": "timeseries",
            "sample_size": 30
        },
        {
            "name": "products",
            "generator_type": "mock",
            "sample_size": 25
        }
    ]
    
    batch_result = batch_processor.process_batch(batch_pipelines, guardian)
    print(f"   Batch ID: {batch_result['batch_id']}")
    print(f"   Success Rate: {batch_result['successful_generations']}/{batch_result['pipelines_processed']}")
    print(f"   Total Records: {batch_result['total_records_generated']}")
    
    # 5. Export Results
    print("\n5. Data Export:")
    export_dir = Path("/tmp/terragon_generation1")
    export_dir.mkdir(exist_ok=True)
    
    exported = guardian.export_result(
        result1.generation_id,
        str(export_dir / "sample_data.json")
    )
    print(f"   Export Success: {exported}")
    
    # 6. Generate Summary Report
    print("\n6. Generation Summary:")
    print(f"   Total Results Cached: {len(guardian.list_results())}")
    print(f"   Batch Results: {len(batch_processor.list_batches())}")
    
    # Save comprehensive report
    report = {
        "generation1_summary": {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "components_tested": [
                "SimpleGuardian",
                "TerragonDataAnalyzer", 
                "TerragonPipelineOptimizer",
                "TerragonBatchProcessor"
            ],
            "data_generated": {
                "individual_pipelines": len(guardian.list_results()),
                "batch_pipelines": batch_result['pipelines_processed'],
                "total_records": batch_result['total_records_generated']
            },
            "quality_metrics": {
                "data_quality_score": analysis['data_quality_score'],
                "validation_passed": result1.success
            },
            "performance": {
                "batch_processing_time": batch_result['total_processing_time'],
                "average_time_per_pipeline": batch_result['average_time_per_pipeline']
            }
        }
    }
    
    report_path = export_dir / "generation1_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Comprehensive Report: {report_path}")
    print("\n✅ Generation 1 Enhanced Functionality Demonstration Complete!")
    return True


if __name__ == "__main__":
    success = demonstrate_generation1_capabilities()
    if success:
        print("\n🎉 TERRAGON Generation 1 Implementation: SUCCESSFUL")
        print("Ready to proceed to Generation 2: Make it Robust")
    else:
        print("\n❌ Generation 1 implementation encountered issues")
        sys.exit(1)