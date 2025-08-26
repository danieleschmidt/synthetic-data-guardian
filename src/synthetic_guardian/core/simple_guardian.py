"""
Simple Guardian Implementation - Generation 1: Make it Work
Core functionality for synthetic data generation with minimal dependencies.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleGenerationResult:
    """Simple result container for synthetic data generation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    generation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))


@dataclass
class SimplePipeline:
    """Simple pipeline configuration for data generation."""
    name: str
    generator_type: str = "mock"
    parameters: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, str] = field(default_factory=dict)
    sample_size: int = 100


class SimpleDataGenerator:
    """Simple data generator with mock capabilities."""
    
    def __init__(self):
        self.generators = {
            "mock": self._generate_mock_data,
            "tabular": self._generate_tabular_data,
            "timeseries": self._generate_timeseries_data,
        }
    
    def generate(self, pipeline: SimplePipeline) -> SimpleGenerationResult:
        """Generate synthetic data based on pipeline configuration."""
        try:
            logger.info(f"Starting generation with pipeline: {pipeline.name}")
            
            generator_func = self.generators.get(pipeline.generator_type)
            if not generator_func:
                return SimpleGenerationResult(
                    success=False,
                    errors=[f"Unknown generator type: {pipeline.generator_type}"]
                )
            
            # Generate the data
            data = generator_func(pipeline)
            
            # Create result
            result = SimpleGenerationResult(
                success=True,
                data=data,
                metadata={
                    "pipeline_name": pipeline.name,
                    "generator_type": pipeline.generator_type,
                    "sample_size": len(data.get("records", [])) if isinstance(data, dict) else 0,
                    "generation_time": time.time(),
                }
            )
            
            logger.info(f"✅ Generation completed: {result.generation_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {str(e)}")
            return SimpleGenerationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _generate_mock_data(self, pipeline: SimplePipeline) -> Dict[str, Any]:
        """Generate simple mock data."""
        import random
        
        records = []
        for i in range(pipeline.sample_size):
            record = {
                "id": i + 1,
                "name": f"User_{i+1:04d}",
                "value": round(random.uniform(0, 1000), 2),
                "category": random.choice(["A", "B", "C", "D"]),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', 
                                       time.localtime(time.time() - random.randint(0, 86400)))
            }
            records.append(record)
        
        return {
            "records": records,
            "schema": {
                "id": "integer",
                "name": "string", 
                "value": "float",
                "category": "categorical",
                "timestamp": "datetime"
            }
        }
    
    def _generate_tabular_data(self, pipeline: SimplePipeline) -> Dict[str, Any]:
        """Generate tabular synthetic data."""
        import random
        
        # Use schema if provided, otherwise default schema
        schema = pipeline.schema or {
            "user_id": "integer",
            "age": "integer[18:80]",
            "income": "float[20000:150000]",
            "city": "categorical"
        }
        
        records = []
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        
        for i in range(pipeline.sample_size):
            record = {}
            for field, field_type in schema.items():
                if field_type == "integer" or field_type.startswith("integer["):
                    if "[" in field_type:
                        # Parse range like "integer[18:80]"
                        range_part = field_type.split("[")[1].rstrip("]")
                        min_val, max_val = map(int, range_part.split(":"))
                        record[field] = random.randint(min_val, max_val)
                    else:
                        record[field] = i + 1
                elif field_type == "float" or field_type.startswith("float["):
                    if "[" in field_type:
                        range_part = field_type.split("[")[1].rstrip("]")
                        min_val, max_val = map(float, range_part.split(":"))
                        record[field] = round(random.uniform(min_val, max_val), 2)
                    else:
                        record[field] = round(random.uniform(0, 1000), 2)
                elif field_type == "categorical":
                    if field == "city":
                        record[field] = random.choice(cities)
                    else:
                        record[field] = random.choice(["A", "B", "C"])
                else:
                    record[field] = f"{field}_{i+1}"
            
            records.append(record)
        
        return {
            "records": records,
            "schema": schema
        }
    
    def _generate_timeseries_data(self, pipeline: SimplePipeline) -> Dict[str, Any]:
        """Generate simple time series data."""
        import random
        import math
        
        records = []
        base_time = time.time() - (pipeline.sample_size * 3600)  # Start N hours ago
        
        for i in range(pipeline.sample_size):
            timestamp = base_time + (i * 3600)  # Hourly data points
            
            # Create synthetic time series with trend and noise
            trend = i * 0.1
            seasonal = 10 * math.sin(2 * math.pi * i / 24)  # Daily pattern
            noise = random.gauss(0, 2)
            value = 100 + trend + seasonal + noise
            
            record = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                "value": round(value, 2),
                "series_id": "ts_001"
            }
            records.append(record)
        
        return {
            "records": records,
            "schema": {
                "timestamp": "datetime",
                "value": "float", 
                "series_id": "string"
            }
        }


class SimpleGuardian:
    """Simple Guardian - Generation 1 implementation with core functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = SimpleDataGenerator()
        self.results_cache = {}
        
        logger.info("✅ SimpleGuardian initialized")
    
    def generate(self, pipeline: Union[SimplePipeline, Dict[str, Any]]) -> SimpleGenerationResult:
        """Generate synthetic data using the specified pipeline."""
        
        # Convert dict to SimplePipeline if needed
        if isinstance(pipeline, dict):
            pipeline = SimplePipeline(**pipeline)
        
        # Generate the data
        result = self.generator.generate(pipeline)
        
        # Cache the result
        self.results_cache[result.generation_id] = result
        
        return result
    
    def get_result(self, generation_id: str) -> Optional[SimpleGenerationResult]:
        """Retrieve a cached generation result."""
        return self.results_cache.get(generation_id)
    
    def list_results(self) -> List[str]:
        """List all cached generation result IDs."""
        return list(self.results_cache.keys())
    
    def export_result(self, generation_id: str, file_path: str) -> bool:
        """Export a generation result to file."""
        try:
            result = self.results_cache.get(generation_id)
            if not result:
                logger.error(f"No result found for ID: {generation_id}")
                return False
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on file extension
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump({
                        "generation_id": result.generation_id,
                        "success": result.success,
                        "data": result.data,
                        "metadata": result.metadata,
                        "timestamp": result.timestamp
                    }, f, indent=2)
            else:
                # Default to JSON
                file_path = file_path + '.json' if not file_path.endswith('.json') else file_path
                with open(file_path, 'w') as f:
                    json.dump(result.data, f, indent=2)
            
            logger.info(f"✅ Result exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Export failed: {str(e)}")
            return False
    
    def validate_simple(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform simple validation on generated data."""
        try:
            validation_report = {
                "valid": True,
                "checks": [],
                "warnings": [],
                "errors": []
            }
            
            # Basic structure validation
            if "records" not in data:
                validation_report["errors"].append("Missing 'records' field")
                validation_report["valid"] = False
            else:
                records = data["records"]
                validation_report["checks"].append(f"Found {len(records)} records")
                
                # Check for empty records
                if len(records) == 0:
                    validation_report["warnings"].append("No records found")
                
                # Sample record validation
                if records:
                    first_record = records[0]
                    validation_report["checks"].append(f"Record structure: {list(first_record.keys())}")
                    
                    # Check for null values
                    null_fields = [k for k, v in first_record.items() if v is None]
                    if null_fields:
                        validation_report["warnings"].append(f"Null values in fields: {null_fields}")
            
            # Schema validation
            if "schema" in data:
                schema = data["schema"]
                validation_report["checks"].append(f"Schema defined for {len(schema)} fields")
            else:
                validation_report["warnings"].append("No schema provided")
            
            logger.info(f"✅ Validation completed: {'PASS' if validation_report['valid'] else 'FAIL'}")
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return {
                "valid": False,
                "errors": [str(e)],
                "checks": [],
                "warnings": []
            }


def create_sample_pipeline(name: str = "sample_pipeline") -> SimplePipeline:
    """Create a sample pipeline for testing."""
    return SimplePipeline(
        name=name,
        generator_type="tabular",
        parameters={"quality": "high"},
        schema={
            "user_id": "integer",
            "age": "integer[18:65]", 
            "income": "float[25000:200000]",
            "city": "categorical"
        },
        sample_size=50
    )


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Simple Guardian - Generation 1")
    
    # Create guardian
    guardian = SimpleGuardian()
    
    # Create and run sample pipeline
    pipeline = create_sample_pipeline("test_generation_1")
    result = guardian.generate(pipeline)
    
    print(f"Generation Result: {result.success}")
    print(f"Generation ID: {result.generation_id}")
    print(f"Sample Size: {result.metadata.get('sample_size', 0)}")
    
    if result.success and result.data:
        # Validate the data
        validation = guardian.validate_simple(result.data)
        print(f"Validation Status: {'PASS' if validation['valid'] else 'FAIL'}")
        
        # Export result
        export_success = guardian.export_result(result.generation_id, f"/tmp/generation_1_{result.generation_id}.json")
        print(f"Export Success: {export_success}")
    
    print("✅ Generation 1 testing completed")