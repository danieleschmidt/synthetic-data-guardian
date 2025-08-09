# 🚀 Synthetic Data Guardian - Usage Examples

## Quick Start Examples

### 1. Basic Tabular Data Generation

```python
import asyncio
from synthetic_guardian import Guardian, PipelineBuilder

async def generate_customer_data():
    # Create Guardian instance
    guardian = Guardian()
    await guardian.initialize()
    
    # Build pipeline
    pipeline = (PipelineBuilder()
        .with_name("customer_pipeline")
        .with_generator("tabular", backend="simple")
        .with_schema({
            "age": {"type": "integer", "min": 18, "max": 80},
            "income": {"type": "float", "min": 20000, "max": 200000},
            "category": {"type": "categorical", "categories": ["A", "B", "C"]}
        })
        .add_validator("statistical", threshold=0.8)
        .add_validator("privacy", epsilon=1.0)
        .build())
    
    # Generate data
    result = await guardian.generate(
        pipeline_config=pipeline,
        num_records=1000,
        seed=42
    )
    
    print(f"Generated {len(result.data)} records")
    print(f"Quality Score: {result.quality_score:.2f}")
    
    await guardian.cleanup()

# Run the example
asyncio.run(generate_customer_data())
```

### 2. Time Series Data Generation

```python
import asyncio
from synthetic_guardian import Guardian, PipelineBuilder

async def generate_sensor_data():
    guardian = Guardian()
    await guardian.initialize()
    
    pipeline = (PipelineBuilder()
        .with_name("sensor_pipeline")
        .with_generator("timeseries", 
                       sequence_length=100,
                       features=["temperature", "humidity", "pressure"])
        .with_data_type("timeseries")
        .build())
    
    result = await guardian.generate(
        pipeline_config=pipeline,
        num_records=100,  # 100 time points
        seed=42
    )
    
    print(f"Generated time series with {len(result.data)} points")
    await guardian.cleanup()

asyncio.run(generate_sensor_data())
```

### 3. CLI Usage Examples

```bash
# Generate synthetic data from configuration
synthetic-guardian generate \
    --pipeline config.yaml \
    --num-records 10000 \
    --output synthetic_data.csv \
    --format csv

# Validate synthetic data quality
synthetic-guardian validate \
    --input synthetic_data.csv \
    --reference real_data.csv \
    --output validation_report.json

# Start API server
synthetic-guardian serve --port 8080

# Generate compliance report
synthetic-guardian report \
    --data ./data \
    --standard gdpr \
    --output compliance_report.pdf
```

### 4. Watermarking Example

```python
import asyncio
import pandas as pd
from synthetic_guardian import Guardian

async def watermark_example():
    guardian = Guardian()
    await guardian.initialize()
    
    # Sample data
    data = pd.DataFrame({
        'value1': [1, 2, 3, 4, 5],
        'value2': [10, 20, 30, 40, 50]
    })
    
    # Embed watermark
    watermark_result = await guardian.watermark(
        data=data,
        method='statistical',
        message='my_dataset_v1'
    )
    
    # Verify watermark
    verification = await guardian.verify_watermark(
        data=watermark_result['data'],
        method='statistical'
    )
    
    print(f"Watermark detected: {verification['is_watermarked']}")
    await guardian.cleanup()

asyncio.run(watermark_example())
```

### 5. Configuration File Example

Create `config.yaml`:

```yaml
pipeline:
  name: "financial_data_pipeline"
  description: "Generate synthetic financial transaction data"
  
generation:
  backend: "ctgan"
  epochs: 100
  batch_size: 500
  schema:
    transaction_id: "uuid"
    amount: 
      type: "float"
      min: 0.01
      max: 10000.0
    timestamp: "datetime"
    merchant_category: 
      type: "categorical"
      categories: ["retail", "food", "transport", "utilities"]
    
validation:
  validators:
    - type: "statistical"
      threshold: 0.9
    - type: "privacy" 
      epsilon: 1.0
    - type: "bias"
      protected_attributes: ["merchant_category"]

watermarking:
  enabled: true
  method: "statistical"
  strength: 0.8
```

## Advanced Features

### Pipeline Chaining

```python
# Create multiple pipelines and chain them
pipeline1 = (PipelineBuilder()
    .with_name("base_data")
    .with_generator("tabular")
    .build())

pipeline2 = (PipelineBuilder()
    .with_name("enriched_data")
    .with_generator("text")
    .build())

# Generate base data first
base_result = await guardian.generate(pipeline1, 1000)

# Use base data to generate enriched version
enriched_result = await guardian.generate(
    pipeline2, 
    1000,
    conditions={"base_data": base_result.data}
)
```

### Custom Validation

```python
from synthetic_guardian.validators import StatisticalValidator

# Run specific validators
validation_report = await guardian.validate(
    data=synthetic_data,
    validators=['statistical', 'privacy'],
    reference_data=original_data
)

# Check validation results
if validation_report.is_passed():
    print("✅ Data validation passed!")
else:
    print("❌ Validation failed:")
    for error in validation_report.get_all_errors():
        print(f"  - {error}")
```

### Enterprise Deployment

```python
# Production configuration
config = {
    'name': 'prod-synthetic-guardian',
    'log_level': 'INFO',
    'max_workers': 8,
    'enable_watermarking': True,
    'enable_lineage': True
}

guardian = Guardian(config=config)

# Use in production with proper error handling
try:
    result = await guardian.generate(pipeline, 50000, seed=42)
    
    # Save with metadata
    result.save_data('output/synthetic_data.parquet', 
                    format='parquet', 
                    include_metadata=True)
    
except Exception as e:
    logger.error(f"Generation failed: {str(e)}")
    # Handle error appropriately
```

## Testing Your Installation

```bash
# Basic functionality test
python3 -c "
import sys; sys.path.insert(0, 'src')
from synthetic_guardian.utils.logger import get_logger
print('✅ Synthetic Data Guardian core imports working!')
"

# Run minimal test suite  
python3 test_minimal.py
```

## Next Steps

1. **Install Dependencies:** `pip install pandas numpy scipy scikit-learn`
2. **Run Examples:** Try the code examples above
3. **Explore CLI:** Use `synthetic-guardian --help`
4. **Read Documentation:** Check README.md for detailed information
5. **Scale Up:** Deploy with Docker and enterprise configurations