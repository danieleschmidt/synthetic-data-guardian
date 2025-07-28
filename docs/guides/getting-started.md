# Getting Started with Synthetic Data Guardian

## Quick Start Guide

This guide will help you get up and running with Synthetic Data Guardian in under 10 minutes.

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git
- 8GB RAM minimum (16GB recommended)

### Installation Options

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/synthetic-data-guardian.git
cd synthetic-data-guardian

# Start the development environment
make dev

# Verify installation
curl http://localhost:8080/api/v1/health
```

#### Option 2: Python Package

```bash
# Install from PyPI
pip install synthetic-data-guardian

# Verify installation
synthetic-guardian --version
```

### First Synthetic Dataset

#### 1. Prepare Sample Data

Create a sample CSV file with customer data:

```csv
age,income,credit_score,email
25,45000,720,user1@example.com
34,67000,680,user2@example.com
42,89000,750,user3@example.com
```

#### 2. Generate Synthetic Data

```python
from synthetic_guardian import Guardian, GenerationPipeline

# Initialize the guardian
guardian = Guardian()

# Create a generation pipeline
pipeline = GenerationPipeline(
    name="customer_demo",
    description="Demo customer data generation"
)

# Configure the pipeline
pipeline.configure(
    generator="sdv",
    data_type="tabular",
    input_data="customer_sample.csv"
)

# Add validation
pipeline.add_validator("statistical_fidelity", threshold=0.9)
pipeline.add_validator("privacy_preservation", epsilon=1.0)

# Generate synthetic data
result = guardian.generate(
    pipeline=pipeline,
    num_records=1000
)

print(f"Generated {len(result.data)} records")
print(f"Quality score: {result.quality_score}")
print(f"Privacy score: {result.privacy_score}")
```

#### 3. Validate Results

```python
# View data statistics
print(result.data.describe())

# Check privacy metrics
print(f"Re-identification risk: {result.privacy_report.reidentification_risk:.2%}")

# Export synthetic data
result.data.to_csv("synthetic_customers.csv", index=False)
```

### Next Steps

1. **Explore Advanced Features**
   - Multi-modal data generation
   - Custom validation rules
   - Watermarking and lineage tracking

2. **Integration**
   - REST API usage
   - CI/CD pipeline integration
   - Cloud deployment

3. **Production Deployment**
   - Kubernetes configuration
   - Monitoring setup
   - Security hardening

### Common Issues

#### Memory Errors
- Reduce batch size or dataset size
- Increase Docker memory allocation
- Use streaming generation for large datasets

#### Poor Data Quality
- Increase training epochs
- Ensure sufficient input data
- Adjust generator parameters

#### Privacy Concerns
- Lower epsilon value for stronger privacy
- Enable additional privacy validators
- Review data minimization settings

### Getting Help

- 📖 [Documentation](https://docs.your-org.com/synthetic-guardian)
- 💬 [Community Discord](https://discord.gg/your-org)
- 🐛 [Issue Tracker](https://github.com/your-org/synthetic-data-guardian/issues)
- 📧 [Support Email](mailto:support@terragonlabs.com)

### What's Next?

- [User Guide](user-guide.md) - Comprehensive feature documentation
- [Developer Guide](developer-guide.md) - Advanced usage and integration
- [API Reference](../api/) - Complete API documentation
- [Examples](../examples/) - Real-world use cases and code samples