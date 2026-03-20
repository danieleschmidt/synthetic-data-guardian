# synthetic-data-guardian

Enterprise synthetic data pipeline with validation, watermarking, and lineage tracking.

## Features
- **Generation**: Faker-based person and transaction datasets
- **Validation**: Column distribution checks and correlation preservation
- **Watermarking**: LSB steganography for data provenance
- **Lineage**: JSON tracking of every transform applied

## Usage
```bash
pip install -r requirements.txt
uvicorn synthetic_data_guardian.app:app --reload
```

## API
- `POST /generate` — Generate synthetic dataset
- `POST /validate` — Validate a dataset
- `GET /health` — Health check

## Tests
```bash
pytest tests/
```
