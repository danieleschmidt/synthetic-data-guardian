import pytest
from synthetic_data_guardian.generator import generate_persons, generate_transactions
from synthetic_data_guardian.validator import validate_dataset, validate_column_distribution, correlation
from synthetic_data_guardian.watermark import embed_watermark, extract_watermark, verify_watermark
from synthetic_data_guardian.lineage import DataLineage
from fastapi.testclient import TestClient
from synthetic_data_guardian.app import app

client = TestClient(app)


def test_generate_persons():
    records = generate_persons(10, seed=42)
    assert len(records) == 10
    assert "name" in records[0]
    assert "email" in records[0]
    assert "age" in records[0]
    assert "salary" in records[0]

def test_generate_transactions():
    records = generate_transactions(5, seed=42)
    assert len(records) == 5
    assert "amount" in records[0]
    assert "currency" in records[0]

def test_generate_persons_deterministic():
    r1 = generate_persons(5, seed=99)
    r2 = generate_persons(5, seed=99)
    assert r1[0]["name"] == r2[0]["name"]

def test_validate_dataset_persons():
    records = generate_persons(50, seed=42)
    result = validate_dataset(records)
    assert result["valid"] is True
    assert result["record_count"] == 50

def test_validate_column_distribution_valid():
    values = [float(i) for i in range(10, 80)]
    result = validate_column_distribution(values, 0, 100)
    assert result["valid"] is True

def test_validate_column_distribution_invalid():
    values = [1000.0, 2000.0, 3000.0]  # way out of range
    result = validate_column_distribution(values, 0, 10, tolerance=0.01)
    assert result["valid"] is False

def test_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]  # perfectly correlated
    c = correlation(x, y)
    assert abs(c - 1.0) < 0.001

def test_watermark_embed_extract():
    records = generate_persons(20, seed=42)
    watermark = "TEST"
    watermarked = embed_watermark(records, watermark)
    extracted = extract_watermark(watermarked, len(watermark))
    assert extracted == watermark

def test_watermark_verify():
    records = generate_persons(20, seed=42)
    watermark = "WM01"
    watermarked = embed_watermark(records, watermark)
    result = verify_watermark(watermarked, watermark)
    assert result["verified"] is True

def test_watermark_preserves_record_count():
    records = generate_persons(10, seed=42)
    watermarked = embed_watermark(records, "AB")
    assert len(watermarked) == 10

def test_lineage_tracking():
    lineage = DataLineage(dataset_id="test-123")
    lineage.record("generate", 0, 100, {"type": "persons"})
    lineage.record("watermark", 100, 100, {})
    assert len(lineage.events) == 2
    assert lineage.events[0].transform == "generate"
    assert lineage.events[1].transform == "watermark"

def test_lineage_serialization():
    lineage = DataLineage(dataset_id="test-456")
    lineage.record("generate", 0, 50)
    d = lineage.to_dict()
    assert d["dataset_id"] == "test-456"
    assert len(d["events"]) == 1
    restored = DataLineage.from_dict(d)
    assert restored.dataset_id == "test-456"
    assert len(restored.events) == 1

def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200

def test_api_generate_persons():
    resp = client.post("/generate", json={"dataset_type": "persons", "count": 10, "seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) == 10
    assert data["validation"]["valid"] is True

def test_api_generate_with_watermark():
    resp = client.post("/generate", json={"dataset_type": "persons", "count": 20, "seed": 42, "watermark": "HI"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) == 20
    assert len(data["lineage"]["events"]) >= 2
