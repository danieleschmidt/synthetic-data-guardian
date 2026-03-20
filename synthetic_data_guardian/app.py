"""FastAPI endpoints for synthetic data guardian."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .generator import generate_persons, generate_transactions
from .validator import validate_dataset
from .watermark import embed_watermark, verify_watermark
from .lineage import DataLineage

app = FastAPI(title="Synthetic Data Guardian", version="0.1.0")


class GenerateRequest(BaseModel):
    dataset_type: str = "persons"
    count: int = 100
    seed: Optional[int] = None
    watermark: Optional[str] = None


class GenerateResponse(BaseModel):
    records: List[Dict[str, Any]]
    lineage: Dict[str, Any]
    validation: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate synthetic data with optional watermarking."""
    lineage = DataLineage(dataset_id=f"{req.dataset_type}_{int(__import__('time').time())}")

    if req.dataset_type == "persons":
        records = generate_persons(req.count, seed=req.seed)
    elif req.dataset_type == "transactions":
        records = generate_transactions(req.count, seed=req.seed)
    else:
        raise HTTPException(400, f"Unknown dataset type: {req.dataset_type}")

    lineage.record("generate", 0, len(records), {"type": req.dataset_type, "count": req.count})

    if req.watermark:
        records = embed_watermark(records, req.watermark)
        lineage.record("watermark", len(records), len(records), {"watermark_len": len(req.watermark)})

    validation = validate_dataset(records)
    lineage.record("validate", len(records), len(records), {"valid": validation["valid"]})

    return GenerateResponse(records=records, lineage=lineage.to_dict(), validation=validation)


@app.post("/validate")
def validate(records: List[Dict[str, Any]]):
    """Validate a dataset."""
    return validate_dataset(records)
