"""Lineage tracking for synthetic data transforms."""
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LineageEvent:
    transform: str
    timestamp: float
    input_count: int
    output_count: int
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DataLineage:
    dataset_id: str
    created_at: float = field(default_factory=time.time)
    events: List[LineageEvent] = field(default_factory=list)

    def record(self, transform: str, input_count: int, output_count: int,
               params: Optional[Dict] = None, notes: str = "") -> None:
        self.events.append(LineageEvent(
            transform=transform,
            timestamp=time.time(),
            input_count=input_count,
            output_count=output_count,
            params=params or {},
            notes=notes,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "events": [asdict(e) for e in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataLineage":
        lineage = cls(dataset_id=data["dataset_id"], created_at=data["created_at"])
        for e in data.get("events", []):
            lineage.events.append(LineageEvent(**e))
        return lineage
