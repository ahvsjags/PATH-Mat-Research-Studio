from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MEMORY_KINDS = {
    "evidence_memory",
    "world_state_memory",
    "episodic_memory",
    "policy_memory",
    "claim_memory",
}
CLAIM_STATUSES = {"verified", "proxy", "hypothesis", "protocol", "external_pending"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    kind: str
    payload: dict[str, Any]
    source_ids: tuple[str, ...] = ()
    claim_status: str | None = None
    sequence: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"unknown memory kind: {self.kind}")
        if self.claim_status is not None and self.claim_status not in CLAIM_STATUSES:
            raise ValueError(f"unknown claim status: {self.claim_status}")
        if not self.content_hash:
            body = {
                "record_id": self.record_id,
                "kind": self.kind,
                "payload": self.payload,
                "source_ids": self.source_ids,
                "claim_status": self.claim_status,
                "sequence": self.sequence,
            }
            object.__setattr__(self, "content_hash", hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """Append-only, deterministic memory for the central brain.

    Records are never silently overwritten. The store is intentionally small
    and dependency-free; a later service layer can persist the same contract
    in SQLite or an event database without changing the Agent protocol.
    """

    def __init__(self, records: Iterable[MemoryRecord] | None = None) -> None:
        self._records: list[MemoryRecord] = list(records or [])
        self._next_sequence = max((record.sequence for record in self._records), default=-1) + 1

    def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        record_id: str | None = None,
        source_ids: Iterable[str] = (),
        claim_status: str | None = None,
    ) -> MemoryRecord:
        record_id = record_id or f"{kind}:{self._next_sequence:06d}"
        record = MemoryRecord(
            record_id=record_id,
            kind=kind,
            payload=dict(payload),
            source_ids=tuple(sorted(set(str(item) for item in source_ids))),
            claim_status=claim_status,
            sequence=self._next_sequence,
        )
        self._records.append(record)
        self._next_sequence += 1
        return record

    def records(self, kind: str | None = None) -> list[MemoryRecord]:
        return [record for record in self._records if kind is None or record.kind == kind]

    def get(self, record_id: str) -> MemoryRecord | None:
        return next((record for record in self._records if record.record_id == record_id), None)

    def has_sources(self, source_ids: Iterable[str]) -> bool:
        available = {source_id for record in self._records for source_id in record.source_ids}
        return set(str(item) for item in source_ids).issubset(available)

    def snapshot(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(canonical_json(row) for row in self.snapshot()) + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MemoryStore":
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(MemoryRecord(**json.loads(line)))
        return cls(records)
