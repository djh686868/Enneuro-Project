"""Deterministic JSON/JSONL reporting for AutoML runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .protocol import AgentEvent, SearchResult


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class EventLogger:
    """Collect events and write a complete run report.

    The logger intentionally does not add wall-clock timestamps: reports from
    equal seeds can therefore be byte-for-byte compared in tests and CI.
    """

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: str, trial_id: int | None = None, **payload: Any) -> AgentEvent:
        record = AgentEvent(len(self.events), event, trial_id, payload)
        self.events.append(record)
        return record

    def write_jsonl(self, path: str | Path, events: Iterable[AgentEvent] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = events if events is not None else self.events
        with target.open("w", encoding="utf-8") as handle:
            for event in records:
                handle.write(json.dumps(event.to_dict(), sort_keys=True, default=_json_default) + "\n")
        return target

    def write_json(self, path: str | Path, report: SearchResult | dict[str, Any]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict() if isinstance(report, SearchResult) else report
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return target

    def write_report(self, directory: str | Path, report: SearchResult) -> tuple[Path, Path]:
        directory = Path(directory)
        return (
            self.write_json(directory / "report.json", report),
            self.write_jsonl(directory / "events.jsonl", report.events),
        )


# A concise alias used by callers that prefer the report-oriented name.
JSONReporter = EventLogger
