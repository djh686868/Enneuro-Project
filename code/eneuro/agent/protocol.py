"""Small, dependency-free protocols and value objects for the AutoML agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SearchSpace:
    """Search space used by :class:`SequentialRandomSearch`.

    ``lr`` is sampled uniformly in log10 space.  Batch sizes and optimizers are
    deliberately discrete so every emitted configuration is JSON serializable.
    """

    lr: tuple[float, float] = (1e-4, 1e-1)
    batch_size: tuple[int, ...] = (16, 32, 64)
    optimizer: tuple[str, ...] = ("sgd", "adam")

    def __post_init__(self) -> None:
        lo, hi = self.lr
        if lo <= 0 or hi <= lo:
            raise ValueError("lr must be a positive (low, high) interval")
        if not self.batch_size or any(int(v) <= 0 for v in self.batch_size):
            raise ValueError("batch_size must contain positive integers")
        if not self.optimizer or any(not str(v) for v in self.optimizer):
            raise ValueError("optimizer must contain at least one name")

    def sample(self, rng: Any) -> dict[str, Any]:
        import numpy as np

        return {
            "lr": float(10.0 ** rng.uniform(np.log10(self.lr[0]), np.log10(self.lr[1]))),
            "batch_size": int(rng.choice(self.batch_size)),
            "optimizer": str(rng.choice(self.optimizer)),
        }


@dataclass
class TrialResult:
    """Result returned by an executor; lower ``objective`` is better."""

    objective: float
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": float(self.objective),
            "metrics": dict(self.metrics),
            "status": self.status,
            "error": self.error,
        }


class Executor(Protocol):
    def execute(self, config: Mapping[str, Any]) -> TrialResult | Mapping[str, Any]:
        """Evaluate one configuration. Lower objective values are preferred."""


@dataclass(frozen=True)
class AgentEvent:
    """A deterministic event suitable for JSON and JSONL serialization."""

    index: int
    event: str
    trial_id: int | None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "event": self.event,
            "trial_id": self.trial_id,
            "payload": dict(self.payload),
        }


@dataclass
class SearchResult:
    trials: list[dict[str, Any]]
    events: list[AgentEvent]
    best_config: dict[str, Any] | None
    best_result: TrialResult | None
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "trials": self.trials,
            "events": [event.to_dict() for event in self.events],
            "best_config": self.best_config,
            "best_result": self.best_result.to_dict() if self.best_result else None,
        }
