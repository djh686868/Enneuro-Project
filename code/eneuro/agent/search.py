"""Sequential random search with an explicit observe/decide/execute/feedback loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

import numpy as np

from .logger import EventLogger
from .protocol import Executor, SearchResult, SearchSpace, TrialResult


class SequentialRandomSearch:
    """Run a fixed number of reproducible trials in sequence."""

    def __init__(self, executor: Executor | Callable[[Mapping[str, Any]], Any],
                 search_space: SearchSpace | None = None, *, n_trials: int = 8,
                 seed: int = 0, logger: EventLogger | None = None):
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        self.executor = executor
        self.search_space = search_space or SearchSpace()
        self.n_trials, self.seed = int(n_trials), int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.logger = logger or EventLogger()
        self.trials: list[dict[str, Any]] = []
        self._seen_configs: set[str] = set()
        self._current_config: dict[str, Any] | None = None
        self._current_trial = -1

    def observe(self, trial_id: int | None = None) -> dict[str, Any]:
        """Publish prior observations before selecting the next configuration."""
        tid = self._current_trial + 1 if trial_id is None else int(trial_id)
        observation = {"completed_trials": len(self.trials), "best_objective": self._best_value()}
        self.logger.emit("observe", tid, **observation)
        return observation

    def decide(self, trial_id: int | None = None) -> dict[str, Any]:
        tid = self._current_trial + 1 if trial_id is None else int(trial_id)
        config = self.search_space.sample(self.rng)
        key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        # Continuous log-scale sampling almost never collides, but explicit
        # duplicate rejection keeps discrete-only spaces reproducible too.
        for _ in range(100):
            if key not in self._seen_configs:
                break
            config = self.search_space.sample(self.rng)
            key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        self._seen_configs.add(key)
        self._current_config, self._current_trial = config, tid
        self.logger.emit("decide", tid, config=config)
        return config

    def execute(self, config: Mapping[str, Any] | None = None, trial_id: int | None = None) -> TrialResult:
        config = dict(config or self._current_config or {})
        tid = self._current_trial if trial_id is None else int(trial_id)
        try:
            raw = self.executor.execute(config) if hasattr(self.executor, "execute") else self.executor(config)
            result = raw if isinstance(raw, TrialResult) else TrialResult(float(raw["objective"]), dict(raw))
        except Exception as exc:  # Keep a failed trial observable and continue searching.
            result = TrialResult(float("inf"), status="failed", error=f"{type(exc).__name__}: {exc}")
        self.logger.emit("execute", tid, config=config, result=result.to_dict())
        return result

    def feedback(self, config: Mapping[str, Any], result: TrialResult, trial_id: int | None = None) -> None:
        tid = self._current_trial if trial_id is None else int(trial_id)
        record = {"trial_id": tid, "config": dict(config), "result": result.to_dict()}
        self.trials.append(record)
        self.logger.emit("feedback", tid, result=result.to_dict(), best_objective=self._best_value())

    def run(self) -> SearchResult:
        for trial_id in range(self.n_trials):
            self.observe(trial_id)
            config = self.decide(trial_id)
            result = self.execute(config, trial_id)
            self.feedback(config, result, trial_id)
        best = min((t for t in self.trials if t["result"]["status"] == "completed"),
                   key=lambda t: t["result"]["objective"], default=None)
        best_result = TrialResult(**best["result"]) if best else None
        return SearchResult(self.trials, list(self.logger.events), best["config"] if best else None, best_result, self.seed)

    def _best_value(self) -> float | None:
        values = [t["result"]["objective"] for t in self.trials if t["result"]["status"] == "completed"]
        return float(min(values)) if values else None


# Friendly aliases for downstream callers.
AutoMLAgent = SequentialRandomSearch
SequentialRandomSearcher = SequentialRandomSearch
