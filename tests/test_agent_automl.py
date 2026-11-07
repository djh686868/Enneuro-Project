import json
import sys

import numpy as np

sys.path.insert(0, "code")

from eneuro.agent import (EventLogger, MNISTExecutor, SearchSpace,
                          SequentialRandomSearch, SyntheticObjectiveExecutor)


def test_search_is_reproducible_and_has_complete_event_cycle(tmp_path):
    def run():
        logger = EventLogger()
        agent = SequentialRandomSearch(SyntheticObjectiveExecutor(), n_trials=5, seed=11, logger=logger)
        result = agent.run()
        return result, logger

    first, logger = run()
    second, _ = run()
    assert first.to_dict() == second.to_dict()
    assert [e.event for e in first.events] == [name for _ in range(5) for name in ("observe", "decide", "execute", "feedback")]
    json_path, jsonl_path = logger.write_report(tmp_path, first)
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 20
    assert json.loads(json_path.read_text(encoding="utf-8"))["seed"] == 11


def test_search_space_samples_log_scale_and_discrete_values():
    space = SearchSpace(lr=(1e-4, 1e-1), batch_size=(8, 32), optimizer=("sgd", "adam"))
    rng = np.random.default_rng(3)
    values = [space.sample(rng) for _ in range(200)]
    assert all(1e-4 <= c["lr"] <= 1e-1 for c in values)
    assert {c["batch_size"] for c in values} <= {8, 32}
    assert {c["optimizer"] for c in values} <= {"sgd", "adam"}
    # Uniform log sampling should put observations in both outer decades.
    assert min(c["lr"] for c in values) < 1e-3
    assert max(c["lr"] for c in values) > 1e-2


def test_mnist_executor_runs_real_minibatch_objective():
    executor = MNISTExecutor(seed=2, epochs=1, max_samples=60)
    result = executor.execute({"lr": 0.03, "batch_size": 16, "optimizer": "adam"})
    assert result.status == "completed"
    assert 0.0 <= result.objective <= 1.0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0

