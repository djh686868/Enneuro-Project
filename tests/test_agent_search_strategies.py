import sys

import numpy as np
import pytest

sys.path.insert(0, "code")

from eneuro.agent import SearchSpace, TrialResult, create_search


def _objective(config):
    return TrialResult(np.log10(config["lr"]) + config["batch_size"] / 100)


def _configs(result):
    return [trial["config"] for trial in result.trials]


@pytest.mark.parametrize("strategy", ["random", "local", "evolutionary", "beyesian"])
def test_strategy_reuses_executor_and_is_reproducible(strategy):
    space = SearchSpace(lr=(1e-5, 1e-2), batch_size=(8, 16, 32), optimizer=("sgd", "adam"))

    def run():
        return create_search(strategy, _objective, space, n_trials=13, seed=4).run()

    result = run()
    assert result.to_dict() == run().to_dict()
    assert len(result.trials) == 13
    assert len(result.events) == 4 * len(result.trials)
    assert result.best_result.objective == min(_objective(config).objective for config in _configs(result))
    for config in _configs(result):
        assert space.lr[0] <= config["lr"] <= space.lr[1]
        assert config["batch_size"] in space.batch_size
        assert config["optimizer"] in space.optimizer


def test_local_search_perturbs_one_coordinate_after_success():
    space = SearchSpace(batch_size=(16,), optimizer=("adam",))
    result = create_search("local", _objective, space, n_trials=2, seed=1).run()
    initial, neighbor = _configs(result)
    assert neighbor["batch_size"] == initial["batch_size"]
    assert neighbor["optimizer"] == initial["optimizer"]
    assert 0 < abs(np.log10(neighbor["lr"] / initial["lr"])) <= 0.3


@pytest.mark.parametrize(("strategy", "warmup"), [("evolutionary", 6), ("beyesian", 5)])
def test_adaptive_strategy_uses_feedback_after_random_warmup(strategy, warmup):
    def reverse_objective(config):
        return TrialResult(-_objective(config).objective)

    forward = _configs(create_search(strategy, _objective, n_trials=warmup + 1, seed=4).run())
    reverse = _configs(create_search(strategy, reverse_objective, n_trials=warmup + 1, seed=4).run())
    random = _configs(create_search("random", _objective, n_trials=warmup + 1, seed=4).run())
    assert forward[:warmup] == reverse[:warmup] == random[:warmup]
    assert forward[warmup] != reverse[warmup]


def test_bo_warmup_excludes_failed_trials_and_respects_stop_condition():
    def fail(config):
        raise RuntimeError("training failed")

    agent = create_search("beyesian", fail, n_trials=None, seed=4)
    result = agent.run(should_stop=lambda: len(agent.trials) == 7)
    random = create_search("random", fail, n_trials=7, seed=4).run()
    assert _configs(result) == _configs(random)
    assert all(trial["result"]["status"] == "failed" for trial in result.trials)
    assert result.best_config is None
