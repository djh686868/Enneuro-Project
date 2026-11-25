import json

import numpy as np
import pytest

from eneuro.base import Tensor
from eneuro.data import DataLoader
from eneuro.nn.loss import softmaxCrossEntropy
from eneuro.nn.module import Linear, Sequential
from eneuro.nn.optim import SGD
from eneuro.profiling import ProfileSession, current_profile


def test_profile_records_samples_and_failed_workload(tmp_path, monkeypatch):
    closed = []

    class Collector:
        def __init__(self, gpu):
            assert gpu
            self.count = 0

        def sample(self):
            self.count += 1
            return {
                "host": {"memory_used_bytes": 100 * self.count},
                "process": {"rss_bytes": 10 * self.count},
                "gpus": [{"index": 0, "uuid": "fake-gpu", "name": "fake",
                          "memory_used_bytes": 20 * self.count}],
            }

        def close(self):
            closed.append(True)

    monkeypatch.setattr("eneuro.profiling.session.SystemCollector", Collector)
    profile = ProfileSession({"output_dir": tmp_path, "max_marks": 1})
    with pytest.raises(RuntimeError, match="training failed"):
        with profile:
            assert current_profile() is profile
            assert json.loads(profile.report_path.read_text(encoding="utf-8"))["status"] == "running"
            profile.mark("first")
            with ProfileSession(False) as disabled:
                assert current_profile() is None
                disabled.mark("ignored")
            assert disabled.report is None
            assert current_profile() is profile
            profile.mark("second", loss=float("nan"))
            with profile.span("train"):
                raise RuntimeError("training failed")

    assert current_profile() is None
    assert closed == [True]
    report = json.loads(profile.report_path.read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in profile.samples_path.read_text(encoding="utf-8").splitlines()]
    assert len(samples) == report["sample_count"] >= 1
    assert report["peaks"]["host"]["memory_used_bytes"] == 100 * len(samples)
    assert report["peaks"]["gpus"]["fake-gpu"]["memory_used_bytes"] == 20 * len(samples)
    assert report["status"] == "failed"
    assert report["timings"]["train"]["wall_ms"]["count"] == 1
    assert report["marks"][0]["metadata"] == {"loss": None}
    assert report["marks_dropped"] == 1


def test_trainer_profiling_preserves_updates_and_counts_sampled_phases(tmp_path):
    from eneuro.train import Trainer

    features = np.arange(12, dtype=np.float32).reshape(6, 2) / 12
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    loader = DataLoader(list(zip(features, labels)), batch_size=2)
    validation = DataLoader(list(zip(features, labels)), batch_size=6)

    def run(config):
        np.random.seed(3)
        model = Sequential(Linear(2))
        model(Tensor(features[:1]))
        trainer = Trainer(model, softmaxCrossEntropy, SGD(model.get_params_list(), lr=0.01))
        trainer.fit(loader, validation, epochs=1, batch_size=2, verbose=False, profiling=config)
        return trainer, [param.data.copy() for param in model.get_params_list()]

    trainer, profiled = run({"output_dir": tmp_path, "system_metrics": False, "batch_every": 2})
    baseline, unprofiled = run(False)
    assert baseline.profile_path is baseline.profile_report is None
    for profiled_param, baseline_param in zip(profiled, unprofiled):
        np.testing.assert_array_equal(profiled_param, baseline_param)

    report = trainer.profile_report
    assert report["metrics"]["train_samples"] == 6
    assert report["metrics"]["train_samples_per_s"] > 0
    for phase in ("data_wait", "transfer", "forward", "loss", "backward", "optimizer"):
        assert report["timings"][phase]["wall_ms"]["count"] == 2
    assert report["timings"]["validation_forward"]["wall_ms"]["count"] == 1
    assert trainer.profile_path.is_file()
