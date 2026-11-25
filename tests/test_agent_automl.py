# 本文件验证的是“调参 agent 的控制流”和“可复现报告”，而不是某个具体
# 模型的最高精度。测试故意使用小规模/确定性执行器，保证运行快速稳定。
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "code")

from eneuro.agent import (EventLogger, MNISTExecutor, SearchSpace,
                          SequentialRandomSearch, SyntheticObjectiveExecutor, create_search)
from eneuro.agent.car_executor import CarBehavioralCloningExecutor, read_steering_rows
from eneuro.agent.donkeycar_executor import DonkeyCarSteeringExecutor
from eneuro.agent.evaluation import run_evaluation
from eneuro.agent.timed_search import run_timed_search
from eneuro.profiling import ProfileSession, current_profile


def test_search_is_reproducible_and_has_complete_event_cycle(tmp_path):
    """同一个 seed 应产生相同试验序列，并写出四类闭环事件。"""
    def run():
        logger = EventLogger()
        agent = SequentialRandomSearch(SyntheticObjectiveExecutor(), n_trials=5, seed=11, logger=logger)
        result = agent.run()
        return result, logger

    first, logger = run()
    second, _ = run()
    # 完整结果相等意味着配置采样、目标函数和事件编号都可复现。
    assert first.to_dict() == second.to_dict()
    assert [e.event for e in first.events] == [name for _ in range(5) for name in ("observe", "decide", "execute", "feedback")]
    json_path, jsonl_path = logger.write_report(tmp_path, first)
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 20
    assert json.loads(json_path.read_text(encoding="utf-8"))["seed"] == 11


def test_search_profiles_attempts_and_can_disable_an_ambient_session(tmp_path):
    active_runs = []

    def objective(config):
        active = current_profile()
        active_runs.append(active.run_id if active is not None else None)
        if len(active_runs) == 2:
            raise ValueError("failed trial")
        return {"objective": 0.5}

    with ProfileSession({"system_metrics": False}) as ambient:
        result = create_search(
            "local", objective, n_trials=2,
            profiling={"output_dir": tmp_path, "system_metrics": False, "max_marks": 1},
        ).run()
        disabled = create_search("local", objective, n_trials=1, profiling=False).run()
        assert current_profile() is ambient

    summary = result.profiling
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert report["run_id"] == summary["run_id"]
    assert active_runs == [report["run_id"], report["run_id"], None]
    assert report["timings"]["automl.trial"]["wall_ms"]["count"] == 2
    assert report["marks_dropped"] > 0
    records = [trial["result"] for trial in result.trials]
    assert [record["status"] for record in records] == ["completed", "failed"]
    assert all(record["profiling"]["run_id"] == report["run_id"] for record in records)
    assert all(record["profiling"]["end_offset_ns"] >= record["profiling"]["start_offset_ns"]
               for record in records)
    assert "profiling" not in disabled.to_dict()
    assert "profiling" not in disabled.trials[0]["result"]


def test_timed_search_profiles_refit_with_trial_session(tmp_path):
    class Executor:
        def execute(self, config, *, deadline, on_epoch):
            on_epoch(1, 0.4, 0.2)
            return SyntheticObjectiveExecutor().execute(config)

        def evaluate_final(self, config, *, on_epoch, checkpoint):
            on_epoch(1, 0.3, None)
            return {"test_score": 0.2}

    output = tmp_path / "timed"
    status = run_timed_search(
        Executor(), SearchSpace(), budget_s=10, trials=2, seed=1, output=output,
        protocol={"task": "test"}, plot_ylabel="Objective",
        format_validation=str, format_epoch_validation=str, format_test=str,
        profiling={"system_metrics": False},
    )
    assert status == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    profile = report["profiling"]
    trial_window = report["trials"][-1]["result"]["profiling"]
    refit_window = report["outer_test"]["profiling"]
    assert trial_window["run_id"] == refit_window["run_id"] == profile["run_id"]
    assert trial_window["end_offset_ns"] <= refit_window["start_offset_ns"]
    assert Path(profile["report_path"]).parent == output / "profiling"
    assert profile["timings"]["automl.trial"]["wall_ms"]["count"] == 2
    assert profile["timings"]["automl.refit"]["wall_ms"]["count"] == 1


def test_search_space_samples_log_scale_and_discrete_values():
    """确认学习率按 log10 均匀采样，其他超参保持离散取值。"""
    space = SearchSpace(lr=(1e-4, 1e-1), batch_size=(8, 32), optimizer=("sgd", "adam"))
    rng = np.random.default_rng(3)
    values = [space.sample(rng) for _ in range(200)]
    assert all(1e-4 <= c["lr"] <= 1e-1 for c in values)
    assert {c["batch_size"] for c in values} <= {8, 32}
    assert {c["optimizer"] for c in values} <= {"sgd", "adam"}
    # 对数均匀采样应覆盖低端和高端数量级，而不是集中在高学习率区域。
    assert min(c["lr"] for c in values) < 1e-3
    assert max(c["lr"] for c in values) > 1e-2


def test_mnist_executor_runs_real_minibatch_objective():
    """确认 MNIST 执行器确实完成了 minibatch 训练并返回合法准确率。"""
    executor = MNISTExecutor(seed=2, epochs=1, max_samples=60)
    with ProfileSession({"system_metrics": False}) as profile:
        result = executor.execute({"lr": 0.03, "batch_size": 16, "optimizer": "adam"})
    assert result.status == "completed"
    assert 0.0 <= result.objective <= 1.0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert profile.report["metrics"]["train_samples_per_s"] > 0
    assert profile.report["metrics"]["validation_samples_per_s"] > 0


@pytest.mark.parametrize("profiling", [False, {"system_metrics": False}])
def test_evaluation_locks_both_methods_before_outer_scoring(tmp_path, profiling):
    calls = []
    baseline = {"lr": 0.001, "batch_size": 16, "optimizer": "adam"}

    class Executor:
        def __init__(self, seed):
            self.seed = seed

        def execute(self, config):
            calls.append(("inner", config))
            if len(calls) == 1:
                return {"objective": float("nan")}
            return {"objective": 0.5 if config == baseline else self.seed / 10}

        def evaluate_final(self, config):
            assert [phase for phase, _ in calls[:3]] == ["inner"] * 3
            calls.append(("outer", config))
            return {"test_loss": 0.1 if config == baseline else 0.2}

    result = run_evaluation(
        Executor, SearchSpace(), baseline_config=baseline, n_trials=2, seed=1, runs=2,
        output=tmp_path, protocol={"task": "test", "objective": "val_loss", "test_metric": "test_loss"},
        profiling=profiling,
    )
    report = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert [phase for phase, _ in calls] == ["inner", "inner", "inner", "outer", "outer"] * 2
    assert calls[3][1] == result.best_config
    assert result.trials[0]["result"]["status"] == "failed"
    assert result.trials[0]["result"]["objective"] is None
    random_curve = report["anytime"]["curves"]["random_search"]
    assert random_curve[0]["n"] == 1 and random_curve[0]["se"] is None
    assert random_curve[1]["n"] == 2 and np.isclose(random_curve[1]["se"], 0.05)
    assert report["paired_comparison"]["n_pairs"] == 2
    assert ("profiling" in report) == bool(profiling)
    if profiling:
        profile = report["profiling"]
        assert profile == result.profiling
        assert profile["timings"]["automl.trial"]["wall_ms"]["count"] == 6
        assert profile["timings"]["automl.refit"]["wall_ms"]["count"] == 4
        for run in report["runs"]:
            for method in ("random_search", "fixed_baseline"):
                assert run[method]["outer_test"]["profiling"]["run_id"] == profile["run_id"]


def test_car_splits_keep_camera_views_of_each_csv_row_together(tmp_path):
    data_dir = tmp_path / "data"
    images = data_dir / "IMG"
    images.mkdir(parents=True)
    lines = ["center,left,right,steering"]
    for row in range(10):
        center, left, right = [f"{camera}-{row}.jpg" for camera in ("center", "left", "right")]
        lines.append(f"IMG/{center}, IMG\\{left}, data/IMG/{right},0")
        for name in (center, left, right):
            (images / name).touch()
    (data_dir / "driving_log.csv").write_text("\n".join(lines), encoding="utf-8")

    executor = CarBehavioralCloningExecutor(data_dir, cameras=("center", "left", "right"), max_samples=29)
    splits = executor.split_metadata()
    assert [splits[split]["images"] for split in ("train", "validation", "test")] == [24, 3, 2]
    assert splits["train"]["last_csv_row_index"] < splits["validation"]["first_csv_row_index"]
    assert splits["validation"]["last_csv_row_index"] < splits["test"]["first_csv_row_index"]


def test_car_multitarget_holdout_and_scaling(tmp_path):
    images = tmp_path / "IMG"
    images.mkdir()
    lines = ["center,left,right,steering,throttle,brake,speed"]
    for i in range(30):
        names = [f"{camera}-{i}.jpg" for camera in ("center", "left", "right")]
        for name in names:
            (images / name).touch()
        # 训练、验证和外层各有独立刹车记录，外层速度刻意超出开发集范围。
        brake = 0.5 if i in {5, 18, 28} else 0
        speed = i if i < 27 else 1000
        lines.append(f"{','.join(f'IMG/{name}' for name in names)},0.1,0.8,{brake},{speed}")
    (tmp_path / "driving_log.csv").write_text("\n".join(lines), encoding="utf-8")

    cameras = ("center", "left", "right")
    rows = read_steering_rows(tmp_path, camera=cameras, include_telemetry=True)
    assert np.allclose([row["targets"] for row in rows[:3]],
                       [[0.1, 0.8, 0, 0], [0.3, 0.8, 0, 0], [-0.1, 0.8, 0, 0]])
    executor = CarBehavioralCloningExecutor(tmp_path, cameras=cameras, model="resnet18")
    splits = executor.split_metadata()
    assert splits["validation"]["csv_row_ranges_inclusive"] == [[17, 19]]
    assert splits["train"]["braking_driving_rows"] == splits["validation"]["braking_driving_rows"] == 1
    assert splits["test"]["csv_row_ranges_inclusive"] == [[27, 29]]
    stats = executor.target_normalization()
    assert np.isclose(stats["speed"]["mean"], np.mean([i for i in range(27) if not 17 <= i <= 19]))
    assert stats["throttle"]["scale"] == 1.0


def test_donkeycar_holdout_orders_ids_and_preserves_steering(tmp_path):
    for image_id in (0, 1, 2, 10, 11, 12, 13, 14, 15, 16):
        (tmp_path / f"{image_id}_-0.0667.jpg").touch()
    (tmp_path / "13_1.9778.jpg").touch()
    (tmp_path / "invalid.jpg").touch()

    executor = DonkeyCarSteeringExecutor(tmp_path)
    splits = executor.split_metadata()
    assert [splits[name]["images"] for name in ("train", "validation", "test")] == [8, 1, 2]
    assert splits["train"]["last_image_id"] == 13
    assert splits["validation"]["first_image_id"] == 14
    assert splits["test"]["first_image_id"] == 15
    rows, train, _, _ = executor._splits
    assert [rows[i]["steering"] for i in train if rows[i]["group"] == 13] == [-0.0667, 1.9778]
    assert executor.target_names == ("steering",)
    assert executor.target_normalization() is None

