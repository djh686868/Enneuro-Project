# 本文件验证的是“调参 agent 的控制流”和“可复现报告”，而不是某个具体
# 模型的最高精度。测试故意使用小规模/确定性执行器，保证运行快速稳定。
import json
import sys

import numpy as np

sys.path.insert(0, "code")

from eneuro.agent import (EventLogger, MNISTExecutor, SearchSpace,
                          SequentialRandomSearch, SyntheticObjectiveExecutor)


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
    result = executor.execute({"lr": 0.03, "batch_size": 16, "optimizer": "adam"})
    assert result.status == "completed"
    assert 0.0 <= result.objective <= 1.0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0

