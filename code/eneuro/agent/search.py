"""实现最小可运行的顺序随机搜索 Agent。

一次 trial 的固定顺序是 ``observe -> decide -> execute -> feedback``：
先读取历史结果，再提出配置，交给执行器训练，最后记录反馈并更新当前
最优值。该文件只负责编排，不绑定具体模型，便于后续替换成 TPE、贝叶斯
优化或 Successive Halving 策略。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

import numpy as np

from .logger import EventLogger
from .protocol import Executor, SearchResult, SearchSpace, TrialResult


class SequentialRandomSearch:
    """在固定预算内顺序执行可复现的随机搜索。

    当前版本采用依赖少、容易审计的随机搜索基线；每轮只执行一个 trial，
    适合先验证完整闭环，再逐步替换更复杂的决策策略。
    """

    def __init__(self, executor: Executor | Callable[[Mapping[str, Any]], Any],
                 search_space: SearchSpace | None = None, *, n_trials: int = 8,
                 seed: int = 0, logger: EventLogger | None = None):
        # trial 数必须为正，避免得到没有任何结果的空报告。
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        # executor 可以是实现 execute() 的对象，也可以是普通可调用函数。
        self.executor = executor
        self.search_space = search_space or SearchSpace()
        self.n_trials, self.seed = int(n_trials), int(seed)
        # 使用独立 Generator，不污染调用方的全局 NumPy 随机状态。
        self.rng = np.random.default_rng(self.seed)
        self.logger = logger or EventLogger()
        self.trials: list[dict[str, Any]] = []
        # 用规范化 JSON 保存已尝试配置，避免离散空间重复试验。
        self._seen_configs: set[str] = set()
        self._current_config: dict[str, Any] | None = None
        self._current_trial = -1

    def observe(self, trial_id: int | None = None) -> dict[str, Any]:
        """发布历史观测，作为本轮决策的输入。

        当前基线只发布已完成 trial 数和当前最优值；完整历史仍保存在
        ``self.trials`` 中，后续模型驱动策略可以直接利用。
        """
        tid = self._current_trial + 1 if trial_id is None else int(trial_id)
        observation = {"completed_trials": len(self.trials), "best_objective": self._best_value()}
        self.logger.emit("observe", tid, **observation)
        return observation

    def decide(self, trial_id: int | None = None) -> dict[str, Any]:
        """生成下一组未使用配置，并记录决策事件。"""
        tid = self._current_trial + 1 if trial_id is None else int(trial_id)
        config = self.search_space.sample(self.rng)
        key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        # 连续 log-scale 几乎不会重复；显式去重也让纯离散空间更可控。
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
        """调用执行器运行配置，并把异常转为可观测的失败 trial。

        单个配置失败时仍继续消耗剩余预算，失败目标使用正无穷，因而不会
        被错误地选为最优结果。
        """
        config = dict(config or self._current_config or {})
        tid = self._current_trial if trial_id is None else int(trial_id)
        try:
            raw = self.executor.execute(config) if hasattr(self.executor, "execute") else self.executor(config)
            result = raw if isinstance(raw, TrialResult) else TrialResult(float(raw["objective"]), dict(raw))
        except Exception as exc:  # 失败也必须可观测，并继续搜索。
            result = TrialResult(float("inf"), status="failed", error=f"{type(exc).__name__}: {exc}")
        self.logger.emit("execute", tid, config=config, result=result.to_dict())
        return result

    def feedback(self, config: Mapping[str, Any], result: TrialResult, trial_id: int | None = None) -> None:
        """保存本轮配置和结果，并发布反馈事件。"""
        tid = self._current_trial if trial_id is None else int(trial_id)
        record = {"trial_id": tid, "config": dict(config), "result": result.to_dict()}
        self.trials.append(record)
        self.logger.emit("feedback", tid, result=result.to_dict(), best_objective=self._best_value())

    def run(self) -> SearchResult:
        """按固定预算执行完整闭环并返回最终汇总。

        本版本不并发调度，因此事件顺序与 trial 顺序完全一致，便于调试
        和教学演示。
        """
        for trial_id in range(self.n_trials):
            self.observe(trial_id)
            config = self.decide(trial_id)
            result = self.execute(config, trial_id)
            self.feedback(config, result, trial_id)
        # 忽略失败 trial，只在成功结果中寻找最小 objective。
        best = min((t for t in self.trials if t["result"]["status"] == "completed"),
                   key=lambda t: t["result"]["objective"], default=None)
        best_result = TrialResult(**best["result"]) if best else None
        return SearchResult(self.trials, list(self.logger.events), best["config"] if best else None, best_result, self.seed)

    def _best_value(self) -> float | None:
        """读取当前成功 trial 中的最小目标值。"""
        values = [t["result"]["objective"] for t in self.trials if t["result"]["status"] == "completed"]
        return float(min(values)) if values else None


# 兼容更直观的调用命名；它们与核心实现完全相同。
AutoMLAgent = SequentialRandomSearch
SequentialRandomSearcher = SequentialRandomSearch
