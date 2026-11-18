"""为四种搜索策略复用 observe -> decide -> execute -> feedback 训练闭环。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import count
import json
from math import erfc, isfinite, log10, pi, sqrt
from typing import Any

import numpy as np

from .logger import EventLogger
from .protocol import Executor, SearchResult, SearchSpace, TrialResult


class _SequentialSearch:

    def __init__(self, executor: Executor | Callable[[Mapping[str, Any]], Any],
                 search_space: SearchSpace | None = None, *, n_trials: int | None = 8,
                 seed: int = 0, logger: EventLogger | None = None):
        # trial 数必须为正，避免得到没有任何结果的空报告。
        if n_trials is not None and n_trials < 1:
            raise ValueError("n_trials must be positive")
        # executor 可以是实现 execute() 的对象，也可以是普通可调用函数。
        self.executor = executor
        self.search_space = search_space or SearchSpace()
        self.n_trials = None if n_trials is None else int(n_trials)
        self.seed = int(seed)
        # 使用独立 Generator，不污染调用方的全局 NumPy 随机状态。
        self.rng = np.random.default_rng(self.seed)
        self.logger = logger or EventLogger()
        self.trials: list[dict[str, Any]] = []
        # 用规范化 JSON 保存已尝试配置，避免离散空间重复试验。
        self._seen_configs: set[str] = set()
        self._current_config: dict[str, Any] | None = None
        self._current_trial = -1

    def observe(self, trial_id: int | None = None) -> dict[str, Any]:
        """发布试验数和当前最优目标，完整反馈保存在 self.trials。"""
        tid = self._current_trial + 1 if trial_id is None else int(trial_id)
        observation = {"completed_trials": len(self.trials), "best_objective": self._best_value()}
        self.logger.emit("observe", tid, **observation)
        return observation

    def decide(self, trial_id: int | None = None) -> dict[str, Any]:
        """生成下一组配置并记录决策事件。"""
        tid = self._current_trial + 1 if trial_id is None else int(trial_id)
        config = self._next_config()
        self._seen_configs.add(self._config_key(config))
        self._current_config, self._current_trial = config, tid
        self.logger.emit("decide", tid, config=config)
        return config

    @staticmethod
    def _config_key(config: Mapping[str, Any]) -> str:
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def _is_unseen(self, config: Mapping[str, Any]) -> bool:
        return self._config_key(config) not in self._seen_configs

    def _random_config(self) -> dict[str, Any]:
        config = self.search_space.sample(self.rng)
        # 连续 log-scale 几乎不会重复；显式去重也让纯离散空间更可控。
        for _ in range(100):
            if self._is_unseen(config):
                break
            config = self.search_space.sample(self.rng)
        return config

    def _next_config(self) -> dict[str, Any]:
        return self._random_config()

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
            if result.status == "completed" and not isfinite(result.objective):
                raise ValueError("trial objective must be finite")
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

    def run(self, *, should_stop: Callable[[], bool] | None = None) -> SearchResult:
        """执行闭环，直到达到试验数上限或外部停止条件。

        本版本不并发调度，因此事件顺序与 trial 顺序完全一致，便于调试
        和教学演示。
        """
        if self.n_trials is None and should_stop is None:
            raise ValueError("unbounded trial count requires a stop condition")
        trial_ids = count() if self.n_trials is None else range(self.n_trials)
        for trial_id in trial_ids:
            if should_stop is not None and should_stop():
                break
            self.observe(trial_id)
            config = self.decide(trial_id)
            result = self.execute(config, trial_id)
            self.feedback(config, result, trial_id)
        return self.result()

    def result(self) -> SearchResult:
        """汇总已经收到反馈的试验，运行中断时也可调用。"""
        # 忽略失败 trial，只在成功结果中寻找最小 objective。
        best = min(self._successful_trials(), key=_objective, default=None)
        best_result = TrialResult(**best["result"]) if best else None
        return SearchResult(self.trials, list(self.logger.events), best["config"] if best else None, best_result, self.seed)

    def _best_value(self) -> float | None:
        """读取当前成功 trial 中的最小目标值。"""
        return min(map(_objective, self._successful_trials()), default=None)

    def _successful_trials(self, trials=None):
        return [
            trial for trial in (self.trials if trials is None else trials)
            if trial["result"]["status"] == "completed" and _objective(trial) is not None
        ]


def _objective(trial: Mapping[str, Any]) -> float | None:
    return trial["result"]["objective"]


class SequentialRandomSearch(_SequentialSearch):
    """逐轮随机采样，在成功试验中选择最佳配置。"""


def _neighbor(config: Mapping[str, Any], space: SearchSpace, rng: np.random.Generator) -> dict[str, Any]:
    """在 log10 学习率、相邻批大小或其他优化器中改变一项。"""
    batch_sizes = sorted(set(space.batch_size))
    batch_index = batch_sizes.index(config["batch_size"])
    adjacent = batch_sizes[max(0, batch_index - 1):batch_index] + batch_sizes[batch_index + 1:batch_index + 2]
    other_optimizers = [name for name in space.optimizer if name != config["optimizer"]]
    fields = ["lr"]
    if adjacent:
        fields.append("batch_size")
    if other_optimizers:
        fields.append("optimizer")
    field = rng.choice(fields)

    candidate = dict(config)
    if field == "lr":
        lo, hi = map(log10, space.lr)
        center = log10(config["lr"])
        step = 0.1 * (hi - lo)
        candidate["lr"] = float(np.clip(
            10.0 ** rng.uniform(max(lo, center - step), min(hi, center + step)), *space.lr,
        ))
    elif field == "batch_size":
        candidate["batch_size"] = int(rng.choice(adjacent))
    else:
        candidate["optimizer"] = str(rng.choice(other_optimizers))
    return candidate


class LocalSearch(_SequentialSearch):
    """围绕当前最优的成功配置逐项探索邻域。"""

    def _next_config(self) -> dict[str, Any]:
        best = min(self._successful_trials(), key=_objective, default=None)
        if best is not None:
            for _ in range(16):
                candidate = _neighbor(best["config"], self.search_space, self.rng)
                if self._is_unseen(candidate):
                    return candidate
        return self._random_config()


class EvolutionarySearch(_SequentialSearch):
    """按代选择父代，逐个评估交叉和变异后的子代。"""

    _population_size = 6

    def _next_config(self) -> dict[str, Any]:
        size = self._population_size
        generation_start = len(self.trials) // size * size
        if generation_start == 0:
            return self._random_config()
        previous = self.trials[generation_start - size:generation_start]
        parents = sorted(self._successful_trials(previous), key=_objective)[:size // 2]
        if len(parents) >= 2:
            for _ in range(16):
                first, second = self.rng.choice(len(parents), size=2, replace=False)
                a, b = parents[first]["config"], parents[second]["config"]
                crossed = {name: (a[name] if self.rng.integers(2) else b[name]) for name in a}
                candidate = _neighbor(crossed, self.search_space, self.rng)
                if self._is_unseen(candidate):
                    return candidate
        return self._random_config()


class BayesianOptimizationSearch(_SequentialSearch):
    """以高斯过程估计目标值，用期望改进选择下一组超参数。"""

    def _encode(self, configs: list[dict[str, Any]]) -> np.ndarray:
        lo, hi = map(log10, self.search_space.lr)
        span = hi - lo or 1.0
        batch_sizes = tuple(dict.fromkeys(self.search_space.batch_size))
        optimizers = tuple(dict.fromkeys(self.search_space.optimizer))
        return np.asarray([
            [(log10(config["lr"]) - lo) / span]
            + [config["batch_size"] == size for size in batch_sizes]
            + [config["optimizer"] == name for name in optimizers]
            for config in configs
        ], dtype=float)

    @staticmethod
    def _kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # 学习率已缩放到 [0, 1]，离散变量在独热编码空间中比较。
        distances = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-distances)

    def _next_config(self) -> dict[str, Any]:
        successful = self._successful_trials()
        if len(successful) < 5:
            return self._random_config()

        # 保留历史最优，并限制拟合规模以控制限时搜索中的决策开销。
        history = successful[-127:]
        best = min(successful, key=_objective)
        if not any(trial is best for trial in history):
            history = [best, *history]
        candidates = [self.search_space.sample(self.rng) for _ in range(128)]
        candidates = [config for config in candidates if self._is_unseen(config)]
        if not candidates:
            return self._random_config()

        observed = self._encode([trial["config"] for trial in history])
        proposed = self._encode(candidates)
        values = np.asarray([_objective(trial) for trial in history], dtype=float)
        values = values / (np.max(np.abs(values)) or 1.0)
        values -= values.mean()
        values /= values.std() or 1.0
        covariance = self._kernel(observed, observed) + 1e-6 * np.eye(len(history))
        cross_covariance = self._kernel(observed, proposed)
        try:
            factor = np.linalg.cholesky(covariance)
            weights = np.linalg.solve(factor, values)
            projection = np.linalg.solve(factor, cross_covariance)
        except np.linalg.LinAlgError:
            return self._random_config()

        mean = weights @ projection
        variance = 1.0 - np.sum(projection ** 2, axis=0)
        deviation = np.sqrt(np.maximum(variance, 1e-12))
        improvement = values.min() - mean
        z = improvement / deviation
        cdf = np.asarray([0.5 * erfc(-value / sqrt(2.0)) for value in z])
        density = np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi)
        expected_improvement = improvement * cdf + deviation * density
        return candidates[int(np.argmax(expected_improvement))]


_SEARCHERS = {
    "random": SequentialRandomSearch,
    "local": LocalSearch,
    "evolutionary": EvolutionarySearch,
    "beyesian": BayesianOptimizationSearch,
}
SEARCH_STRATEGIES = tuple(_SEARCHERS)


def create_search(strategy: str, executor: Executor | Callable[[Mapping[str, Any]], Any],
                  search_space: SearchSpace | None = None, *, n_trials: int | None = 8,
                  seed: int = 0, logger: EventLogger | None = None) -> _SequentialSearch:
    """通过公开策略 ID 创建搜索器，共用同一个训练执行器接口。"""
    if strategy not in _SEARCHERS:
        raise ValueError(f"unknown search strategy {strategy!r}; choose from {', '.join(SEARCH_STRATEGIES)}")
    return _SEARCHERS[strategy](executor, search_space, n_trials=n_trials, seed=seed, logger=logger)


# 兼容更直观的调用命名；它们与核心实现完全相同。
AutoMLAgent = SequentialRandomSearch
SequentialRandomSearcher = SequentialRandomSearch
