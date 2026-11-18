"""固定外层留出集上的调参评估、基线比较和 anytime 曲线。"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping, Sequence
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .logger import EventLogger
from .protocol import SearchResult, SearchSpace, TrialResult
from .search import SEARCH_STRATEGIES, SequentialRandomSearch, create_search


class EvaluationExecutor(Protocol):
    def execute(self, config: Mapping[str, Any]) -> TrialResult:
        """仅用内层数据训练、评分。"""

    def evaluate_final(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """配置确定后，在开发集重训并在外层留出集评分一次。"""


def _search_method(strategy: str) -> str:
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(f"unknown search strategy {strategy!r}; choose from {', '.join(SEARCH_STRATEGIES)}")
    return f"{strategy}_search"


def _anytime_curve(runs: list[list[dict]], budget: int) -> list[dict]:
    """按已尝试的配置数对齐，失败尝试仍占预算，最优值保持到下次改进。"""
    curves = []
    for trials in runs:
        incumbent = None
        curve = []
        for trial in trials:
            result = trial["result"]
            score = result["objective"]
            if result["status"] == "completed" and score is not None and isfinite(score):
                incumbent = score if incumbent is None else min(incumbent, score)
            curve.append(incumbent)
        curves.append(curve + [incumbent] * (budget - len(curve)))

    points = []
    for attempt, scores in enumerate(zip(*curves), start=1):
        valid = [score for score in scores if score is not None]
        n = len(valid)
        sd = float(np.std(valid, ddof=1)) if n >= 2 else None
        points.append({
            "evaluations": attempt,
            "n": n,
            "mean": float(np.mean(valid)) if n else None,
            "sd": sd,
            "se": sd / sqrt(n) if sd is not None else None,
        })
    return points


def _paired_comparison(runs: list[dict], metric: str, search_method: str) -> dict:
    """按相同种子配对。对绝对差值取平均秩，枚举符号和的分布（含并列秩）。"""
    pairs = []
    for run in runs:
        scores = [run[method]["outer_test"].get(metric) for method in (search_method, "fixed_baseline")]
        if all(score is not None and isfinite(score) for score in scores):
            pairs.append({"seed": run["seed"], search_method: scores[0],
                          "fixed_baseline": scores[1], "difference": scores[0] - scores[1]})

    # 四舍五入仅用于判定浮点产生的近似并列（例如同一个准确率差）。
    differences = [pair["difference"] for pair in pairs]
    rounded = [round(difference, 12) for difference in differences]
    nonzero = sorted((difference for difference in rounded if difference), key=abs)
    n = len(nonzero)
    summary = {
        "metric": metric, "pairs": pairs, "n_pairs": len(pairs), "n_nonzero": n,
        "mean_difference": float(np.mean(differences)) if pairs else None,
        "median_difference": float(np.median(differences)) if pairs else None,
        "p_value": None,
    }
    # 小于 6 个非零配对时，精确双侧检验的最小 p 值也大于 0.05。
    if n < 6 or len(pairs) != len(runs):
        summary["test_status"] = "insufficient_nonzero_pairs" if len(pairs) == len(runs) else "missing_pairs"
        return summary

    rank_counts = {0: 1}
    observed = total = start = 0
    while start < n:
        end = start + 1
        while end < n and abs(nonzero[end]) == abs(nonzero[start]):
            end += 1
        # 双倍平均秩保留整数，动态规划只需统计各秩和出现的次数。
        rank = start + 1 + end
        for difference in nonzero[start:end]:
            observed += rank if difference > 0 else 0
            total += rank
            expanded = rank_counts.copy()
            for rank_sum, count in rank_counts.items():
                expanded[rank_sum + rank] = expanded.get(rank_sum + rank, 0) + count
            rank_counts = expanded
        start = end
    extreme = sum(count for rank_sum, count in rank_counts.items()
                  if abs(2 * rank_sum - total) >= abs(2 * observed - total))
    summary.update(test_status="computed", test="exact_two_sided_wilcoxon_signed_rank",
                   p_value=extreme / (2 ** n))
    return summary


def plot_anytime(output: Path, curves: dict, objective: str, runs: int, *,
                 x_key: str = "evaluations") -> bool:
    try:
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return False

    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.subplots()
    labels = {
        "random_search": "Random search",
        "local_search": "Local search",
        "evolutionary_search": "Evolutionary search",
        "beyesian_search": "Bayesian optimization",
        "fixed_baseline": "Fixed baseline",
    }
    for (method, points), color in zip(curves.items(), ("#126782", "#b35923")):
        x = [point[x_key] for point in points]
        mean = np.array([point["mean"] for point in points], dtype=float)
        se = np.array([point["se"] for point in points], dtype=float)
        label = labels.get(method, method)
        axis.step(x, mean, where="post", color=color, label=f"{label} (up to n={runs})")
        axis.scatter(x, mean, s=14, color=color)
        axis.fill_between(x, mean - se, mean + se, step="post", color=color, alpha=0.2)
    x_label = "Configuration attempts (including failures)"
    if x_key == "elapsed_s":
        x_label = "Search wall time (seconds)"
        axis.set_xlim(left=0)
    else:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    title = "Anytime performance (mean ± SE)" if runs > 1 else "Anytime performance"
    axis.set(xlabel=x_label, ylabel=objective, title=title)
    axis.legend()
    FigureCanvasSVG(figure).print_svg(str(output))
    return True


def _outer_score(executor: EvaluationExecutor, config: dict | None, metric: str) -> dict:
    if config is None:
        return {"status": "skipped", "error": "no valid inner evaluation"}
    try:
        metrics = executor.evaluate_final(config)
        if not isfinite(metrics[metric]):
            raise ValueError("outer test metric must be finite")
        return {**metrics, "status": "completed"}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def run_evaluation(executor_factory: Callable[[int], EvaluationExecutor], search_space: SearchSpace, *,
                   baseline_config: dict, n_trials: int, seed: int, runs: int,
                   output: str | Path, protocol: Mapping[str, Any], strategy: str = "random") -> SearchResult:
    """每个种子用相同协议比较两种方法，先锁定各自配置，再做外层评估。"""
    if runs < 1:
        raise ValueError("runs must be positive")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    search_method = _search_method(strategy)
    directory = Path(output)
    objective = protocol["objective"]
    metric = protocol["test_metric"]
    records = []
    primary = None
    for run_seed in range(seed, seed + runs):
        search_executor = executor_factory(run_seed)
        baseline_executor = executor_factory(run_seed)
        logger = EventLogger()
        agent = create_search(strategy, search_executor, search_space, n_trials=n_trials,
                              seed=run_seed, logger=logger)
        start = time.perf_counter()
        result = agent.run()
        search_elapsed = time.perf_counter() - start
        if primary is None:
            primary = result
            logger.write_json(directory / "report.json", {**result.to_dict(), "strategy": strategy})
            logger.write_jsonl(directory / "events.jsonl", result.events)

        baseline = SequentialRandomSearch(baseline_executor, n_trials=1, seed=run_seed)
        start = time.perf_counter()
        fixed_result = baseline.execute(baseline_config, 0)
        fixed_elapsed = time.perf_counter() - start
        fixed_config = baseline_config if fixed_result.status == "completed" else None

        # 此处才访问测试集。指标只写评估报告，不传回搜索器。
        records.append({
            "seed": run_seed,
            search_method: {
                "trials": result.trials, "selected_config": result.best_config,
                "search_elapsed_s": search_elapsed,
                "outer_test": _outer_score(search_executor, result.best_config, metric),
            },
            "fixed_baseline": {
                "trials": [{"trial_id": 0, "config": baseline_config, "result": fixed_result.to_dict()}],
                "selected_config": fixed_config, "search_elapsed_s": fixed_elapsed,
                "outer_test": _outer_score(baseline_executor, fixed_config, metric),
            },
        })

    curves = {method: _anytime_curve([run[method]["trials"] for run in records], n_trials)
              for method in (search_method, "fixed_baseline")}
    has_plot = plot_anytime(directory / "anytime.svg", curves, objective, runs)
    EventLogger().write_json(directory / "evaluation.json", {
        "protocol": dict(protocol),
        "strategy": strategy,
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "python": platform.python_version(), "numpy": np.__version__},
        "max_configurations": n_trials,
        "baseline_config": baseline_config,
        "search_space": {"lr_log_uniform": list(search_space.lr), "batch_size": list(search_space.batch_size),
                         "optimizer": list(search_space.optimizer)},
        "runs": records,
        "anytime": {"axis": "attempted_configurations", "curves": curves,
                    "plot": "anytime.svg" if has_plot else None},
        "paired_comparison": _paired_comparison(records, metric, search_method),
    })
    return primary
