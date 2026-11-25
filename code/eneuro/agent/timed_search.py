"""供 MNIST 和车辆演示共享的限时搜索、实时日志和外层评估。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from math import isfinite
from pathlib import Path
from typing import TextIO

from ..profiling import ProfileSession
from ._profiling import Profiling, profile_phase, profile_reference, profile_run, profile_window
from .evaluation import _search_method, plot_anytime
from .logger import EventLogger
from .protocol import AgentEvent, SearchSpace, TrialResult
from .search import create_search


class _ConsoleLogger(EventLogger):
    """实时输出决策与反馈，同时将完整搜索事件写入 JSONL。"""

    def __init__(self, stream: TextIO, start: float, budget_s: float,
                 format_validation: Callable[[float], str]):
        super().__init__()
        self.stream, self.start, self.budget_s = stream, start, budget_s
        self.format_validation = format_validation

    def emit(self, event, trial_id=None, **payload):
        elapsed = time.perf_counter() - self.start
        record = super().emit(event, trial_id, elapsed_s=elapsed, **payload)
        self.stream.write(json.dumps(record.to_dict(), allow_nan=False) + "\n")
        self.stream.flush()
        if event == "decide":
            config = payload["config"]
            print(f"[试验 {trial_id + 1}] 开始 | {config['optimizer']} "
                  f"lr={config['lr']:.4g} batch={config['batch_size']} | "
                  f"剩余 {max(0, self.budget_s - elapsed):.1f}s", flush=True)
        elif event == "feedback":
            result = payload["result"]
            if result["status"] == "completed":
                score = self.format_validation(result["objective"])
                best = self.format_validation(payload["best_objective"])
                print(f"[试验 {trial_id + 1}] 完成 | 验证{score} | 历史最佳{best} | "
                      f"搜索已用 {elapsed:.1f}s", flush=True)
            else:
                print(f"[试验 {trial_id + 1}] {result['status']} | {result['error']}", flush=True)
        return record


def _curve(events: Sequence[AgentEvent], search_elapsed: float, attempts: int) -> list[dict]:
    """只在完成验证且分数提高时增加点，在搜索结束处延长最后一段。"""
    points = []
    best = None
    for event in events:
        if event.event != "feedback":
            continue
        value = event.payload["best_objective"]
        if value is None or (best is not None and value >= best):
            continue
        best = value
        elapsed = event.payload["result"]["metrics"]["completion_elapsed_s"]
        points.append({"elapsed_s": elapsed, "evaluations": event.trial_id + 1,
                       "mean": value, "se": None, "n": 1})
    if points:
        points.append({**points[-1], "elapsed_s": search_elapsed, "evaluations": attempts})
    return points


def _show_epoch(label: str, start: float, format_validation: Callable[[float], str],
                deadline: float | None = None, *, profile: ProfileSession | None = None):
    def show(epoch: int, loss: float, validation: float | None) -> None:
        now = time.perf_counter()
        if profile is not None:
            profile.mark("automl.epoch_end", phase=label, epoch=epoch, loss=loss,
                         validation=validation, elapsed_s=now - start)
        score = "" if validation is None else f" | 验证{format_validation(validation)}"
        remaining = "" if deadline is None else f" | 剩余 {max(0, deadline - now):.1f}s"
        print(f"  [{label}] epoch {epoch} | 训练损失 {loss:.4f}{score} | "
              f"已用 {now - start:.1f}s{remaining}", flush=True)
    return show


def run_timed_search(executor, space: SearchSpace, *, budget_s: float, trials: int | None,
                     seed: int, output: Path, protocol: dict, plot_ylabel: str,
                     format_validation: Callable[[float], str],
                     format_epoch_validation: Callable[[float], str],
                     format_test: Callable[[dict], str], strategy: str = "random",
                     profiling: Profiling = None) -> int:
    """执行限时搜索，记录右连续阶梯曲线并独立重训获胜配置。"""
    if not isfinite(budget_s) or budget_s <= 0:
        raise ValueError("--budget-minutes 必须是正数")
    if trials is not None and trials < 1:
        raise ValueError("--trials 必须是正整数")
    search_method = _search_method(strategy)

    # 在计时前检查绘图依赖，运行后始终使用无窗口的 SVG 后端。
    import matplotlib
    from matplotlib.backends import backend_svg  # noqa: F401

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"搜索策略 {strategy} | 搜索预算 {budget_s:.1f}s | 输出目录 {output}", flush=True)

    with profile_run(
        profiling, name="automl_timed_search", output_dir=output / "profiling",
        seed=seed, strategy=strategy, protocol=protocol,
    ) as (profile, owned):
        status, report = _run_timed_search(
            executor, space, budget_s=budget_s, trials=trials, seed=seed, output=output,
            protocol={**protocol, "matplotlib": matplotlib.__version__},
            plot_ylabel=plot_ylabel, format_validation=format_validation,
            format_epoch_validation=format_epoch_validation, format_test=format_test,
            strategy=strategy, search_method=search_method, profile=profile,
        )
    if owned:
        report["profiling"] = profile_reference(profile, completed=True)
        _write_report(output / "report.json", report)
    return status


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _run_timed_search(executor, space, *, budget_s, trials, seed, output, protocol,
                      plot_ylabel, format_validation, format_epoch_validation, format_test,
                      strategy, search_method, profile):
    interrupted = fatal_error = False
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        start = time.perf_counter()
        deadline = start + budget_s
        logger = _ConsoleLogger(stream, start, budget_s, format_validation)
        on_epoch = _show_epoch("搜索", start, format_epoch_validation, deadline, profile=profile)

        def execute(config) -> TrialResult:
            nonlocal fatal_error
            try:
                result = executor.execute(config, deadline=deadline, on_epoch=on_epoch)
            except Exception:
                # 数值发散由执行器返回失败试验，未处理的异常终止后续搜索。
                fatal_error = True
                raise
            completed = time.perf_counter()
            if result.status == "completed":
                if completed > deadline:
                    return TrialResult(float("inf"), status="timed_out", error="验证完成时已超过搜索预算")
                result.metrics["completion_elapsed_s"] = completed - start
            return result

        agent = create_search(strategy, execute, space, n_trials=trials, seed=seed, logger=logger)
        try:
            search = agent.run(should_stop=lambda: time.perf_counter() >= deadline or fatal_error)
        except KeyboardInterrupt:
            interrupted = True
            search = agent.result()
        search_elapsed = time.perf_counter() - start

    if interrupted:
        stop_reason = "interrupted"
    elif fatal_error:
        stop_reason = "trial_failed"
    elif search_elapsed >= budget_s:
        stop_reason = "time_budget"
    else:
        stop_reason = "trial_limit"

    points = _curve(search.events, search_elapsed, len(search.trials))
    plot_path = output / "anytime.svg"
    has_plot = bool(points) and plot_anytime(
        plot_path, {search_method: points}, plot_ylabel, 1, x_key="elapsed_s",
    )
    report = {
        **search.to_dict(),
        "strategy": strategy,
        "protocol": protocol,
        "search": {"budget_s": budget_s, "elapsed_s": search_elapsed,
                   "max_trials": trials, "stop_reason": stop_reason},
        "anytime": {"axis": "elapsed_s", "points": points, "plot": plot_path.name if has_plot else None},
        "outer_test": {"status": "skipped"},
    }
    report_path = output / "report.json"

    _write_report(report_path, report)
    print(f"[搜索结束] {stop_reason} | {len(search.trials)} 次试验 | 用时 {search_elapsed:.1f}s", flush=True)

    if search.best_config is not None and not interrupted and stop_reason != "trial_failed":
        print(f"[外层测试] 最佳配置 {search.best_config} | 用训练+验证数据重训，单独计时", flush=True)
        refit_start = time.perf_counter()
        profile_start = profile.offset_ns if profile is not None else None
        try:
            with profile_phase(profile, "automl.refit", seed=seed, strategy=strategy,
                               config=search.best_config):
                metrics = executor.evaluate_final(
                    search.best_config,
                    on_epoch=_show_epoch("重训", refit_start, format_epoch_validation, profile=profile),
                    checkpoint=output / "best_checkpoint.bin",
                )
                # 保持失败时的报告可读，拒绝会导致 JSON 无效的外层指标。
                json.dumps(metrics, allow_nan=False)
            report["outer_test"] = {"status": "completed", **metrics}
            print(f"[外层测试] 测试{format_test(metrics)}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            report["outer_test"] = {"status": "interrupted"}
        except Exception as exc:
            report["outer_test"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[外层测试] {report['outer_test']['error']}", flush=True)
        report["outer_test"]["elapsed_s"] = time.perf_counter() - refit_start
        if profile is not None:
            report["outer_test"]["profiling"] = profile_window(profile, profile_start)
        _write_report(report_path, report)
    print(f"报告 {report_path}", flush=True)
    if has_plot:
        print(f"Anytime Performance Curve {plot_path}", flush=True)
    if interrupted:
        return 130, report
    return (0 if has_plot and report["outer_test"]["status"] == "completed" else 1), report
