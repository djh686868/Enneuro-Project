"""自动调参运行结果的确定性 JSON/JSONL 记录器。

日志器不参与搜索决策，只负责把 Agent 发出的事件收集起来并持久化。
JSON 适合一次性读取完整报告，JSONL 则适合逐行追踪事件、流式处理或
在训练中断后恢复检查。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .protocol import AgentEvent, SearchResult


def _json_default(value: Any) -> Any:
    """把 NumPy 标量/数组转换成标准 JSON 支持的 Python 类型。"""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class EventLogger:
    """收集事件并写出完整运行报告。

    日志器刻意不自动写入墙上时间；相同 seed 的报告因此可以在测试和 CI
    中稳定比较。若执行器需要耗时，应把 elapsed_s 等指标放入 payload。
    """

    def __init__(self) -> None:
        # 事件序号由列表长度生成，保证从 0 开始且不会重复。
        self.events: list[AgentEvent] = []

    def emit(self, event: str, trial_id: int | None = None, **payload: Any) -> AgentEvent:
        """追加一条事件并返回刚创建的对象。"""
        record = AgentEvent(len(self.events), event, trial_id, payload)
        self.events.append(record)
        return record

    def write_jsonl(self, path: str | Path, events: Iterable[AgentEvent] | None = None) -> Path:
        """按一行一个 JSON 对象写出事件流。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = events if events is not None else self.events
        with target.open("w", encoding="utf-8") as handle:
            for event in records:
                handle.write(json.dumps(event.to_dict(), sort_keys=True, default=_json_default) + "\n")
        return target

    def write_json(self, path: str | Path, report: SearchResult | dict[str, Any]) -> Path:
        """写出最终汇总报告；支持 SearchResult 或普通字典。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict() if isinstance(report, SearchResult) else report
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return target

    def write_report(self, directory: str | Path, report: SearchResult) -> tuple[Path, Path]:
        """在目录中同时生成 ``report.json`` 和 ``events.jsonl``。"""
        directory = Path(directory)
        return (
            self.write_json(directory / "report.json", report),
            self.write_jsonl(directory / "events.jsonl", report.events),
        )


# 兼容偏好“报告器”命名的调用方；两者行为完全一致。
JSONReporter = EventLogger
