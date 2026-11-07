"""自动调参 Agent 使用的轻量数据协议和结果对象。

本文件只描述“数据长什么样”，不负责真正的搜索或训练。把这些对象
单独放在协议层，可以让搜索策略、MNIST 执行器、车辆执行器和测试代码
通过同一个接口协作，也能让最终报告保持稳定的 JSON 结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SearchSpace:
    """描述待搜索的超参数空间。

    ``lr`` 不是在线性区间采样，而是在 ``log10(lr)`` 上均匀采样。这样做
    可以公平覆盖 ``1e-4``、``1e-3``、``1e-2`` 等不同数量级；batch size
    和 optimizer 则保持离散候选值，便于记录、比较和复现实验。

    Attributes:
        lr: 学习率的正数区间 ``(lower, upper)``。
        batch_size: 可选批大小集合。
        optimizer: 可选优化器名称集合。
    """

    lr: tuple[float, float] = (1e-4, 1e-1)
    batch_size: tuple[int, ...] = (16, 32, 64)
    optimizer: tuple[str, ...] = ("sgd", "adam")

    def __post_init__(self) -> None:
        # 在任务启动前尽早发现非法空间，避免跑了几个 trial 才失败。
        lo, hi = self.lr
        if lo <= 0 or hi <= lo:
            raise ValueError("lr must be a positive (low, high) interval")
        if not self.batch_size or any(int(v) <= 0 for v in self.batch_size):
            raise ValueError("batch_size must contain positive integers")
        if not self.optimizer or any(not str(v) for v in self.optimizer):
            raise ValueError("optimizer must contain at least one name")

    def sample(self, rng: Any) -> dict[str, Any]:
        """从空间采样一个可 JSON 序列化的配置。

        ``rng`` 由上层 Agent 持有，因此只要 seed 相同，配置序列就相同。
        """
        import numpy as np

        return {
            "lr": float(10.0 ** rng.uniform(np.log10(self.lr[0]), np.log10(self.lr[1]))),
            "batch_size": int(rng.choice(self.batch_size)),
            "optimizer": str(rng.choice(self.optimizer)),
        }


@dataclass
class TrialResult:
    """一次试验的标准化结果。

    当前搜索器按“目标值越小越好”处理 ``objective``，因此分类准确率
    通常会转换成 ``1 - accuracy``。``metrics`` 保留更多可解释指标，
    例如 accuracy、MSE、MAE 和运行时间；失败 trial 则用 ``status`` 和
    ``error`` 记录原因，而不是让整个搜索任务立即中断。
    """

    objective: float
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换成报告层可直接序列化的普通字典。"""
        return {
            "objective": float(self.objective),
            "metrics": dict(self.metrics),
            "status": self.status,
            "error": self.error,
        }


class Executor(Protocol):
    def execute(self, config: Mapping[str, Any]) -> TrialResult | Mapping[str, Any]:
        """执行一个配置并返回 ``TrialResult`` 或等价字典。

        搜索器只依赖这个最小接口，因此可以接入真实训练、合成目标、
        外部命令或远程任务。执行器内部负责把训练细节压缩成一个目标值
        和一组附加指标。
        """


@dataclass(frozen=True)
class AgentEvent:
    """闭环中的一条事件记录。

    ``index`` 是事件在本次运行中的递增序号；``trial_id`` 指向关联的
    trial；``payload`` 存放本事件的具体内容。这里不自动写墙上时间，
    是为了让相同 seed 的测试报告可以稳定比较。
    """

    index: int
    event: str
    trial_id: int | None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把不可变事件对象转换成普通字典。"""
        return {
            "index": self.index,
            "event": self.event,
            "trial_id": self.trial_id,
            "payload": dict(self.payload),
        }


@dataclass
class SearchResult:
    """一次完整搜索的汇总结果。

    ``trials`` 保存每个配置及其结果，``events`` 保存完整时序，
    ``best_config``/``best_result`` 是最终 incumbent，``seed`` 用于追溯
    随机性来源。该对象既可供 Python 调用方读取，也可直接交给
    ``EventLogger.write_json`` 写入最终报告。
    """

    trials: list[dict[str, Any]]
    events: list[AgentEvent]
    best_config: dict[str, Any] | None
    best_result: TrialResult | None
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """将汇总结果展开为 JSON 兼容结构。"""
        return {
            "seed": self.seed,
            "trials": self.trials,
            "events": [event.to_dict() for event in self.events],
            "best_config": self.best_config,
            "best_result": self.best_result.to_dict() if self.best_result else None,
        }
