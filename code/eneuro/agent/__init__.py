"""EnNeuro 基础自动调参 Agent 的公开入口。

本包把一次自动调参任务拆成几个彼此独立的角色：

* ``protocol`` 定义搜索空间、试验结果和事件的数据协议；
* ``search`` 负责按照固定预算驱动“观察—决策—执行—反馈”循环；
* ``executors`` 提供合成目标和 MNIST 小型分类执行器；
* ``car_executor`` 提供从车辆数据目录按需读取图像的回归执行器；
* ``logger`` 将每一步事件写入 JSON/JSONL，便于复盘和二次分析。

这里集中重导出常用类，调用方可以直接 ``from eneuro.agent import ...``，
而不必了解各个实现文件的内部组织。
"""

# 轻量执行器：不依赖 CUDA，适合先验证 Agent 的逻辑闭环。
from .executors import MNISTExecutor, SyntheticObjectiveExecutor, make_tiny_mnist
# 车辆数据执行器按需读取本地 JPEG。
from .car_executor import CarBehavioralCloningExecutor, SteeringDataset, read_steering_rows
# 报告器和协议对象是上层编排代码最常使用的公共 API。
from .logger import EventLogger, JSONReporter
from .protocol import AgentEvent, Executor, SearchResult, SearchSpace, TrialResult
from .search import (SEARCH_STRATEGIES, AutoMLAgent, BayesianOptimizationSearch, EvolutionarySearch,
                     LocalSearch, SequentialRandomSearch, SequentialRandomSearcher, create_search)

__all__ = [
    "AgentEvent", "Executor", "EventLogger", "JSONReporter", "MNISTExecutor",
    "SearchResult", "SearchSpace", "SequentialRandomSearch", "SequentialRandomSearcher",
    "AutoMLAgent", "SyntheticObjectiveExecutor", "TrialResult", "make_tiny_mnist",
    "LocalSearch", "EvolutionarySearch", "BayesianOptimizationSearch", "create_search", "SEARCH_STRATEGIES",
    "CarBehavioralCloningExecutor", "SteeringDataset", "read_steering_rows",
]
