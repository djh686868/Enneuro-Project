# EnNeuro AutoML Agent 接口

搜索器选择配置，训练执行器用该配置训练并返回验证结果。同一个执行器可以直接交给四种搜索器。命令行示例和数据准备见 [Agent 使用指南](automl_agent_guide.md)。

## 搜索策略

| `strategy` | 从 `eneuro.agent` 导入的类 | 如何提出下一组配置 |
| --- | --- | --- |
| `random` | `SequentialRandomSearch` | 在学习率的对数区间上均匀采样，从候选列表抽取批大小和优化器。 |
| `local` | `LocalSearch` | 从随机配置开始。在当前最佳成功配置上只改一项，候选可以是小幅调整的学习率、相邻批大小或另一种优化器。 |
| `evolutionary` | `EvolutionarySearch` | 第一代随机评估 6 组。后续每代从上一代表现最好的最多 3 个成功配置中选两个父代，逐项交叉，再变异一项。 |
| `beyesian` | `BayesianOptimizationSearch` | 先用随机搜索收集 5 个成功结果，再以高斯过程拟合验证目标，从随机采样的候选中选取期望改进最大的配置。 |

局部搜索在没有成功记录或邻域尝试用尽时回到全局随机采样。进化搜索在可用父代少于 2 个或子代尝试用尽时也回到随机采样。BO 使用归一化的对数学习率、独热编码的离散参数和 RBF 核，拟合时保留最近 127 个成功结果及历史最佳，最多用 128 个候选计算期望改进。BO 的候选耗尽或矩阵分解失败时回到随机采样。四种策略均使用项目已有的 NumPy 依赖。

## 创建和运行搜索器

```python
create_search(strategy, executor, search_space=None, *, n_trials=8, seed=0, logger=None)
```

- `strategy` 使用表中的标识。可从 `eneuro.agent` 导入 `SEARCH_STRATEGIES` 获取全部标识。无效标识会触发 `ValueError`。
- 四个类也可直接构造。构造参数和工厂一致，省略 `strategy` 即可。`AutoMLAgent` 与 `SequentialRandomSearcher` 仍指向随机搜索。
- `search_space` 默认使用 `SearchSpace()`。`lr` 是正学习率区间，`batch_size` 和 `optimizer` 是非空候选序列。优化器名称需要与所用执行器一致。
- `n_trials` 是尝试次数上限，失败尝试也计入。设为 `None` 时，调用 `run(should_stop=...)` 提供停止条件。该回调在启动下一次试验前检查。
- `seed` 控制搜索器的随机数。要复现实验，还需固定数据划分和执行器的训练随机种子。每次独立运行创建一个新搜索器。
- `logger` 可传入 `EventLogger` 实例。默认每个搜索器自行创建记录器。

```python
from eneuro.agent import SearchSpace, SyntheticObjectiveExecutor, create_search

space = SearchSpace(lr=(1e-4, 1e-1), batch_size=(16, 32, 64), optimizer=("sgd", "adam"))
for strategy in ("random", "local", "evolutionary", "beyesian"):
    search = create_search(strategy, SyntheticObjectiveExecutor(), space, n_trials=12, seed=42)
    result = search.run()
    print(strategy, result.best_config)
    search.logger.write_report(f"artifacts/api_example_{strategy}", result)
```

每次试验记录 `observe → decide → execute → feedback` 四个事件。`run()` 返回 `SearchResult`，其中 `trials` 与 `events` 可用于复盘，`best_config` 与 `best_result` 是成功试验中目标值最低的结果。没有成功试验时，这两项都是 `None`。`result.to_dict()` 可转换为报告数据。

## 接入新数据集或模型

执行器实现 `execute(config)` 即可。也可以传入一个签名相同的函数。配置含 `lr`、`batch_size` 和 `optimizer`，执行器返回 `TrialResult` 或含 `objective` 的映射。`objective` 必须是有限的数值，值越小越好。分类可用 `1 - accuracy`，回归可用验证 MSE。

```python
from eneuro.agent import TrialResult, create_search

def run_trial(config):
    # train_and_validate 由数据集启动脚本提供，只读取训练集和验证集
    metrics = train_and_validate(config)
    return TrialResult(objective=1.0 - metrics["accuracy"], metrics=metrics)

result = create_search("beyesian", run_trial, n_trials=12, seed=42).run()
```

训练异常会被记录为失败试验。执行器也可返回 `TrialResult` 并设置 `status` 与 `error`。只有 `status="completed"` 且目标值有限的结果可用于排名。把测试集留给选参后的最终重训和评估。项目的限时 CUDA 入口还让执行器接收 `deadline` 和进度回调，以便在训练过程中检查截止时间。
