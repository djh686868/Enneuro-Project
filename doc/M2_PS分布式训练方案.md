# M2 单机多进程 PS 分布式训练方案

## 1. 项目目标

本阶段计划在 EnNeuro 框架中增加单机多进程 Parameter Server（PS）分布式训练能力。系统由 ParameterServer、Worker、Launcher 和 Monitor 四部分组成，用多个 Worker 并行计算梯度，由 ParameterServer 统一聚合梯度并更新全局模型。

第一版只实现同步 BSP 模式，优先保证训练结果正确、过程可复现，并能够统计训练与通信开销。异步 PS 作为后续选做功能。

## 2. 总体架构

```text
Launcher
  ├─ ParameterServer：保存全局参数、聚合梯度、执行优化器更新
  ├─ Worker 0：读取数据分片、前向传播、反向传播、上传梯度
  ├─ Worker 1：读取数据分片、前向传播、反向传播、上传梯度
  └─ Monitor：记录计算、通信和等待耗时
```

一次同步训练 step 的流程如下：

1. Worker 向 PS 注册并拉取同一版本的模型参数。
2. 各 Worker 使用自己的数据分片进行前向和反向传播。
3. Worker 上传梯度、样本数、step 编号和参数版本号。
4. PS 等待本 step 的所有 Worker，按样本数加权平均梯度。
5. PS 复用框架现有 Optimizer，统一执行一次参数更新。
6. PS 将模型版本号加一，并把新参数返回给 Worker。

Worker 只负责计算梯度，不在本地调用 `optimizer.step()`；优化器及其 Momentum、Adam 等状态只保存在 PS 中。

## 3. 组件设计

### 3.1 ParameterServer

ParameterServer 保存唯一的全局模型和优化器，维护 `model_version`、`step_id`、已注册 Worker 以及当前 step 收到的梯度。

同步模式下，PS 收齐所有 Worker 的梯度后再更新。若各 Worker 实际样本数不同，采用以下方式聚合：

```text
聚合梯度 = Σ(Worker 梯度 × Worker 样本数) / 总样本数
```

PS 还需要检查重复上传、参数名和形状不匹配、旧版本梯度以及 Worker 超时，避免错误梯度进入模型。

### 3.2 Worker

Worker 保存模型副本和自己的数据分片。每个训练 step 执行：加载本地 batch、前向传播、计算 loss、反向传播、提取梯度、上传 PS、接收并加载新参数。

现有 Trainer 需要拆出单 batch 接口，使单进程训练和 Worker 复用相同的前向、标签处理、loss 和反向逻辑。建议提供 `train_batch(..., update=True)`；Worker 使用 `update=False`，只产生梯度而不更新本地参数。

### 3.3 Launcher

Launcher 负责创建和回收 PS、Worker 进程，启动顺序为：启动 PS、等待服务就绪、启动 Worker、等待训练结束、保存最终模型、关闭所有进程。

Windows 统一使用 `multiprocessing.get_context("spawn")`。所有子进程入口必须定义在模块顶层，并通过 `if __name__ == "__main__"` 启动，避免重复创建进程。第一版 Worker 内部不再开启额外的数据加载子进程。

### 3.4 Monitor

Monitor 使用 `time.perf_counter_ns()` 记录前向、反向、PUSH、PULL、屏障等待和完整 step 耗时，同时记录收发字节数。每个进程写入独立 JSONL 文件，训练结束后统一汇总，避免多进程同时写一个文件。

主要指标包括吞吐量、平均 step 耗时、通信耗时占比和屏障等待时间。

## 4. 通信协议

第一版使用本机 TCP 通信。消息采用“固定长度帧头 + JSON 元数据 + 数组负载”的结构，避免 TCP 拆包导致接收内容不完整。

主要消息如下：

| 消息 | 作用 |
| --- | --- |
| `REGISTER` | Worker 注册身份、Worker 总数和参数清单 |
| `PULL` | Worker 请求当前模型参数 |
| `PUSH` | Worker 上传梯度、样本数、step 和参数版本 |
| `UPDATE` | PS 返回更新后的模型参数和新版本号 |
| `HEARTBEAT` | Worker 定期报告存活状态和当前进度 |

公共字段包括 `protocol_version`、`request_id`、`worker_id`、`step_id`、`model_version` 和时间戳。协议版本与模型版本分开管理：前者表示消息格式版本，后者表示模型已经更新的次数。

开发初期可以用 JSON 数组完成正确性验证；性能实验前将 NumPy 数组改为二进制负载，减少 JSON 编解码和数据膨胀带来的影响。

## 5. 数据划分与同步策略

每个 epoch 根据 `seed + epoch` 生成一次可复现的全局随机排列，再按 Worker rank 划分：

```python
worker_indices = shuffled_indices[rank::world_size]
```

划分器需要保证各分片没有重叠、所有分片的并集等于完整数据集，并且分片大小差不超过 1。

同步 BSP 模式要求同一 step 的所有 Worker 使用相同的 `model_version`。PS 收齐所有 Worker 的梯度之前不更新参数，更新完成前 Worker 不能进入下一 step。若某个 Worker 超时，则本次训练失败并由 Launcher 统一结束其他进程，避免无限等待。

## 6. 正确性测试

分布式正确性的核心标准是：相同随机种子、相同初始模型和相同全局 batch 下，单进程训练与 PS 两 Worker 训练的 loss、模型参数和优化器状态应在浮点误差范围内一致。

第一版测试使用 CPU、简单 MLP、固定数据顺序和 SGD，不使用 Dropout、BatchNorm、随机数据增强或 GPU。测试顺序为：

1. 单步梯度聚合等价测试。
2. 两 Worker 多步 SGD 参数等价测试。
3. MomentumSGD、Adam 优化器状态等价测试。
4. 完整多进程和 TCP 集成测试。

等价性测试通过后，再进行 Worker 数为 1、2、4 的吞吐量和通信占比实验。

## 7. 当前已完成的新增工作

目前已完成本阶段的前期设计和代码现状调研，具体包括：

1. 完成 PS、Worker、Launcher、Monitor 四组件的职责划分和同步训练主流程设计。
2. 完成 REGISTER、PULL、PUSH、UPDATE、HEARTBEAT 五类消息的协议草案，并明确协议版本、模型版本和 step 编号的区别。
3. 确定同步 BSP 作为第一版实现方案，明确按样本数加权聚合梯度和版本校验规则。
4. 完成数据分片算法及“无重叠、并集完整、结果可复现”的验收标准设计。
5. 完成收敛等价性测试方案，确定以单进程 global batch 对比两 Worker local batch 作为正确性金标准。
6. 完成现有代码依赖分析，发现并记录三个需要优先处理的问题：Optimizer 状态使用类变量、模型参数遍历顺序不稳定、Trainer 缺少可供 Worker 复用的单 batch 接口。
7. 完成 Windows `spawn` 启动约束分析，明确子进程顶层入口、主入口保护和进程生命周期管理要求。

以上工作属于本次 M2 新需求的设计与准备工作，不包含旧模块功能。

## 8. 后续实施计划

1. 修复 Optimizer 独立状态和稳定参数命名，重构 Trainer 单 batch 接口。
2. 实现数据划分器、协议编解码和通信计时封装。
3. 实现同步 ParameterServer 和单 Worker 闭环。
4. 接入两个 Worker、同步屏障和 Windows Launcher。
5. 完成收敛等价性测试，再进行 1、2、4 Worker 扩展性实验。
6. 时间允许时增加异步 PS，并记录梯度 staleness 对收敛的影响。

