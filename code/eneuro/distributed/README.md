# EnNeuro 同步 PS 训练（M2）

第一版支持单机 CPU、Windows `spawn`、同步 BSP。PS 保存模型与优化器；Worker 只计算梯度。每一步所有 Worker 基于相同版本计算，PS 按样本数加权聚合并调用一次 `optimizer.step()`。参数通过完整名称匹配；传输为长度前缀 TCP 帧、JSON 元数据和 NumPy 二进制数组。

## 运行

在 `code` 目录并激活项目环境后运行：

```powershell
python -m unittest discover -s tests -p test_distributed_ps.py -v
python -m eneuro.distributed.demo --workers 2
python -m eneuro.distributed.benchmark --workers 1 2 4 --repeats 3
python -m eneuro.distributed.benchmark --dataset mnist --samples 1024 --batch-size 64 --workers 1 2 4 --repeats 3 --output metrics/mnist-ps
```

业务调用 `launch(model_factory, dataset_factory, loss_fn, optimizer_factory, config)`。四个可调用对象必须定义在可导入模块的顶层，模型参数必须在构造后初始化。配置由 `DistributedConfig` 提供；`checkpoint_path` 可选，设置后复用框架 `Serializer` 保存模型与优化器状态。指标写入 `metrics_dir` 中每个进程各自的 JSONL。

消息：`REGISTER`、`PULL`、`PUSH`、`UPDATE`、`HEARTBEAT`，以及控制/错误消息。`PUSH` 带 Worker、step、模型版本和样本数；PS 拒绝重复、过期、缺失、形状错误或非有限梯度。同步屏障超时会使作业失败。

## 当前边界

- 等价性测试使用确定性 MLP、CPU、SGD/Adam；BatchNorm 运行统计量与随机层尚未做跨 Worker 同步。
- 最后一批少于 Worker 数且 `drop_last=False` 时显式报错；默认 `drop_last=True`。
- `benchmark` 的内置玩具数据集用于验证统计流程，正式扩展性结论应换成较大真实模型和数据集，在相同机器上重复运行并记录 CPU 线程数。
- `--dataset mnist` 读取仓库 `tests/testdata/MNIST_data/mnist.pkl` 的训练集，使用项目 LeNet 层结构（PS 兼容的初始化版本）、交叉熵和 SGD。`--samples` 决定训练子集大小；报告中必须写明实际数值。保持全局 batch 大小不变，Worker 数只改变每个 Worker 的局部 batch。
- `benchmark` 输出 `results.csv`，环境安装 matplotlib 时还会生成吞吐和通信占比曲线图。
- `train_samples_per_second` 使用首个训练 step 开始至最后一个 step 结束的时间和实际处理样本数；`job_samples_per_second` 另含进程启动、注册与结束时间。`communication_ratio` 为 Worker PUSH 往返（含屏障等待）总耗时除以 Worker step 总耗时。
- 异步 PS（M2-10）尚未实现。
