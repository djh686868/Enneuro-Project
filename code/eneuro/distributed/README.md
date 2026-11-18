# EnNeuro 同步 PS 训练（M2）

第一版支持单机 CPU、Windows `spawn`、同步 BSP。PS 保存模型与优化器；Worker 只计算梯度。每一步所有 Worker 基于相同版本计算，PS 按样本数加权聚合并调用一次 `optimizer.step()`。参数通过完整名称匹配；传输为长度前缀 TCP 帧、JSON 元数据和 NumPy 二进制数组。

## 运行

在 `code` 目录并激活项目环境后运行：

```powershell
python -m unittest discover -s tests -p test_distributed_ps.py -v
python -m eneuro.distributed.demo --workers 2
python -m eneuro.distributed.benchmark --workers 1 2 4 --repeats 3
python -m eneuro.distributed.benchmark --dataset mnist --samples 1024 --batch-size 64 --workers 1 2 4 --repeats 3 --output metrics/mnist-ps
python -m eneuro.distributed.equivalence --samples 4096 --batch-size 64
python -m unittest discover -s tests -p test_distributed_autodrive.py -v
python -m eneuro.distributed.benchmark --dataset autodrive --samples 512 --batch-size 16 --workers 1 2 4 --repeats 3 --output metrics/autodrive-ps-512
python -m eneuro.distributed.autodrive_equivalence --samples 512 --batch-size 16
```

业务调用 `launch(model_factory, dataset_factory, loss_fn, optimizer_factory, config)`。四个可调用对象必须定义在可导入模块的顶层，模型参数必须在构造后初始化。配置由 `DistributedConfig` 提供；`checkpoint_path` 可选，设置后复用框架 `Serializer` 保存模型与优化器状态。指标写入 `metrics_dir` 中每个进程各自的 JSONL。

消息：`REGISTER`、`PULL`、`PUSH`、`UPDATE`、`HEARTBEAT`，以及控制/错误消息。`PUSH` 带 Worker、step、模型版本和样本数；PS 拒绝重复、过期、缺失、形状错误或非有限梯度。同步屏障超时会使作业失败。

## 当前边界

- 等价性测试使用确定性 MLP、CPU、SGD/Adam；BatchNorm 运行统计量与随机层尚未做跨 Worker 同步。
- 最后一批少于 Worker 数且 `drop_last=False` 时显式报错；默认 `drop_last=True`。
- `benchmark` 的内置玩具数据集用于验证统计流程，正式扩展性结论应换成较大真实模型和数据集，在相同机器上重复运行并记录 CPU 线程数。
- `--dataset mnist` 读取仓库 `tests/testdata/MNIST_data/mnist.pkl` 的训练集，使用项目 LeNet 层结构（PS 兼容的初始化版本）、交叉熵和 SGD。`--samples` 决定训练子集大小；报告中必须写明实际数值。保持全局 batch 大小不变，Worker 数只改变每个 Worker 的局部 batch。
- `benchmark` 输出 `results.csv`，环境安装 matplotlib 时还会生成吞吐和通信占比曲线图。
- `equivalence` 用真实 MNIST 的每一步全局 batch loss 和最终参数比较单进程与同步 PS 双 Worker；通过后在 `doc/M2_MNIST_PS结果` 输出逐步 CSV 和折线图，需要 matplotlib。
- `--dataset autodrive` 读取仓库本机 `tests/testdata/data/driving_log.csv` 和 `IMG` 目录；按 CSV 行等间距选取指定数量的中心相机图像，缩放到 160x120，以转向角做 MSE 回归。模型是参考 `serving.AutoDriveNet` 的 PS 兼容变体：保留五层卷积和全连接回归头，去掉未同步的 BatchNorm，全部参数预初始化。数据目录目前被 `.gitignore` 排除，复现者需自行准备相同数据。输出原始结果、按全局 batch 聚合的 loss 曲线、吞吐量柱状图加通信占比折线图。
- `autodrive_equivalence` 用同一 512 张小车图像逐步对比单进程与双 Worker 同步 PS 的全局 batch MSE，并检查最终模型参数；通过后在 `doc/M2_AutoDrive_PS结果` 输出每一步的 CSV 和含绝对差的折线图。
- `train_samples_per_second` 使用首个训练 step 开始至最后一个 step 结束的时间和实际处理样本数；`job_samples_per_second` 另含进程启动、注册与结束时间。`communication_ratio` 为 Worker PUSH 往返（含屏障等待）总耗时除以 Worker step 总耗时。
- 异步 PS（M2-10）尚未实现。
