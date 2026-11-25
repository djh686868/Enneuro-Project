# EnNeuro 单机多 Worker 训练

框架在单机 CPU 上使用 `spawn` 启动独立进程。每一步先将同一个全局 batch 分给各个 Worker，再按实际样本数汇总局部平均梯度。两种后端共用数据划分、损失函数和训练入口。

| `backend` | 每一步怎样更新参数 |
| --- | --- |
| `ps`（默认） | Worker 向参数服务器提交梯度。服务器聚合、执行优化器更新并返回参数。 |
| `ring` | Worker 与环上的相邻进程通过 TCP 交换梯度分块。先用 reduce-scatter 对各块求和，再用 allgather 收齐所有分块。每个 Worker 用相同的全局平均梯度更新自己的模型和优化器。 |

## 从其他模块调用

下面的示例可以保存为 `code` 下的脚本运行。将四个可调用对象替换为业务模块自己的模型工厂、数据集工厂、损失函数和优化器工厂即可。省略 `backend` 时沿用 PS。

```python
from pathlib import Path

from eneuro.distributed.demo import make_dataset, make_model, make_optimizer
from eneuro.nn.loss import meanSquaredError
from eneuro.train import DistributedConfig, launch_distributed

if __name__ == "__main__":
    config = DistributedConfig(backend="ring", world_size=2, global_batch_size=4,
                               epochs=2, metrics_dir=Path("metrics/ring-demo"))
    result = launch_distributed(make_model, make_dataset, meanSquaredError,
                                make_optimizer, config)
    print(result["model_version"])
```

`eneuro.distributed.launch` 与 `eneuro.train.launch_distributed` 是同一入口。所有工厂及损失函数都应可由子进程导入，可以用 `functools.partial` 绑定参数。模型工厂须在返回前初始化所有参数。每个 Worker 的数据集须有相同长度和索引含义。损失函数须返回局部 batch 的样本平均损失。

两个后端都返回 `model_version`（已完成的更新步数）、`parameters`（按名称索引的 NumPy 数组）和 `optimizer_state`。配置 `checkpoint_path` 可保存框架可读取的检查点。Worker 的指标位于 `metrics_dir/worker-<rank>.jsonl`。同时启动的作业应使用各自的指标目录和检查点路径。Ring 自动使用本机回环地址和临时端口，PS 设置 `port=0` 可让系统自动分配监听端口。

## 训练性能采集

在 `launch_distributed(...)` 传入 `profiling=True` 或 `profiling={"output_dir": "metrics/profiles"}`。通过 `result["profiling"]["report"]["observations"]["distributed"]["metrics"]` 读取每个 Worker 和 PS 的阶段耗时、收发字节和样本数。PS 的 `push` 阶段包含 `barrier_wait_ms`，该字段汇总 Worker 等待同步的时间。进程 CPU 和 RSS 指标属于启动进程，系统和 GPU 指标属于整台机器。传入 `profiling=False` 可禁用统一采集，后端继续输出原有的 Worker 和 PS JSONL。

## 运行

在 `code` 目录激活项目环境后运行。

```powershell
python -m unittest discover -s tests -p test_distributed_ps.py -v
python -m pytest -q tests/test_distributed_ring.py
python -m eneuro.distributed.demo --workers 2
python -m eneuro.distributed.demo --backend ring --workers 2
python -m eneuro.distributed.benchmark --dataset mnist --samples 1024 --batch-size 64 --workers 1 2 4 --backends ps ring --repeats 3 --output metrics/mnist-ps-ring
python -m eneuro.distributed.equivalence --backend ring --samples 128 --batch-size 32
python -m unittest discover -s tests -p test_distributed_autodrive.py -v
python -m eneuro.distributed.benchmark --dataset autodrive --samples 512 --batch-size 16 --workers 1 2 4 --repeats 3 --output metrics/autodrive-ps-512
python -m eneuro.distributed.autodrive_equivalence --backend ring --samples 8 --batch-size 4 --data-dir ../tests/testdata/data
```

## 当前边界

MNIST 上 1、2、4 个 Worker 的训练吞吐量、端到端耗时、逐步 loss 和最终参数的对照见[性能测试报告](../../../doc/ring_allreduce_benchmark.md)。

- 训练支持 CPU。需要与单进程逐步比较时，应选用确定性模型。BatchNorm 运行统计量和随机层状态尚未做跨 Worker 同步。
- 最后一批少于 Worker 数且 `drop_last=False` 时显式报错；默认 `drop_last=True`。
- `benchmark` 的内置玩具数据集用于验证统计流程，正式扩展性结论应换成较大真实模型和数据集，在相同机器上重复运行并记录 CPU 线程数。
- `--dataset mnist` 默认读取 `tests/testdata/MNIST_data/mnist.pkl` 的训练集。多个工作目录共用数据时，可将环境变量 `ENNEURO_MNIST_DATA` 设为 `mnist.pkl` 的绝对路径，也可给 benchmark 传入 `--mnist-path`。测试使用 LeNet、交叉熵和 SGD。`--samples` 控制子集大小。全局 batch 固定，Worker 数决定局部 batch 大小。
- `benchmark` 输出 `results.csv`，环境安装 matplotlib 时还会生成吞吐和通信占比曲线图。
- `equivalence` 用真实 MNIST 的每一步全局 batch loss 和最终参数比较单进程与所选后端的双 Worker 结果。`--output` 指定逐步 CSV 和折线图的位置，绘图需要 matplotlib。
- `--dataset autodrive` 从 `tests/testdata/data` 下的 `driving_log.csv` 和 `IMG` 读取中心相机图像，按 CSV 行等间距抽样并缩放到 160×120。模型含五层卷积和全连接回归头，用 MSE 预测转向角。数据目录由本机提供。运行 `test_distributed_autodrive.py` 时可以将环境变量 `ENNEURO_AUTODRIVE_DATA` 设为包含 CSV 和图像目录的路径。
- `autodrive_equivalence` 用同一批小车图像逐步对比单进程与所选后端的双 Worker 全局 batch MSE，并检查最终模型参数。图像在其他目录时传入 `--data-dir`。
- `train_samples_per_second` 使用首个训练 step 开始至最后一个 step 结束的时间和实际处理样本数。`job_samples_per_second` 另含进程启动、初始化和结束时间。`communication_ratio` 为各 Worker 每步 PUSH 往返或 Allreduce 的累计耗时除以 Worker step 累计耗时，包含等待时间。
