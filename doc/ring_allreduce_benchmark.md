# MNIST 上 Ring Allreduce 与参数服务器的性能对比

在同一台 CPU 主机上训练 LeNet，使用 2 个 Worker 时，Ring 的平均吞吐量比同步参数服务器（PS）高 11.8%。使用 4 个 Worker 时高 21.7%，端到端耗时从 4.98 秒降至 2.81 秒。Ring 在 2 和 4 个 Worker 下的吞吐量样本标准差分别为均值的 12.7% 和 9.2%。

| Worker 进程数 | PS 训练吞吐量（样本/秒） | Ring 训练吞吐量（样本/秒） | Ring / PS | PS 作业耗时（秒） | Ring 作业耗时（秒） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,239 ± 96 | 2,469 ± 162 | 1.10× | 5.55 ± 0.32 | 3.45 ± 0.20 |
| 2 | 3,348 ± 226 | 3,742 ± 474 | 1.12× | 5.01 ± 0.12 | 3.07 ± 0.12 |
| 4 | 4,689 ± 318 | 5,705 ± 528 | 1.22× | 4.98 ± 0.25 | 2.81 ± 0.05 |

表中为三次独立作业的均值 ± 样本标准差。每次用 4,096 除以首个 Worker 训练步开始到最后一个训练步结束的墙钟时间，得到训练吞吐量。作业计时从调用 `launch` 到返回，包含 Windows 创建进程、加载数据、初始化、训练、收集结果和退出。1 Worker 的 PS 仍另起一个参数服务器进程，1 Worker 的 Ring 在本地更新参数，无跨进程梯度交换。[逐次计时和样本数](../artifacts/ring_allreduce_benchmark/results.csv)

## 更新结果

按相同 Worker 数与重复序号配成 9 对。每个作业执行 64 次 SGD 更新，每一步按 Worker 实际样本数求全局 batch 的加权平均交叉熵。两种后端逐步 loss 的最大绝对差和最后 44,426 个参数值的最大绝对差在全部配对中均为 0。[逐对误差](../artifacts/ring_allreduce_benchmark/equivalence.csv)

## 测量条件

- 表中的吞吐量和耗时来自迁移前在 `cuda` 工作区运行的基准。迁移后在 `agent` 工作区重新验证了分布式训练正确性。
- 2026-11-21，Windows 11，Intel Core i7-12700KF，20 个逻辑处理器。Python 3.12.11、NumPy 2.2.6、OpenBLAS 0.3.29，每个进程的 BLAS 线程数为 1。Worker 在单机 CPU 上通过回环 TCP 通信。
- 取仓库 MNIST 训练集前 4,096 个样本，按原顺序运行 1 个 epoch。固定 seed 42、全局 batch 64、学习率 0.01。1、2、4 Worker 各自每步处理 64、32、16 张图像。相同 Worker 数的三次配对依次按 PS→Ring、Ring→PS、PS→Ring 顺序执行。
- 在 4 Worker 下，PS 的 PUSH 往返耗时平均占 Worker 训练步累计时间的 40.5%，Ring 的梯度规约占 21.0%。PUSH 计时包含服务器等待、聚合及参数更新。Ring 计时包含梯度打包、邻居间交换、等待和平均，本地优化器更新单独计入训练步。端到端和训练吞吐量均涵盖两种实现完整的参数更新开销。
- 这些数据描述这个模型和本机 CPU 进程的扩展性。跨主机链路、GPU 集合通信和更大模型需要分别测量。

## 复现

准备仓库中的 `tests/testdata/MNIST_data/mnist.pkl`，在已安装项目依赖的环境中，从仓库根目录运行以下命令。输出目录包含逐次 CSV、按进程记录的计时和损失数据。安装 matplotlib 后还会生成吞吐量和同步耗时占比图。

```powershell
Set-Location code
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m eneuro.distributed.benchmark --backends ps ring --dataset mnist --samples 4096 --epochs 1 --batch-size 64 --workers 1 2 4 --repeats 3 --output metrics/ring-mnist
```
