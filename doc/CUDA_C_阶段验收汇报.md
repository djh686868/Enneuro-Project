# EnNeuro CUDA C 算子改写阶段验收汇报

> 汇报时长：约 5 分钟  
> 验收阶段：Python 内嵌 CUDA C / CuPy RawModule 快速验证路线  
> GPU：计算能力 `sm_89`；CuPy `13.3.0`

## 1. 工作目标与结论

本阶段的目标是把框架底层高频算子和简单神经网络算子改写为 CUDA C kernel，先通过 CuPy `RawModule` 在 Python 中运行，验证以下三点：

1. CUDA C kernel 与原 CuPy/NumPy 实现的数值结果一致；
2. CUDA C kernel 可以接入 EnNeuro 自动求导和 LeNet-5 训练流程；
3. 在相同输入、相同参数和相同训练配置下，评估 CUDA C 相对 CPU NumPy 与 CuPy 的性能。

本阶段已经完成：基础算子测试、CUDA C RawModule 测试、MNIST LeNet 全量训练和三方 benchmark。最终 30 epoch 训练中，CUDA C RawModule 达到约 `1.99x` CPU NumPy、约 `1.009x` CuPy 的总训练加速，测试准确率为 `98.98%`，功能验收通过。

这里的结果属于快速验证路线结果，下一阶段仍需把 kernel 从 Python 字符串迁移到独立 `.cu` 文件，再编译为 DLL。

## 2. 技术原理

快速验证路线不先构建 DLL，而是把 CUDA C 源码放入 Python 原始字符串：

```python
module = cp.RawModule(
    code=_SRC,
    options=("--std=c++14",),
    name_expressions=["relu_f32"],
)
kernel = module.get_function("relu_f32")
kernel(((n + 255) // 256,), (256,), (x, y, n))
```

CuPy 使用 NVRTC 在运行时编译 kernel。输入和输出保持为 CuPy GPU 数组，Python 只负责准备参数和发射 kernel，实际逐元素计算在 GPU 上完成。

以 ReLU 为例，每个线程负责一个元素：

```cpp
extern "C" __global__ void relu_f32(
    const float* x, float* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
```

线程索引由 block 索引、block 内线程索引和 block 大小共同确定。`i < n` 负责保护最后一个不完整线程块，避免越界访问。

后端分为三种主要模式：

| 后端 | 作用 |
|---|---|
| `cupy` | 使用原有 CuPy 实现 |
| `rawmodule` | 强制使用自研 CUDA C kernel，错误直接暴露 |
| `auto` | 优先使用 CUDA C，失败时回退到 CuPy |

卷积采用分阶段实现：

```text
NCHW 输入
  ↓
CUDA C im2col
  ↓
CuPy/cuBLAS GEMM
  ↓
bias_add CUDA C kernel
  ↓
NCHW 输出
```

这种设计把最需要验证的窗口索引、padding 和布局交给自研 kernel，同时继续使用 CuPy/cuBLAS 完成矩阵乘法。

## 3. 主要代码位置

| 文件 | 作用 |
|---|---|
| `code/eneuro/base/cuda/compiler.py` | RawModule 编译、GPU 检测和按架构缓存 |
| `code/eneuro/base/cuda/dispatch.py` | CUDA C kernel、后端切换和 CuPy 回退 |
| `code/eneuro/base/cuda/__init__.py` | CUDA 后端公共入口 |
| `code/eneuro/base/core.py` | `add`、`mul`、`exp`、`neg`、`sub`、`div`、`pow` 接入 |
| `code/eneuro/base/functions.py` | ReLU、Linear bias、Conv2d、Pooling 接入 |
| `code/tests/test_cuda_kernels.py` | 基础 kernel、im2col、卷积、池化测试 |
| `code/tests/test_cuda_lenet.py` | LeNet 前向和反向冒烟测试 |
| `code/bench_cuda_three_way.py` | CPU/CuPy/RawModule 算子 benchmark |
| `code/train_mnist_three_way.py` | CPU/CuPy/RawModule MNIST 完整训练对比 |

基础算子通过统一分派函数调用：

```python
def relu_forward(x):
    return _unary(x, "relu_f32", lambda a: cp.maximum(a, 0))
```

对于不满足首版 kernel 条件的输入，例如非 `float32`、需要广播的二元运算或非连续数组，代码会使用 CuPy 处理；`rawmodule` 严格模式会保留错误，便于验收时发现问题。

## 4. 正确性验收

pytest 测试命令：

```powershell
python -m pytest code/tests/test_cuda_backend_cpu.py code/tests/test_cuda_kernels.py code/tests/test_cuda_lenet.py -q
```

验收结果：

```text
10 passed in 2.15s
```

测试覆盖：

- `add`、`sub`、`mul`、`div`、`neg`、`exp`、`log`、`pow`；
- ReLU 前向和反向；
- sigmoid 和 bias add；
- im2col 的 stride、padding、dilation 与布局；
- Conv2d 输出与 CuPy 参考实现；
- MaxPool 最大值索引和并列最大值规则；
- LeNet 前向、CrossEntropyLoss 和反向传播。

独立 RawModule 验收脚本结果：

```text
基础逐元素算子：全部通过
im2col：max_abs_error = 0
Conv2d：max_abs_error = 1.43e-6
MaxPool：全部通过
actual_backend = rawmodule
fallback_reason = null
```

## 5. MNIST 三方训练结果

实验配置：

- 训练集：60000 张 MNIST 图像；
- 测试集：10000 张 MNIST 图像；
- 模型：EnNeuro LeNet；
- batch size：32；
- epoch：30；
- 优化器：Adam，学习率 `0.001`；
- 三方使用相同初始参数、相同 batch 顺序和相同数据。

### 5.1 总耗时

| 实现 | 总训练时间 | 相对 CPU NumPy |
|---|---:|---:|
| CPU NumPy | 1442.23 s | 1.00x |
| CuPy | 729.82 s | 1.98x |
| CUDA C RawModule | 723.17 s | 1.99x |

CUDA C RawModule 相对 CuPy 的加速比为：

```text
729.82 / 723.17 = 1.0092x
```

也就是总训练时间减少约 `0.92%`，约节省 `6.65 s`。

### 5.2 训练效果

| 实现 | 第 30 epoch 测试准确率 |
|---|---:|
| CPU NumPy | 99.07% |
| CuPy | 99.17% |
| CUDA C RawModule | 98.98% |

三方均完成收敛，RawModule 与 NumPy 的最大测试准确率差为 `0.49` 个百分点，满足当前功能等价阈值 `1%`。

### 5.3 训练曲线与耗时图

![MNIST LeNet 三方训练曲线与总耗时对比](../artifacts/cuda_stage1_probe/mnist_training_full_three_way.png)

图中左侧为训练 loss，中央为测试准确率，右侧为三方总训练时间。RawModule 与 CuPy 的曲线接近，说明改写没有破坏训练行为；右侧耗时图显示当前 CUDA C 路线已经略快于 CuPy，但优势仍然有限。

## 6. 阶段结论与下一步

本阶段完成了从底层逐元素算子到 LeNet 训练的闭环验证：CUDA C kernel 可以被编译、发射并接入现有自动求导框架，真实 MNIST 训练的准确率与 CuPy 基线一致，并取得轻微端到端速度优势。

当前性能提升有限的主要原因是：卷积矩阵乘法仍使用 CuPy/cuBLAS，池化反向仍使用 CuPy 索引累加，Python 层仍存在多次 kernel 发射和算子调度。下一阶段进入最终交付路线后，重点工作是：

1. 将 kernel 从 `dispatch.py` 字符串迁移到独立 `.cu` 文件；
2. 使用 `nvcc` 编译 Windows x64 DLL；
3. 实现专用 Conv2d 输入梯度和 MaxPool backward gather kernel；
4. 减少 Python 调度和中间张量分配；
5. 重新进行同样的 MNIST 30 epoch 三方验收。

## 7. 可复现实验命令

```powershell
python -m pytest code/tests/test_cuda_backend_cpu.py code/tests/test_cuda_kernels.py code/tests/test_cuda_lenet.py -q
```

```powershell
python code/test_cuda_fast_route.py --backend rawmodule --out artifacts/cuda_stage1_probe/fast_route_test.json
```

```powershell
python code/train_mnist_three_way.py --train-samples 60000 --test-samples 10000 --epochs 30 --batch-size 32 --lr 0.001 --out artifacts/cuda_stage1_probe/mnist_training_full_three_way.json --plot artifacts/cuda_stage1_probe/mnist_training_full_three_way.png
```

报告数据来源：

[mnist_training_full_three_way.json](../artifacts/cuda_stage1_probe/mnist_training_full_three_way.json)

