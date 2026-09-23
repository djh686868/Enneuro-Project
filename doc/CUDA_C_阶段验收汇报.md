# EnNeuro CUDA C 算子改写阶段验收汇报

> 汇报时长：约 5 分钟
> 当前路线：Python 内嵌 CUDA C / CuPy RawModule 快速验证
> 测试设备：Compute Capability 8.9（sm_89），CuPy 13.3.0，CUDA 12.6

## 1. 工作目标与当前结论

本阶段把框架中已完成的底层逐元素、卷积和池化算子接入 CUDA C kernel，并在 MNIST LeNet 训练流程中验证前向、反向和参数更新。

本轮优化完成了卷积输入梯度 CUDA kernel、池化反向 CUDA kernel、RawModule 函数句柄缓存、运行时后端状态统一、LeNet 权重 dtype 修正，以及 Adam 实例状态隔离。CUDA kernel 回归测试为 12 项通过。固定输入算子微基准中 RawModule 比 CuPy 快约 4.99 倍；512 张 MNIST、8 个 batch 的前向/反向测试中快约 19.69 倍；512 张 MNIST、1 epoch 的 Adam 训练冒烟测试中快约 9.38 倍。所有数字都是当前小规模测试的实测值，不代表完整 MNIST 训练的最终加速比。

需要特别说明：此前记录的 60,000 张训练集、30 epoch 三方结果属于优化前版本，不能作为本轮性能结论。检查发现 LeNet 初始化时，float32 随机权重乘以 float64 缩放系数后变成了 float64，导致只支持 float32 的 RawModule 路径大部分时间回退到 CuPy。本轮已修复 dtype，但尚未重跑完整三方训练。

## 2. 技术原理简述

快速验证路线不先构建 DLL，而是把 CUDA C 源码作为字符串交给 CuPy RawModule。CuPy 在运行时通过 NVRTC 编译并加载 kernel，Python 侧负责准备 CuPy 数组和发射配置，逐元素计算则由 GPU 执行。

以 ReLU 为例，每个 CUDA 线程负责一个线性元素；线程索引由 block 编号、block 内线程编号和 block 大小组成，边界条件 i < n 保护最后一个不完整线程块：

```cpp
extern "C" __global__ void relu_f32(
    const float* x, float* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
```

本轮补充的反向 kernel 处理两个原先较慢的算子：

- 卷积输入梯度：一个线程负责一个输入梯度元素，反向枚举与该输入位置相关的输出通道和卷积窗口，并累加上游梯度乘卷积权重。
- 最大池化输入梯度：每个输出梯度线程根据前向保存的最大值索引找到输入位置，再用 atomicAdd 累加。重叠窗口可能写到同一输入元素，因此需要原子加法避免并发写冲突。

池化反向的核心写入如下。为突出计算逻辑，片段省略了从线性线程索引恢复 batch、channel 和空间坐标的代码：

```cpp
long long k = indexes[out_index];
int ih = oh * stride_h + (int)(k / kernel_w) - pad_h;
int iw = ow * stride_w + (int)(k % kernel_w) - pad_w;
if (ih >= 0 && ih < H && iw >= 0 && iw < W)
    atomicAdd(&gx[input_index], gy[out_index]);
```

卷积输入梯度 kernel 则为每个输入位置收集所有能影响该位置的输出梯度。以下片段同样省略坐标解码：

```cpp
float acc = 0.0f;
for (int oc = 0; oc < out_channels; ++oc)
  for (int kh = 0; kh < kernel_h; ++kh)
    for (int kw = 0; kw < kernel_w; ++kw) {
      int oh_num = ih + pad_h - kh;
      int ow_num = iw + pad_w - kw;
      if (oh_num >= 0 && ow_num >= 0 &&
          oh_num % stride_h == 0 && ow_num % stride_w == 0) {
        int oh = oh_num / stride_h, ow = ow_num / stride_w;
        if (oh < out_h && ow < out_w) {
          long go = (((long)n * out_channels + oc) * out_h + oh) * out_w + ow;
          long wi = (((long)oc * channels + c) * kernel_h + kh) * kernel_w + kw;
          acc += gy[go] * w[wi];
        }
      }
    }
long gi = (((long)n * channels + c) * in_h + ih) * in_w + iw;
gx[gi] = acc;
```

卷积前向仍采用分阶段路径：CUDA C im2col 展开输入窗口，CuPy/cuBLAS GEMM 计算矩阵乘法，CUDA C bias kernel 添加偏置。RawModule 函数句柄也会缓存，避免每次调用都重新查找 kernel。

当前有 cupy、rawmodule、auto 三种运行模式。rawmodule 用于严格验收；若 kernel 出错会抛出异常。auto 可在 kernel 不可用时回退到 CuPy。只有 float32 连续数组等受支持输入会走自研 kernel，其他 dtype 或广播场景仍由 CuPy 处理。

## 3. 主要代码位置

| 文件 | 作用 |
|---|---|
| code/eneuro/base/cuda/dispatch.py | CUDA C kernel、RawModule 函数缓存、后端分派、kernel 发射计数 |
| code/eneuro/base/cuda/compiler.py | RawModule 编译、设备与 NVRTC 可用性检测 |
| code/eneuro/base/core.py | add、sub、mul、div、neg、exp、log、pow 等基础算子接入和严格后端错误处理 |
| code/eneuro/base/functions.py | ReLU、bias、Conv2d、MaxPool 前向与反向接入 |
| code/eneuro/nn/module.py | Linear、Conv2d 权重初始化保持请求的 dtype，避免 float32 被缩放系数提升为 float64 |
| code/eneuro/nn/optim.py | 每个 Optimizer 实例独立维护 Adam 等状态 |
| code/tests/test_cuda_kernels.py | CUDA kernel 数值与梯度参考测试 |
| code/tests/test_cuda_lenet.py | LeNet 前向、反向和 RawModule kernel 使用验证 |
| code/bench_cuda_three_way.py | 固定输入算子 benchmark，支持选择测量后端 |
| code/bench_mnist_three_way.py | MNIST LeNet 小批次前向/反向 benchmark，支持仅测 GPU 后端 |
| code/train_mnist_three_way.py | MNIST LeNet 训练 benchmark，支持仅测 GPU 后端并记录 kernel 发射次数 |

## 4. 正确性验证

在 EnNeuro 环境、CUDA 12.6 下运行：

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'; $env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;'+$env:PATH; & 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' -m pytest code/tests/test_cuda_backend_cpu.py code/tests/test_cuda_kernels.py code/tests/test_cuda_lenet.py -q
```

结果：

```text
12 passed in 1.85s
```

测试覆盖基础算子、ReLU 前后向、bias、im2col、卷积前向和梯度、池化最大值索引与重叠窗口梯度，以及 LeNet 的端到端前向和反向。测试同时检查 conv_bwd_x_f32、pool_bwd_f32 确实被发射，避免仅因 CuPy 回退而误报通过。

## 5. 优化后性能测试

### 固定输入算子微基准

配置为 warmup 5 次、测量 30 次。每轮包含 exp、ReLU、im2col、矩阵乘法和 bias：

| 实现 | 平均耗时 | 对比 |
|---|---:|---:|
| CuPy | 0.91969 ms | 1.00x |
| CUDA C RawModule | 0.18432 ms | 4.99x |

两种 GPU 实现相对 NumPy 参考的最大绝对误差均为 1.1444e-05。RawModule 测量中，exp、ReLU、im2col 和 bias kernel 各实际发射 30 次。

![优化后固定输入算子 CuPy 与 RawModule 耗时](../artifacts/cuda_stage1_probe/optimized_cuda_gpu_two_way.png)

### MNIST 固定子集前向与反向

使用仓库内 MNIST 数据，512 张样本、batch size 64、8 个 batch，测量训练前向和反向，不包含 Adam 参数更新：

| 实现 | 平均 batch 耗时 | 平均 loss | 准确率 |
|---|---:|---:|---:|
| CuPy | 76.04 ms | 2.2972087 | 6.25% |
| CUDA C RawModule | 3.86 ms | 2.2972087 | 6.25% |

RawModule 相对 CuPy 为 19.69x，loss 误差为 0。此时模型尚未训练，因此准确率只用于确认两条路径的输出行为，不代表模型效果。RawModule 实际发射了 im2col、bias、ReLU、pool 前后向和卷积输入梯度 kernel。

![MNIST 固定子集前向与反向耗时](../artifacts/cuda_stage1_probe/optimized_mnist_gpu_two_way.png)

### Adam 训练冒烟测试

使用 512 张训练样本、512 张测试样本、batch size 64、1 epoch：

| 实现 | 训练 epoch 耗时 | 训练 loss | 测试准确率 |
|---|---:|---:|---:|
| CuPy | 716.89 ms | 2.1758721 | 56.64% |
| CUDA C RawModule | 76.46 ms | 2.1758721 | 56.84% |

该小规模测试 RawModule 相对 CuPy 为 9.38x。两者 loss 相同，测试准确率相差约 0.20 个百分点。RawModule 确实发射了卷积和池化反向 kernel。样本和 epoch 数较少，结果仅证明训练闭环可运行，不用于推断完整 MNIST 的最终吞吐或泛化能力。

![MNIST Adam 训练冒烟测试](../artifacts/cuda_stage1_probe/optimized_mnist_training_gpu_two_way.png)

## 6. 结果边界与后续工作

本轮已证明已完成的 CUDA C 算子在当前测试输入上数值正确，并且在小规模 GPU benchmark 中比 CuPy 路径快。完整 60,000/10,000 MNIST 三方训练尚未基于修正后的 dtype 和优化器状态重跑，因此当前不能报告新的全量训练时长或正式的完整训练加速比。此前 30 epoch 结果应视作旧实现历史记录，不用于支持当前 RawModule 加速结论。

下一步如果需要完成最终性能验收，应在相同初始化权重、数据顺序、batch size 和环境下重新比较 CuPy 与 RawModule；CPU NumPy 全量训练成本较高，可将其留作参考基线或只跑较小子集。完整训练命令如下：

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'; $env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;'+$env:PATH; & 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' code/train_mnist_three_way.py --train-samples 60000 --test-samples 10000 --epochs 30 --batch-size 32 --lr 0.001 --out artifacts/cuda_stage1_probe/mnist_training_optimized_three_way.json --plot artifacts/cuda_stage1_probe/mnist_training_optimized_three_way.png
```

## 7. 本轮结果文件

- 固定输入算子对比：[optimized_cuda_gpu_two_way.json](../artifacts/cuda_stage1_probe/optimized_cuda_gpu_two_way.json)
- MNIST 前向/反向对比：[optimized_mnist_gpu_two_way.json](../artifacts/cuda_stage1_probe/optimized_mnist_gpu_two_way.json)
- Adam 训练冒烟测试：[optimized_mnist_training_gpu_two_way.json](../artifacts/cuda_stage1_probe/optimized_mnist_training_gpu_two_way.json)
- RawModule-only 微基准：[optimized_rawmodule_benchmark.json](../artifacts/cuda_stage1_probe/optimized_rawmodule_benchmark.json)

已完成 DLL 路线的源码迁移与 nvcc 构建：设备代码位于 `code/eneuro/base/cuda/sources/kernels.cu`，C ABI launcher 位于 `sources/library.cu`，生成的本地二进制为 `bin/enneuro_cuda_sm89.dll`。下一步将为 `extension` 后端接入 ctypes loader，并以同一套数值与性能脚本完成 DLL 三方复验。
