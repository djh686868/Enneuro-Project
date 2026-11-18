# EnNeuro CUDA C 算子改写与 ResNet18/DonkeyCar 验证报告

**当前路线：** CuPy RawModule/NVRTC 快速验证路线  
**CUDA 环境：** CUDA 12.6，RTX 4060，Compute Capability 8.9  
**CuPy：** 13.3.0  
**报告用途：** 说明当前已经改写的算子、实现原理和实际验证结果

## 1. 当前结论

本阶段已经把 ResNet18 所需的主要计算路径接入 CUDA C。小张量的 3×3、stride=1、padding=1 卷积使用 CUDA C Winograd F(2×2, 3×3) 前向核；大张量根据形状复用 CuPy 的向量化 im2col/GEMM、FFT 或 Winograd 路径，避免低占用的实验性 kernel 拖慢完整训练。卷积输入梯度在小张量上使用 CUDA C gather kernel，大张量使用 CuPy 向量化反向；权重梯度和偏置梯度使用 GPU 矩阵乘法与归约。

BatchNorm2d、GlobalAveragePooling、ReLU、残差加法、最大池化和基础逐元素算子也已经接入 RawModule。小张量仍执行 CUDA C kernel；大张量切换到 CuPy 向量化算子，避免大量低收益 kernel launch。ResNet18 可以在 GPU 上完成 DonkeyCar 图像的前向、反向、参数更新、验证和 checkpoint 保存。

当前已经完成的是快速验证路线。新增的设备代码位于 `.cu` 文件中，RawModule 运行时直接读取同一份源文件。Winograd、BatchNorm 和 GAP 的 DLL C ABI 包装还没有单独加入 `extension` 后端，因此当前实验应使用 `--backend rawmodule`。

针对完整 DonkeyCar 训练中 RawModule 慢于历史 CuPy 基线的问题，已补充三项修复：损失函数、GAP、BatchNorm 和 Adam 的标量运算显式保持 `float32`；大卷积复用与 CuPy 基线相同的形状缓存和自动路径选择；训练数据切分不再重复 `astype`，并释放原始全量数组。训练断点默认写入二进制 `.pkl`，避免把数组转换为多 GiB 的 JSON 文本。

## 2. 已改写算子清单

| 类别 | 已改写或接入的算子 | 当前实现 |
|---|---|---|
| 基础逐元素 | `add`、`sub`、`mul`、`div`、`neg` | CUDA C 一线程一元素 |
| 数学函数 | `exp`、`log`、`pow`、`sigmoid` | CUDA C `expf/logf/powf/tanhf` |
| 激活函数 | `relu`、`relu_backward` | CUDA C 前向和反向 |
| 偏置 | `bias_add` | 支持二维 Linear 和四维 NCHW |
| 卷积前处理 | `im2col` | CUDA C 窗口展开，支持 stride、padding、dilation |
| 3×3 主卷积 | `conv2d_forward` | 小张量 CUDA C Winograd；大张量按形状选择 CuPy 向量化路径 |
| 1×1/下采样卷积 | `conv2d_forward` | 小张量 CUDA C im2col；大张量复用 CuPy im2col/GEMM 或 FFT |
| 卷积反向 | `conv2d_backward_input` | 小张量 CUDA C gather；大张量 CuPy 向量化反向 |
| 卷积参数梯度 | `gW`、`gb` | CuPy/cuBLAS GEMM 和 GPU 归约 |
| 最大池化 | `maxpool_forward`、`maxpool_backward` | CUDA C 窗口扫描、索引保存和 atomicAdd 反向 |
| 归一化 | `BatchNorm2d` 前向和反向 | 统计量使用 CuPy 归约，归一化和输入梯度使用 CUDA C |
| 全局平均池化 | `GlobalAveragePooling` 前向和反向 | CUDA C 求和与广播梯度 |
| 残差连接 | `add` | 复用 CUDA C 二元逐元素 kernel |

以下运算继续使用 CuPy 的成熟实现：GEMM、通道归约、BatchNorm 均值/方差归约、权重梯度归约、reshape、transpose 和 flatten。这样可以先验证算子边界和训练闭环，再逐步替换归约和矩阵乘法。

当前 CUDA C Winograd 路径的支持范围是 `float32 NCHW`、3×3 卷积、stride=1、padding=1、dilation=1，默认只处理输出元素不超过 4096 的小张量。大张量由性能保护策略转入 CuPy 的向量化实现；其他卷积形状继续按 CuPy 的 FFT、GEMM 或 im2col 规则选择路径。可通过 `ENNEURO_WINOGRAD_MAX_OUTPUT` 和 `ENNEURO_RAW_REFERENCE_CONV_OUTPUT` 调整阈值。

## 3. 主要代码位置

| 文件 | 作用 |
|---|---|
| [`code/eneuro/base/cuda/dispatch.py`](../code/eneuro/base/cuda/dispatch.py) | 后端选择、RawModule 编译缓存、Winograd/BatchNorm/GAP 调度 |
| [`code/eneuro/base/cuda/sources/kernels.cu`](../code/eneuro/base/cuda/sources/kernels.cu) | 基础 kernel、卷积反向、Winograd、BatchNorm、GAP 的 CUDA C 实现 |
| [`code/eneuro/base/functions.py`](../code/eneuro/base/functions.py) | 将 Conv2d、BatchNorm2d、GlobalAveragePooling 接入 CUDA 后端 |
| [`code/eneuro/nn/module.py`](../code/eneuro/nn/module.py) | ResidualBlock、ResNet18 和 BatchNorm 层定义 |
| [`code/tests/test_resnet18_donkeycar.py`](../code/tests/test_resnet18_donkeycar.py) | DonkeyCar 图像读取、数据划分、ResNet18 训练和测试 |
| [`code/tests/test_cuda_resnet18_extensions.py`](../code/tests/test_cuda_resnet18_extensions.py) | Winograd、BatchNorm、GAP 数值对照测试 |
| [`code/test_cuda_fast_route.py`](../code/test_cuda_fast_route.py) | 基础 CUDA kernel 快速回归测试 |

## 4. 改写技术原理

### 4.1 RawModule 调用流程

Python 侧保留 Tensor、自动求导和 CuPy 显存管理。调用算子时，`dispatch.py` 检查当前后端、数组 dtype 和布局，然后把设备指针、张量尺寸和标量参数传给 RawModule kernel。

基础算子使用一维网格。线程索引为：

```cpp
int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) y[i] = ...;
```

当前使用 256 个线程一个 block。最后一个 block 通过 `i < n` 处理尾部元素。RawModule 函数句柄按 kernel 名称缓存，新增的 Winograd、BatchNorm 和 GAP kernel 使用懒加载模块，避免基础算子每次启动都编译完整扩展源码。

### 4.2 Winograd F(2×2, 3×3)

普通 3×3 卷积在每个输出位置需要 9 次乘加。Winograd 将输入和卷积核分成 4×4 tile，转换到 Winograd 域后进行逐元素乘法，最后再转换回 2×2 输出 tile：

```text
V = Bᵀ d B
U = G g Gᵀ
M = Σ(V ⊙ U)
Y = Aᵀ M A
```

其中 `d` 是输入 tile，`g` 是 3×3 卷积核，`V/U/M` 的大小为 4×4。CUDA kernel 中一个线程负责一个输出元素，线程根据输出位置找到对应 tile，完成输入变换、卷积核变换、通道累加和输出逆变换。边界输入按零填充处理，输出尺寸与当前框架的 cross-correlation 语义一致。

该实现没有翻转卷积核，仍然计算框架原有的 cross-correlation：

```text
y[n, oc, oh, ow] = Σ x[n, ic, ih, iw] * w[oc, ic, kh, kw]
```

### 4.3 卷积反向

输入梯度采用 gather 方式。每个线程负责一个 `gx[n,c,ih,iw]`，反向遍历可能影响该输入位置的输出通道和卷积窗口：

```text
gx[n,c,ih,iw] = Σ gy[n,oc,oh,ow] * w[oc,c,kh,kw]
```

这样每个线程只写一个输出位置，不需要为重叠窗口执行 col2im scatter，也不需要 atomicAdd。`gW` 复用 im2col 结果并调用 GPU GEMM，`gb` 对 `gy` 做通道归约。

### 4.4 BatchNorm2d

对 NCHW 输入，每个通道统计：

```text
μc = mean(x[:, c, :, :])
σ²c = mean((x[:, c, :, :] - μc)²)
x̂ = (x - μc) / sqrt(σ²c + ε)
y = γc x̂ + βc
```

均值和方差使用 CuPy 归约，CUDA C kernel 负责每个元素的标准化和缩放。反向计算 `dγ`、`dβ` 时使用通道归约，`dx` 使用 CUDA C kernel 按标准 BatchNorm 公式写回。

### 4.5 GlobalAveragePooling

对每个 `(n,c)`，CUDA kernel 对空间维度求平均：

```text
y[n,c] = (1 / H·W) Σ x[n,c,h,w]
```

反向梯度对每个空间位置写入 `gy[n,c] / (H·W)`。这也修正了原有反向路径只广播、没有除以空间元素数量的问题。

### 4.6 大张量性能保护

初版 RawModule 将一线程一输出的 CUDA C kernel 用在整个 ResNet18 上。该映射适合小张量验收，但在 batch=4 的大图训练中会带来较低占用、较高寄存器压力和大量 kernel launch。当前调度器默认把输出元素超过 4096 的卷积、归一化、激活、残差加法、池化和 GAP 交给 CuPy 向量化实现；小张量仍走 CUDA C kernel，因此算子仍可独立验证。

卷积的大张量选择与 CuPy 基线保持一致：满足大批量条件时才使用向量化 Winograd；大核且空间尺寸足够时使用 FFT；其余情况使用 im2col/GEMM。这样 RawModule 训练不会因为实验性 kernel 覆盖范围过宽而退化。若需要专门测试 CUDA C 大张量路径，可提高 `ENNEURO_RAW_REFERENCE_CONV_OUTPUT` 和 `ENNEURO_WINOGRAD_MAX_OUTPUT`，但该设置不作为 DonkeyCar 默认训练配置。

## 5. 当前验证结果

### 5.1 基础 CUDA 快速路线

执行 `code/test_cuda_fast_route.py --backend rawmodule`，基础算子、im2col、卷积前向和最大池化全部通过：

| 项目 | 结果 |
|---|---:|
| 测试数量 | 15 |
| 通过数量 | 15 |
| 卷积最大绝对误差 | `1.43e-6` |
| 最大池化最大绝对误差 | `0` |
| 实际后端 | `rawmodule` |
| CuPy 版本 | `13.3.0` |
| GPU 架构 | `sm_89` |

结果文件：[`fast_route_after_performance_fix.json`](../artifacts/cuda_stage1_probe/fast_route_after_performance_fix.json)。

### 5.2 新增扩展算子数值对照

使用随机 `float32` 输入与 CuPy 参考计算对照：

| 算子 | 最大绝对误差 |
|---|---:|
| Winograd 3×3 前向 | `2.15e-6` |
| BatchNorm 前向 | `4.77e-7` |
| BatchNorm 输入梯度 | `7.15e-7` |
| GlobalAveragePooling 前向 | `0` |
| GlobalAveragePooling 反向 | `0` |

这些误差来自 float32 运算顺序差异，均处于当前测试阈值内。

### 5.3 ResNet18 DonkeyCar 训练烟雾测试

数据目录为 `tests/test_donkey/data`，共识别到 6200 张 `id_angle.jpg` 图像。测试使用 8 张样本、32×32 输入、batch size 2、1 个 epoch，完成了训练、验证、测试和 checkpoint 保存。

| 指标 | 结果 |
|---|---:|
| 模型 | ResNet18Steering |
| 可训练参数 | 194049 |
| 训练样本 | 5 |
| 验证样本 | 1 |
| 测试样本 | 2 |
| 训练 MSE | `0.662148` |
| 验证 MSE | `0.235360` |
| 测试 MSE | `0.164821` |
| 测试 MAE | `0.397785` |
| epoch 时间 | `14.47 s` |
| CUDA Winograd 发射次数 | `39` |
| BatchNorm 前向发射次数 | `51` |
| GAP 前向发射次数 | `3` |

该结果证明 ResNet18 的残差连接、下采样、BatchNorm、池化、全局平均池化、反向传播和 Adam 更新可以在当前 CUDA 后端闭环运行。它是功能烟雾测试，不能代表完整 DonkeyCar 数据集的最终精度。

### 5.4 单个 3×3 卷积微基准

在输入 `(N=2,C=64,H=32,W=32)`、输出通道 64 的 3×3 卷积上，预热后测量得到：

| 路径 | 平均耗时 |
|---|---:|
| CuPy im2col + GEMM | `0.560 ms` |
| CUDA C Winograd | `0.345 ms` |

该形状下 CUDA C 路径约为 CuPy im2col 路径的 `1.62×`。这只是单个卷积形状的结果，完整模型还会受到 BatchNorm、池化、显存传输和 kernel launch 数量影响。

### 5.5 本次性能修复复测

问题现象是完整 DonkeyCar 训练中 RawModule 单 epoch 达到 `558.8746 s`，而此前 CuPy 最快约 `300 s`。定位结果是大尺寸卷积仍使用一线程一输出的 CUDA C Winograd/反向 kernel，同时 BatchNorm、ReLU、残差加法和池化为大张量频繁发射独立 kernel。

修复后使用同一数据目录、同一 `image-size=256`、`batch-size=4`、1 个 epoch 进行短跑对比（64 个样本，用于避免再次等待完整数据集）：

| 路径 | epoch 时间 | 相对 CuPy |
|---|---:|---:|
| CuPy | `15.8932 s` | `1.00×` |
| RawModule（修复后） | `15.9625 s` | `0.996×` |

RawModule 与 CuPy 已恢复到同一量级；复测时 `cuda_launch_counts` 只剩小型 `bias_f32` 发射，大型卷积、BN、ReLU、池化、GAP 和残差加法均走向量化路径。该短跑用于验证性能方向，不能替代完整 6200 张图像的最终基准；建议在同一进程预热后再进行完整 benchmark。

### 5.6 本次训练稳定性修复

旧实现中，MSE 反向的批次除法、GlobalAveragePooling 反向的空间除法以及部分 BatchNorm 常量使用 Python 标量。CuPy 在这些表达式上可能把 `float32` 梯度提升为 `float64`，随后所有卷积梯度和 Adam 缓冲区都会扩大一倍。修复后这些常量都按当前数组 dtype 创建；CPU ResNet18 单批次回归检查中所有参数梯度均保持 `float32`。

旧版训练脚本对已经是 `float32` 的每个数据切分再次执行 `astype`，并保留原始 `x_all`，256×256 全量数据会产生多份数 GiB 级副本。现在使用 `np.asarray(..., dtype=np.float32)`、切分后释放 `x_all/y_all`，避免无意义的主机内存复制。

检查点保存不再包含 epoch 末尾的梯度（下一批次会重新计算），只保存模型参数和 Adam 状态；加载时会把 Adam 数组恢复到模型所在的 CuPy/NumPy 后端。CPU 烟雾测试得到：二进制检查点可正常保存、重新加载到 epoch 1，110 个模型参数的梯度字段为空，76 组 Adam 状态保持 `float32`。

![已有 CUDA GPU 对比图](../artifacts/cuda_stage1_probe/optimized_cuda_gpu_two_way.png)

![已有 MNIST GPU 对比图](../artifacts/cuda_stage1_probe/optimized_mnist_gpu_two_way.png)

## 6. DonkeyCar 训练命令

推荐先使用 32×32 输入验证完整数据闭环。`--max-samples 0` 表示使用目录中全部样本；完整 256×256 训练建议使用 batch size 16，并使用 `.pkl` 检查点：

```powershell
$env:ENNEURO_CUDA_BACKEND='rawmodule'; $env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'; $env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;'+$env:PATH; & 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' code\tests\test_resnet18_donkeycar.py --data-dir tests\test_donkey\data --max-samples 0 --image-size 32 --epochs 4 --batch-size 8 --device cuda --backend rawmodule --save-checkpoint artifacts\cuda_stage1_probe\resnet18_donkeycar_full_fixed.pkl --save-each-epoch
```

256×256 完整训练命令：

```powershell
$env:ENNEURO_CUDA_BACKEND='rawmodule'; $env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'; $env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;'+$env:PATH; & 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' code\tests\test_resnet18_donkeycar.py --data-dir tests\test_donkey\data --max-samples 0 --image-size 256 --epochs 4 --batch-size 16 --device cuda --backend rawmodule --save-checkpoint artifacts\cuda_stage1_probe\resnet18_donkeycar_full_fixed.pkl --save-each-epoch
```

训练 checkpoint 推荐使用 `.pkl`。该格式直接保存二进制数组，当前 ResNet18 的 checkpoint 通常为 MB 级；`.json` 仍可用于兼容旧工具，但会把 GPU 数组展开成巨大的缩进文本，不适合每个 epoch 保存。

快速复核命令：

```powershell
$env:ENNEURO_CUDA_BACKEND='rawmodule'; & 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' code\test_cuda_fast_route.py --backend rawmodule --out artifacts\cuda_stage1_probe\fast_route_resnet_extension.json
```

如果 pytest 报 `ModuleNotFoundError: pygments`，先确认当前解释器能看到已安装的 Pygments：

```powershell
& 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' -c "import pygments; print(pygments.__file__)"
```

## 7. 当前边界

1. RawModule 路线已经完成并用于 ResNet18 训练；新增 Winograd、BatchNorm 和 GAP 的 DLL C ABI 入口还需要在正式交付路线中补齐。
2. CUDA C Winograd 当前只覆盖 float32 的标准 3×3 stride=1 padding=1 卷积；默认仅用于小张量，1×1、stride=2、dilation 和其他边界组合按 CuPy 基线选择路径。
3. 已验证 DonkeyCar 子集训练，尚未耗时运行完整 6200 张图像的多 epoch 训练，因此不能把子集 MSE 或 MAE 当作完整数据集最终指标。
4. 当前 benchmark 对比的是 CUDA C RawModule 与 CuPy im2col/GEMM 路径。它不等同于 cuDNN 或 PyTorch 的高度优化卷积性能。

## 8. 后续正式交付路线

快速验证通过后，正式交付按以下顺序进行：

1. 在 `sources/library.cu` 中增加 Winograd、BatchNorm、GAP 的 C ABI launcher。
2. 在 Python 侧实现 `extension` 后端的 ctypes loader。
3. 使用同一组输入、误差阈值和 DonkeyCar 训练脚本复验 RawModule、CuPy、DLL 三条路径。
4. 记录 DLL 的 ABI 版本、CUDA 架构、源码 hash、编译命令和性能结果。
