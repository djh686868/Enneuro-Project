# EnNeuro CUDA C 算子改写近期工作汇报文稿

> 汇报主题：从 CuPy GPU 路径到可验证的 CUDA C 算子后端  
> 建议时长：10–12 分钟  
> 报告日期：2026-09-20  
> 测试环境：Compute Capability 8.9（sm_89）、CuPy 13.3.0、CUDA 12.6

## 一句话结论

这一阶段我们没有重写整个 EnNeuro 框架，而是在保留 Python 自动求导、CuPy 显存管理和 cuBLAS 矩阵乘法的前提下，把高频、可明确映射到线程的热点算子改写为 CUDA C kernel，并完成了 RawModule 路线的正确性和小规模性能闭环。当前固定输入算子微基准、MNIST 前向/反向和 1 epoch 训练冒烟测试分别测得约 4.99 倍、19.69 倍和 9.38 倍的 RawModule 加速；同时，正式 DLL 的设备源码已经迁移并构建完成，但 `ctypes`/`extension` 后端还没有接入框架，也尚未基于修正后的 dtype 重新完成完整 MNIST 三方训练。

## 开场讲稿

各位老师好，今天汇报我们近期围绕 EnNeuro CUDA C 算子改写所做的工作。我们关注的问题不是“把 Python 全部改成 C”，而是：在不破坏现有自动求导和模型接口的情况下，能不能把最影响 GPU 效率的几个算子交给我们自己编写的 CUDA C kernel，并用可复现的数值和性能实验确认它确实被调用、确实带来收益。

今天我主要讲四件事：第一，我们对原有 CuPy 路径和瓶颈做了哪些分析；第二，目前已经完成了哪些代码和验证；第三，当前数据能支持什么结论、不能支持什么结论；第四，下一步如何从 RawModule 验证版推进到 DLL 接入、完整训练和更高性能的实现。

## 1. 为什么要做 CUDA C 算子改写

EnNeuro 原本已经能够在 GPU 上运行，底层主要依赖 CuPy。CuPy 并不是在 CPU 上逐个元素计算，而是会把数组表达式分解为若干 GPU kernel。因此问题并不是“CuPy 没有用 GPU”，而是一个逻辑算子可能被拆成多次 kernel 启动，并产生用完即丢的中间数组。例如卷积窗口展开、池化反向的索引搬运和 `col2im`，都可能带来额外的显存读写和 kernel launch。

我们的改写目标是把这些热点中的一部分合并成可控的 CUDA C kernel：明确一个线程负责哪个输出位置，明确边界条件，尽量减少临时数组和 Python/CuPy 层的调度次数。矩阵乘法本身继续交给 CuPy/cuBLAS，因为这是成熟库已经高度优化的部分；如果自研 kernel 不适合某个 dtype、布局或形状，就保留 CuPy 回退路径。

这里有一个需要特别说明的取舍：CUDA C 的优势不是“语言天然比 Python 快”，而是我们可以改变计算的组织方式。Python 仍然负责准备张量、保存自动求导关系和选择后端，GPU 线程负责密集的逐元素、窗口和梯度计算。

## 2. 我们做过的分析

### 2.1 后端路线分析

我们把实现路线分成三层，而不是一次性跳到 DLL：

| 路线 | 当前作用 | 结论 |
| --- | --- | --- |
| CuPy | 功能基线、回退路径和矩阵乘法 | 保留，作为公平对照 |
| RawModule / NVRTC | 把统一的 `.cu` 源码在运行时编译并验证 | 已完成第一轮闭环 |
| ctypes + CUDA DLL | 把同一份设备源码用 `nvcc` 编译成正式扩展 | DLL 已构建，Python loader 待接入 |

这种顺序的好处是：先验证索引、边界、梯度和训练行为，再处理 ABI、动态库加载和发布问题。否则一旦 DLL 路线出错，很难判断问题来自数学公式还是来自编译/调用边界。

### 2.2 算子与线程映射分析

我们把算子按并行结构拆开：

- **逐元素算子**：add、sub、mul、div、neg、exp、log、pow、ReLU、sigmoid 等，采用“一线程对应一个线性元素”的映射，并用 `i < n` 处理尾部线程。
- **卷积前向**：CUDA C 负责 `im2col`，矩阵乘法仍由 CuPy/cuBLAS 完成，bias 加法由 CUDA C kernel 完成。
- **卷积输入梯度**：不再先生成完整的 `gcol` 再做 `col2im`，而是让一个线程负责一个 `gx[n,c,h,w]`，反向枚举能够影响它的输出位置并在寄存器中累加 `gy × W`。这种 gather 方式避免了多个线程同时写同一个 `gx`。
- **最大池化前向**：一个线程扫描一个输出窗口，同时保存最大值位置 `argmax`。
- **最大池化反向**：当前版本按保存的 `argmax` 从输出梯度写回输入；重叠窗口使用 `atomicAdd`。因此它已经是 CUDA C kernel，但还不是确定性 gather 版本，原子加法的写入顺序可能带来极小的浮点差异。

### 2.3 公平比较分析

在早期三方实验中我们发现，LeNet 初始化时 `float32` 随机权重乘以 `float64` 缩放系数，会把权重提升为 `float64`，从而绕过只支持 `float32` 的 RawModule 快速路径。与此同时，Adam 的状态曾经使用类级共享字典，连续比较不同后端时可能复用前一个实验的动量状态。

因此本轮除了写 kernel，还修正了两项实验基础设施：

1. 让 Linear/Conv2d 的初始化缩放系数保持请求的 dtype；
2. 让每个 Optimizer 实例拥有独立的 Adam 状态。

这两项修正不是“额外优化”，而是为了保证 CuPy 和 RawModule 在相同初始化、相同训练状态下比较。此前 `artifacts/cuda_stage1_probe_1/mnist_three_way.json` 中的数值不一致属于优化前记录，不能拿来支持本轮 RawModule 的性能结论。

### 2.4 测量与验收分析

我们在 benchmark 中加入了 warm-up、CUDA 事件计时、数值误差和 kernel 发射计数。发射计数很重要：如果只看最终 loss 或耗时，某个测试可能实际上已经回退到 CuPy；计数可以确认 `im2col_f32`、`pool_bwd_f32`、`conv_bwd_x_f32` 等自研 kernel 真的被调用。

![后端架构与回退关系](../figures/cuda-c-backend-route.png)

## 3. 已经完成的内容

### 3.1 统一 CUDA C 源码与 RawModule 缓存

设备代码已经从 Python 字符串迁移到 `code/eneuro/base/cuda/sources/kernels.cu`。RawModule 和后续 DLL 都读取同一份设备源码，避免两条路线的数学公式逐渐分叉。`compiler.py` 按源码哈希、计算能力和编译选项缓存 RawModule；`dispatch.py` 还缓存 kernel 函数句柄，减少重复查找和编译开销。

当前 `kernels.cu` 实际包含 16 个 `__global__` kernel：4 个二元逐元素、7 个一元/激活相关、1 个 bias、1 个 im2col、1 个池化前向、1 个池化反向和 1 个卷积输入梯度 kernel。所有首版 kernel 以 float32、连续数组和已经验证的布局为主要覆盖范围。

### 3.2 接入自动求导和模型算子

基础算子入口已经能根据后端状态选择 CuPy 或 RawModule。Conv2d 的前向路径接入了 CUDA im2col、cuBLAS GEMM 和 CUDA bias；反向路径把输入梯度切换到 `conv_bwd_x_f32`，权重梯度和偏置梯度仍由 CuPy/cuBLAS 完成。Pooling 前向和反向也已经接入自研 kernel。上层 Tensor、Layer 和 LeNet 的公共接口不变，`auto` 模式在不支持的输入或 kernel 失败时仍可回退到 CuPy，`rawmodule` 模式则严格抛错，便于验收。

![卷积与池化算子的数据流](../figures/cuda-operator-dataflow.png)

### 3.3 正确性验证

项目阶段验收报告记录了在 CUDA 12.6 环境下的 12 项 CUDA kernel/LeNet 回归测试通过。测试覆盖基础逐元素算子、ReLU 前后向、bias、im2col、卷积前向和梯度、池化最大值索引与重叠窗口梯度，以及 LeNet 端到端前向/反向；同时检查关键 kernel 的发射，避免“测试通过但实际走了 CuPy 回退”。

这组“12 passed”是当前项目报告中的历史运行记录。正式提交前仍需要在清洁环境中重跑并保存完整日志，确保依赖和驱动条件没有变化。

### 3.4 DLL 源码迁移与构建

正式路线的 `sources/library.cu`、`sources/api.h` 和 `build_cuda.ps1` 已经加入，生成了面向 `sm_89` 的 `enneuro_cuda_sm89.dll`。ABI 设计只传递 CuPy 管理的设备指针、host 侧形状元数据和 CUDA stream，不在 DLL 内重复分配张量或强制同步。当前已完成动态库和 ABI 符号的加载验证；但 `dispatch.py` 中的 `extension` 选项尚未接入真正的 ctypes loader，因此这一步应表述为“DLL 构建完成”，不能表述为“框架已经支持 extension 后端”。

## 4. 当前实测结果

下表来自 `artifacts/cuda_stage1_probe/` 下的优化后 JSON 结果。所有结果都在同一台 sm_89 设备上测量，RawModule 与 CuPy 使用相同输入和相同模型设置。

| 场景 | CuPy | CUDA C RawModule | 当前观察 |
| --- | ---: | ---: | --- |
| 固定输入算子微基准，warm-up 5、测量 30 次 | 0.91969 ms | 0.18432 ms | 4.99×；两条路径最大绝对误差均为 1.1444e-05 |
| MNIST 512 张、batch 64、8 个 batch，前向+反向 | 76.04 ms/batch | 3.86 ms/batch | 19.69×；平均 loss 都是 2.2972087，准确率都为 6.25% |
| MNIST 512/512、batch 64、1 epoch Adam 训练 | 716.89 ms/epoch | 76.46 ms/epoch | 9.38×；训练 loss 都是 2.1758721，测试准确率为 56.64%/56.84% |

![CUDA C RawModule 与 CuPy 的耗时对比](../artifacts/cuda_report_figures/cuda-speedup-summary.png)

第一组结果说明在固定的组合算子上，RawModule 已经能获得明显的端到端耗时下降。第二组结果说明收益不只来自一个孤立 kernel：前向、反向和卷积输入梯度都进入了自研路径。第三组结果说明训练闭环可以运行，参数更新后的 loss 和准确率行为基本一致。

这里的准确率需要正确解读：第二组实验是在训练前的固定子集上测量，6.25% 只是确认两条路径的输出行为一致，不代表模型质量；第三组只有 512 个训练样本和 1 个 epoch，只能称为训练冒烟测试，不能代表完整 MNIST 的最终精度或吞吐。

从 launch count 看，8 个 batch 的前向/反向测试实际发射了 `im2col_f32=36`、`bias_f32=45`、`relu_f32=36`、`pool_f32=18`、`relu_bwd_f32=36`、`pool_bwd_f32=18`、`conv_bwd_x_f32=18`；1 epoch 训练也发射了对应的前向和反向 kernel。这是确认没有静默回退的重要证据。

## 5. 这些结果能说明什么，不能说明什么

### 可以说明的内容

1. 在当前已覆盖的 float32、连续数组和典型 LeNet 形状上，手写 CUDA C kernel 能够接入现有自动求导链路。
2. 当前测试输入上，RawModule 与 CuPy 的 loss 和主要输出行为一致；固定输入微基准的最大绝对误差处于 1.1444e-05 量级。
3. 在本机小规模 benchmark 中，减少中间数组、合并窗口计算和缓存 RawModule 函数句柄，确实带来了可观的耗时下降。
4. 同一份 `kernels.cu` 已经可以同时服务 RawModule 验证路线和 DLL 构建路线，为下一步扩展打下基础。

### 目前不能说明的内容

1. 不能把 4.99×、19.69× 或 9.38× 直接外推为完整 MNIST 或所有模型形状的加速比。
2. 不能说当前已经完成了全量 CUDA C 算子改写。`col2im`、自研 GEMM、Softmax/CrossEntropy 的完整 CUDA kernel、sum/mean/broadcast 归约等仍保留 CuPy；卷积 `gW/gb` 也仍由 CuPy/cuBLAS 计算。
3. 不能说 DLL 后端已经在框架中可选。动态库已构建并能加载 ABI 符号，但 Python `ctypes` loader 和 extension dispatch 尚未完成。
4. 不能把池化反向称为确定性 gather。当前实现是重叠窗口上的 `scatter + atomicAdd`，后续才考虑无原子、可复现的 gather 版本。
5. 不能把旧的三方不一致结果当作本轮结论；完整 60,000/10,000 MNIST、30 epoch 实验需要在 dtype 修正后重新执行。

## 6. 下一步计划与验收标准

![CUDA C 改写路线图](../figures/cuda-stage-roadmap.png)

### P0：把当前结果变成可复核的阶段验收

- 在固定 CUDA 12.6、CuPy 13.3、sm_89 环境中重跑 12 项回归测试，保存命令、依赖版本和完整日志。
- 重新执行 60,000/10,000 MNIST 三方训练；统一随机种子、初始化权重、数据顺序、batch size 和学习率。
- 增加 `cupyx.cudnn` 作为接近 PyTorch/cuDNN 的性能参照，并继续用 CUDA Event 测端到端时间，而不是只测单个 kernel。
- 验收条件：数值对拍通过、RawModule launch count 非零、三方指标可解释，且所有未覆盖场景明确记录为回退。

### P1：完成 DLL extension 接入

- 增加 Python `ctypes` loader，绑定 `eneuro_abi_version`、错误码、stream 和 `eneuro_launch_v1`。
- 让 `ENNEURO_CUDA_BACKEND=extension` 真正调用 DLL，并对 DLL 与 RawModule 做同一套 kernel、LeNet 和 benchmark 对拍。
- 对 DLL 缺失、ABI 版本不匹配、设备架构不匹配给出可诊断的错误，并在 `auto` 模式回退到 CuPy。
- 验收条件：CuPy、RawModule、extension 三条路径在同一输入上数值一致，且 DLL 路径的 launch count/耗时有独立记录。

### P2：补齐覆盖矩阵和边界行为

- 系统测试非方形输入、padding/stride 边界、奇数尺寸、不同 batch/channel、dilation，以及 dtype 不满足时的回退。
- 视需求补充独立 `col2im`、one-hot CrossEntropy、sum/mean/broadcast 等路径的测试或明确保留项。
- 对池化反向比较 atomicAdd 版与 gather 版，分别报告速度、误差和确定性。

### P3：针对大形状做性能深化

- 对大尺寸卷积评估 implicit GEMM/CUTLASS，避免显式写入完整 im2col。
- 对 `gW` 使用 split-K 或 workspace 两阶段归约，提供速度优先和确定性优先两种模式。
- 评估 bias+ReLU epilogue 融合、NHWC/channels-last、FP16/BF16 输入与 FP32 累加、Tensor Core/TF32。
- 以形状、dtype、layout 和确定性要求作为缓存键，按形状选择自研 kernel、cuDNN 或 CuPy 回退，而不是强迫一个 kernel 覆盖所有输入。

### P4：工程化与长期维护

- 把 CUDA 12.6、不同 SM 架构、CPU 回退和无 GPU 环境纳入 CI/验收矩阵。
- 保留 kernel 发射计数、实际后端、回退原因和耗时诊断，避免性能回归只能靠猜。
- 将二进制产物与源码版本、kernel hash、编译参数绑定，保证 DLL 可追溯。

## 7. 收尾讲稿

如果用一句话总结当前阶段，我们已经完成了“可验证的 CUDA C 第一版”，还没有完成“覆盖所有算子、所有形状的最终后端”。第一版的价值在于把最关键的路径跑通：自研 kernel 能进入真实的 LeNet 前向、反向和参数更新，数值对拍成立，小规模实测也看到了明显收益；同时，CuPy 回退和 cuBLAS 保留让框架仍然可以继续工作。

下一阶段的重点不是盲目增加 kernel 数量，而是先完成 DLL extension 的闭环和完整训练复验，再用 cupyx.cudnn 建立性能上限，最后针对确实存在收益的形状推进 gather、implicit GEMM、split-K 和算子融合。这样每一次改写都能回答三个问题：算得对不对、真的走没走、是否值得保留。

## 8. 建议的 PPT 页面拆解

| 页码 | 页面主题 | 主要内容 | 推荐素材 |
| ---: | --- | --- | --- |
| 1 | 工作目标与一句话结论 | 为什么改、当前到哪一步 | 本文“一句话结论” |
| 2 | 原有 CuPy 路径的瓶颈 | kernel 启动、中间数组、回退 | 口头示意 |
| 3 | 三层后端路线 | CuPy、RawModule、DLL | `cuda-c-backend-route.png` |
| 4 | 算子如何分工 | 自研 kernel 与 cuBLAS/CuPy 的边界 | `cuda-operator-dataflow.png` |
| 5 | 已完成的代码改动 | 16 个 kernel、缓存、dtype/Adam 修正 | 文件路径表 |
| 6 | 正确性证据 | 12 项历史回归、launch count、误差 | 测试结果表 |
| 7 | 性能结果 | 4.99×、19.69×、9.38× | `cuda-speedup-summary.png` |
| 8 | 结果边界 | 小规模、未全量重跑、DLL 未接入、atomicAdd | “不能说明的内容” |
| 9 | 下一阶段路线 | extension、全量训练、cuDNN、性能深化 | `cuda-stage-roadmap.png` |
| 10 | 需要的支持/决策 | 是否进入 DLL 接入和完整验收 | P0/P1 验收标准 |

## 9. 证据索引

- 阶段验收与历史测试记录：`doc/CUDA_C_阶段验收汇报.md`
- RawModule/CUDA C 统一分派：`code/eneuro/base/cuda/dispatch.py`
- 设备 kernel 源码：`code/eneuro/base/cuda/sources/kernels.cu`
- RawModule 编译缓存：`code/eneuro/base/cuda/compiler.py`
- DLL ABI：`code/eneuro/base/cuda/sources/api.h`、`library.cu`
- CUDA kernel 回归：`code/tests/test_cuda_kernels.py`、`code/tests/test_cuda_lenet.py`
- 固定输入微基准：`artifacts/cuda_stage1_probe/optimized_cuda_gpu_two_way.json`
- MNIST 前向/反向：`artifacts/cuda_stage1_probe/optimized_mnist_gpu_two_way.json`
- Adam 训练冒烟：`artifacts/cuda_stage1_probe/optimized_mnist_training_gpu_two_way.json`
- 当前图示源文件：`figures/cuda-c-backend-route.mmd`、`cuda-operator-dataflow.mmd`、`cuda-stage-roadmap.mmd`
