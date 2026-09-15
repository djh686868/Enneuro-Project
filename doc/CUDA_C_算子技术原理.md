# EnNeuro CUDA C 算子技术原理

## 1. Python、CuPy 与 CUDA C 的分工

Python 保留模型组织、自动求导、参数管理和训练循环；CuPy 负责数组生命周期和成熟的 cuBLAS reduction/GEMM；CUDA C kernel 负责窗口索引、逐元素运算、边界判断和反向 gather。改写的目的主要是减少中间数组和 kernel launch，并非因为 C 语法本身天然更快。

## 2. 逐元素算子

每个线程处理一个线性元素 `i`，读取输入并写回输出。`add/mul/sub/div` 使用相同的二元模板，`exp/log/relu/sigmoid` 使用一元模板。ReLU 的反向为 `gy*(x>0)`；sigmoid 反向为 `gy*y*(1-y)`。广播梯度需要沿被广播维度求和，首版使用 CuPy reduction。

## 3. 卷积与 im2col

NCHW 卷积公式为：

```text
y[n,o,oh,ow] = b[o] + Σ x[n,c,oh*sh+kh*dh-ph,ow*sw+kw*dw-pw] * W[o,c,kh,kw]
```

越界位置按零处理。输出尺寸使用 `get_conv_outsize(H, KH, S, P, D)`。CUDA kernel 中必须使用 `ih=oh*S+kh*D-P`、`iw=ow*S+kw*D-P`，不能只复制无 dilation 的 padding 逻辑。

`im2col` 把每个窗口展平成矩阵行：输入为 `(N,C,H,W)`，输出为 `(N*OH*OW,C*KH*KW)`，卷积因此变成 `col @ W.T + bias`。矩阵乘法保留 cuBLAS，CUDA C 负责高效生成连续的 col 和 bias epilogue。

## 4. 卷积反向 gather

权重梯度为 `gW[o,c,kh,kw] = Σ gy[n,o,oh,ow] * x[n,c,ih,iw]`，偏置梯度为 `gb[o] = Σ gy[n,o,oh,ow]`。

输入梯度采用 gather：每个线程唯一负责一个 `gx[n,c,h,w]`，反查所有满足 stride、padding、dilation 整除条件的输出窗口，累加 `gy*W` 后只写一次。这样避免多个线程 scatter 到同一地址，也避免创建 `KH*KW` 倍大小的 `gcol` 和使用 `atomicAdd`，结果更确定。

## 5. MaxPool

forward 对每个输出窗口保存最大值和窗口内扁平 argmax；相同最大值采用首次出现规则。backward 对每个输入像素 gather，反查包含它的窗口，只有 argmax 指向该像素时才累加对应 `gy`，所以无需中间 `gcol`。

## 6. GPU 线程和内存布局

首版采用 `blockDim.x=256`、`gridDim.x=ceil(total/256)`。线性线程编号按 `n,c,h,w` 解码；NCHW 连续布局中相邻线程优先访问相邻 `w` 地址，形成合并访存。每个 kernel 都检查 `tid < total`，显式处理边界，输入/权重/输出保持 contiguous，首版统一使用 float32 累加。

## 7. RawModule/NVRTC 与 DLL

快速路线由 `RawModule` 在运行时把 Python 中的 CUDA C 字符串交给 NVRTC 编译，再按 kernel 名称取得函数并发射。缓存键至少包含 `(op, compute_capability, dtype, options)`。正式路线将相同源码拆成 `.cu/.cuh`，nvcc 编译 DLL，导出只含 device pointer、整数 shape/stride/pad/dilation 和 dtype 的 C ABI；`ctypes` 通过 CuPy pointer 调用。

## 8. LeNet 数据流和比较

`28×28 → conv1(k5,pad2) → pool2 → conv2(k5) → pool2 → flatten(400) → Linear(120) → Linear(84) → Linear(10)`。同一初始权重分别运行 CuPy、RawModule 和 DLL，比较每层输出、`gx/gW/gb`、loss、参数更新和 accuracy。浮点加法顺序可能不同，因此使用 max absolute 和 relative L2，而不是逐位相等。

计时必须先 warm-up，再在 CUDA Event 前后同步；日志或指标需要转 NumPy 时才调用 `cp.asnumpy`，计算路径禁止隐式 host transfer。小 batch 可能使手写 kernel 不如 CuPy，应保留自动回退并记录原因。


## 9. Operator interface contract

All dispatch functions accept CuPy ndarray inputs and call `cp.ascontiguousarray` before launching a kernel. The first implementation supports float32 only. Unsupported dtype, layout, or shape must return the CuPy implementation and record `fallback_reason`.

```python
add_forward(x0, x1) -> y
mul_forward(x0, x1) -> y
exp_forward(x) -> y
relu_forward(x) -> y
relu_backward(x, gy) -> gx
im2col_forward(x, kernel, stride=1, pad=0, dilation=1) -> col
conv2d_forward(x, w, b=None, stride=1, pad=0, dilation=1) -> y
conv2d_backward(gy, x, w, b=None, stride=1, pad=0, dilation=1) -> (gx, gW, gb)
maxpool_forward(x, kernel, stride=1, pad=0) -> (y, indexes)
maxpool_backward(gy, indexes, input_shape, kernel, stride=1, pad=0) -> gx
```

Fixed shapes are `x=(N,C,H,W)`, `w=(OC,C,KH,KW)`, `b=(OC,)`, `gy=(N,OC,OH,OW)`, and `indexes=(N,C,OH,OW)` with int64 dtype. `conv2d_backward` always returns `(gx,gW,gb)`; `gb` is `None` when bias is absent.

## 10. Relation to automatic differentiation

CUDA kernels operate only on raw CuPy arrays and never create Tensor objects. `Function.__call__` still stores inputs and creators, and `Tensor.backward()` still traverses the graph. Integration order is ReLU, then Pooling, then Conv2d. Linear matrix multiplication remains on the existing cuBLAS path. A kernel failure is therefore isolated to one Function and cannot corrupt the graph.

## 11. Deterministic LeNet experiment

1. Generate fixed initial weights, inputs, and labels with NumPy and save them.
2. Copy the same values to CuPy and select `cupy`, `rawmodule`, and `extension` backends.
3. Run one forward, backward, and optimizer step for each backend.
4. Compare every layer output, loss, `gx/gW/gb`, and updated parameter.
5. Train the MNIST subset for 1–3 epochs and record batch loss, accuracy, and CUDA Event timings.

The report must print the spatial dimensions: `28x28 -> conv1(k5,pad2) -> 28x28 -> pool2 -> 14x14 -> conv2(k5) -> 10x10 -> pool2 -> 5x5`, hence the first Linear input is `16*5*5=400`. The custom model in `code/tests/test_lenet.py` is a compatibility regression only and must not be mixed into the main comparison.

## 12. Error handling and build boundary

CuPy missing, zero devices, NVRTC compilation failure, kernel launch failure, and DLL loading failure must all fall back while preserving the error text. Fallback occurs in Python dispatch, never through an implicit device-to-host copy. RawModule does not require nvcc; the DLL route requires CUDA Toolkit, nvcc, and the PowerShell build script. Compute capability is read from `cp.cuda.Device().compute_capability`. When a CUDA version error appears, switch the current session to CUDA 12.6 according to `AGENTS.md` and rerun the same command. Without real GPU Event data, report “not measured” rather than an estimated speedup.

## 13. 与现有 CuPy 基线的兼容性重点

当前 `im2col_array` 使用常数 0 填充，因此 CUDA MaxPool 首阶段也必须使用 0 填充；不能照搬其他框架的负无穷 padding 语义。重叠窗口的 backward 必须将多个输出梯度相加。索引只保存窗口内偏移 `kh*KW+kw`，不是全局 NCHW 地址。

当前框架有两套同名基础 Function：`core.py` 中的 `Exp/Add/Mul/...` 与 `functions.py` 中的 `Exp/Log/...`。它们都必须经过相同后端选择规则，但不能互相导入造成循环依赖。CUDA 包只依赖可选 CuPy和编译器；Function 模块在调用点局部导入 dispatch。

## 14. 原理到实现的对应关系

| 原理 | 代码职责 | 验证 |
|---|---|---|
| 一线程一元素 | raw kernel 的 tid/grid-stride loop | 一元/二元 shape 与空数组 |
| 滑窗坐标公式 | im2col kernel 的 ih/iw 边界判断 | stride、pad、dilation 矩阵对照 |
| im2col + GEMM | Python dispatch 的 col 与 cp.matmul | Conv forward 与 CPU朴素公式 |
| 输入梯度 gather | conv_gx kernel 每输入像素唯一写者 | 有重叠窗口的有限差分 |
| argmax 回传 | pool forward indexes + pool backward | tie、负数、零填充、重叠 |
| 自动求导保留 | Function 保存 context，Tensor.backward 遍历 | 重复forward和分叉图 |
| DLL 不拥有内存 | ctypes wrapper 传入 a/b/out/aux 指针 | ABI smoke、返回码、OOM/缺DLL |

如果实现结果与此表或第7--10节有冲突，以本计划的固定契约为准；不要根据“常见框架习惯”擅自改语义。
