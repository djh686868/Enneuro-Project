# 卷积/池化反向 Kernel 与 Winograd CUDA 版调研分析

> 本文档为 EnNeuro 框架 CUDA C 算子改写调研的子方向报告，覆盖**卷积反向（gx / gW）、池化反向、Winograd CUDA 版**三个部分。分析基于当前代码库（分支 `give-web-a-try`，`code/eneuro`）与主流框架（cuDNN、PyTorch、CUTLASS、Chainer/CuPy、oneDNN）的公开实现方法，产出**修改模块清单、修改方法与分阶段路线**，供后续实现阶段（综设III）拆分任务使用。

---

## 一、概述

EnNeuro 的 GPU 能力目前完全构建在 **CuPy 数组运算**之上：`Tensor` 包装 numpy/cupy 数组，`Function` 子类实现 `forward/backward`，通过 `get_array_module` / `to_xp` 双后端抽象实现 CPU/GPU 透明切换。全代码库**没有任何自定义 CUDA kernel**（无 `RawKernel`/`RawModule`/`nvcc` 引用），也没有 C++ 扩展构建系统。

本子方向的目标是：为卷积/池化的反向传播链路与 Winograd 卷积引入**手写 CUDA C kernel**，以消除现有 CuPy 数组表达式的三类开销：

1. **kernel launch 开销**：现有实现以"多个小 CuPy 逐元素/跨步切片运算"拼出一步数学变换（如 Winograd 的 BᵀdB 约 8 次逐元素 kernel + 4 次跨步散写），在小尺寸特征图（MNIST 28×28、CIFAR 32×32）下 launch 开销占比显著；
2. **临时张量开销**：反向链路存在大体积中间缓冲（gx 反向的 gcol 可达数百 MB/层，详见 §2.5）；
3. **Python 循环开销**：`im2col_array` / `col2im_array` / `conv2d_backward_input_array` 在 GPU 上仍以 Python 循环驱动 strided 切片（3×3 卷积即 9 次 kernel 调用）。

**技术路线结论**：CUDA C 改写采用 **CuPy `cp.RawModule`（NVRTC 运行时编译）** 作为载体——无需引入 nvcc / 构建链，与团队 Windows 环境及 CLAUDE.md 记录的 CUDA 12.6 驱动兼容性问题解耦（kernel 按设备架构在运行时编译，测试机 RTX 4060 = sm_89）；GEMM 继续走 `cupy.matmul`（即 cuBLAS），不自行实现 GEMM。

---

## 二、现状梳理：涉及代码与数据流

### 2.1 卷积前向：已有 4 条路径 + autotune

`Conv2d`（`code/eneuro/base/functions.py:823-1543`）已具备路径选择器 `_select_forward_path`（`functions.py:852`）、autotune（`functions.py:898`，语义等价于 PyTorch 的 `torch.backends.cudnn.benchmark = True`）与路径缓存 `_path_cache`：

| 路径 | 位置 | 实现方式 |
|---|---|---|
| im2col | `functions.py:1229` | `im2col_array`（Python 循环 KH×KW 次 strided 切片拷贝）+ `tensordot` |
| gemm | `functions.py:1314` | `sliding_window_view` + einsum |
| fft | `functions.py:1262` | `rfftn` / `irfftn`（含 OOM 回退） |
| winograd | `functions.py:1365` | F(2×2,3×3)：BᵀdB 显式公式 → 16 分量批量 GEMM（`matmul(U16, V16T)`，实际走 cuBLAS）→ AᵀMA 显式公式 + 跨步散写 `y_buffer` |

Winograd 前向已实现四类缓存，是 CUDA 版可直接沿用的骨架：

- 变换常量缓存 `_winograd_consts_by_dtype`（`functions.py:1413`）；
- 权重变换缓存 `_winograd_u_cache`（`functions.py:1416`，按数据指针键控，存在隐患，见 §2.5）；
- workspace 缓存 `_winograd_workspace_cache`（`functions.py:1418`，LRU 上限 4 条）；
- 前向 workspace 传反向复用 `_fw_workspace` + 版本校验（`functions.py:1507-1509`），反向侧零拷贝复用 V（`functions.py:1100-1114`）。

该设计正是 cuDNN workspace API（`cudnnGetConvolutionForwardWorkspaceSize`）的思路。

### 2.2 卷积反向：三条梯度路径（改造重点）

`Conv2d.backward`（`functions.py:1009-1040`）：

| 梯度 | 现有实现 | 位置 | 问题 |
|---|---|---|---|
| gx（数据梯度） | forward 走 winograd 时 → `winograd_conv2d_backward`；否则 → `deconv2d`（tensordot → gcol → col2im）；dilation 分支 → `conv2d_backward_input_array` | `functions.py:1042` / `functions.py:1711` / `functions.py:1549` | gcol 临时张量内存爆炸；dilation 分支为 Python 双层循环（KH×KW 次 tensordot + 跨步累加） |
| gW（权重梯度） | `Conv2DGradW`：重新执行一遍 im2col + tensordot | `functions.py:1789` | forward 已构建的 col / V 完全不复用（winograd 反向的 gU einsum 是唯一复用） |
| gb（偏置梯度） | `gys.sum(axis=(0,2,3))` | `functions.py:1039` | 便宜，无需修改 |

`winograd_conv2d_backward`（`functions.py:1042-1227`）结构：复用前向 V → gy 变换为 dM（显式公式 + `gy_buffer` 补零）→ `einsum` 求 gU → S0/S1/S2 显式公式逆变换为 gW → gx 仍走 `tensordot + col2im_array`（未使用 Winograd 域 gx）。

### 2.3 池化：GPU 上最不友好的路径（改造性价比最高）

| 环节 | 现有实现 | 位置 |
|---|---|---|
| MaxPool 前向 | im2col（Python 循环）→ reshape → `argmax(axis=2)` + `max`；索引存 `self.indexes`（int64，(N,C,OH,OW)） | `functions.py:1846-1855` |
| MaxPool 反向 | `xp.zeros(N*C*OH*OW*KH*KW)` 大张量展开 → `gcol[indexes] = gy.ravel()` 平铺散写 → reshape/swapaxes → `col2im_array`（再次 KH×KW 循环） | `functions.py:1873-1897` |
| AvgPool 前向 | im2col + `mean` | `functions.py:1939-1945` |
| AvgPool 反向 | broadcast → `col2im`（经 `Col2im` Function，仍会建图） | `functions.py:1947-1955` |
| 全局平均池化 | mean + broadcast | `functions.py:1961-1971` |

单次 MaxPool 反向约 15+ 次 kernel launch，且伴随 KH·KW 倍膨胀的临时张量。

### 2.4 调用链与集成面

- `Function.__call__`（`core.py:362-390`）：统一入口，含 visualize / 记图钩子；新 kernel 实现在 `forward` 内部替换，不影响调用链。
- `Tensor.backward`（`core.py:207-252`）：梯度按 `x.grad.data + gx.data` 累加（每次全量新分配，可选优化点）。
- `nn/module.py:154-261`：`Conv2d` Layer 仅做参数与分组/深度可分离分发，kernel 分发下沉在 Function 层，Layer 层基本不动。
- `ao/`（自动融合）：`conv_relu` / `conv_bn_relu` 模式（`pattern.py`、`graphoptimizer.py`）替换为 `FusedConvReLU` / `FusedConvBNReLU`，融合算子换实现后节点参数需增加 `impl` 字段。

### 2.5 调研中发现的关联缺陷（建议纳入本子方向修复）

| # | 缺陷 | 位置 | 影响 |
|---|---|---|---|
| 1 | `FusedConvReLU.backward` 写死 `gW = np.tensordot(...)`，且 `im2col_array(...)` 未传 `xp=` | `functions.py:2289-2290` | **GPU 上直接报错**（cupy 数组进入 `np.tensordot`），融合算子接入 CUDA 改写前必须修复 |
| 2 | `_winograd_u_cache` 以数据指针 `w_ptr` 为键 | `functions.py:1459` | CuPy 内存池复用已释放地址：`Parameter.data` 被整体替换（如 `load_state_dict`、设备迁移）后新数组可能拿到旧指针，**用过期的 U 变换权重**；应改为 (Parameter id, 版本号) 键控 |
| 3 | gx 反向 `tensordot(W, gy)` 产生 (C, KH, KW, N, OH, OW) 临时张量 | `functions.py:1206`、`functions.py:1761` | 以 ResNet18 layer1（C=64、N=128、32×32、3×3）估算 ≈ **300 MB/层**瞬时分配；是改用 gather 型 col2im kernel 的最强动机 |
| 4 | `AveragePooling.backward` 中 `KW, KH = pair(...)` 命名互换 | `functions.py:1949` | 非方形核时广播维度错位（方形核下无症状），顺手修正 |

---

## 三、主流框架方法对照

### 3.1 对照表

| 环节 | cuDNN | PyTorch | 对 EnNeuro 的启示 |
|---|---|---|---|
| 前向 conv | implicit GEMM 为主力；Winograd（F(2×2/4×4,3×3)）仅限 3×3 s1 且按启发式选择；FFT 用于大核 | `cudnn_convolution`；`benchmark=True` 即在线 autotune | 现有路径选择器架构与业界一致，CUDA 版只需**换实现层、不动调度层** |
| gx（bwd data） | implicit GEMM / `ALGO_WINOGRAD`（bwd data 存在 Winograd 算法） | `conv_transpose2d` 与 bwd-data 共用 kernel（角色互换复用前向 kernel） | gx 不要走 gcol + col2im；用 **gather 型 kernel** 或复用前向 conv kernel |
| gW（bwd filter） | implicit GEMM + split-K；ALGO_0 用 atomics（非确定），ALGO_1 用 workspace 归约（确定）；枚举中虽有 Winograd 项但实际从不被选中 | `torch.use_deterministic_algorithms` 强制确定性 algo | **gW 保持在空间域 implicit GEMM**，不做 Winograd 域 wgrad；必须提供 split-K（否则并行度只有 OC×C·9 个输出点） |
| Winograd 精度 | fp32 专用；fp16/TF32 下启发式基本不选 Winograd（误差放大） | 同左 | kernel 内变换用 fp32；GEMM 核心可上 TF32/FP16 + fp32 累加，留开关 |
| 带索引 maxpool | cuDNN pooling API 不回传索引 | **自研 CUDA kernel**（fwd 记 argmax；bwd 按输出回写、窗口重叠时 atomicAdd） | 佐证"池化必须自写 kernel"；我们采用更优的 gather 版（无原子、确定性） |
| 生态 | CUTLASS implicit-GEMM 迭代器；Chainer 全部调 `cupyx.cudnn` | — | 先把 `cupyx.cudnn.convolution_*` / `pooling_*` 接成**性能基线对照组**（约半天工作量，为自研 kernel 定标尺） |

### 3.2 三条业界共识（决定本子方向设计）

1. **wgrad 的矛盾是"输出点少、归约维大"**（归约维 = N·OH·OW），cuDNN 为此提供 6 种 bwd-filter 算法，核心手段是 split-K + 原子累加或两段归约；
2. **反传数据（col2im）用 gather 而非 scatter**：对每个输入像素反解覆盖它的 ≤KH·KW 个输出窗口（步长整除检查），直接累加写出——无 atomics、确定性、写合并，并彻底消灭 gcol 临时张量；
3. **Winograd 只服务前向和 gx，gW 留在空间域**；"变换 kernel + 批量 GEMM + 逆变换 kernel"的三 pass 形态是 cuDNN 及多数实现的首选，全融合 kernel（共享内存 + wmma）作为二期优化。

---

## 四、修改模块清单

### 4.1 新增（主体工作量）

```
code/eneuro/base/cuda/            ← 新包，全部 CUDA C 的家
├── __init__.py                   # is_available()、开关读取（env: ENNEURO_CUDA_KERNELS）
├── compiler.py                   # RawModule 编译缓存：key=(kernel名, compute_capability, dtype, options)
│                                 #   cap = cp.cuda.Device().compute_capability → (8,9) → compute_89
├── sources/                      # .cu 源码（字符串常量或文件，NVRTC 友好写法）
│   ├── im2col_col2im.cu          # im2col(带 pad 边界判断，免 pad 拷贝) / gather 型 col2im
│   ├── winograd.cu               # BᵀdB、AᵀMA、dM 变换、B dV Bᵀ、GᵀgG 五个变换 kernel
│   ├── conv_gemm.cu              # implicit GEMM 前向 + wgrad(split-K 两段归约)
│   ├── pool.cu                   # maxpool fwd(argmax) / bwd(gather)
│   └── epilogue.cu               # bias+ReLU 融合尾部（服务 FusedConvReLU）
└── dispatch.py                   # 各路径参数打包 + 失败回退入口
```

### 4.2 修改（按文件）

| 文件:行 | 模块 | 修改内容 |
|---|---|---|
| `functions.py:727-764` | `im2col_array` GPU 分支 | cupy 时改调 im2col kernel（折叠 pad，产出 `(N*OH*OW, C*KH*KW)` 直接可 GEMM 布局）；numpy 路径原样保留 |
| `functions.py:767-817` | `col2im_array` GPU 分支 | 改调 gather 型 col2im kernel（免 pad 大数组、免 KH×KW 循环） |
| `functions.py:852-1005` | `Conv2d` 路径选择/forward | 调度不动；在 `_select_forward_path` 之下加**实现级二级分发**（`winograd → winograd_impl(x,W,b)`），实现级按 (xp is cp 且 kernel 可用) 选 CUDA/NumPy，autotune 候选集合不变 |
| `functions.py:1009-1040` | `Conv2d.backward` | 记录 `self._fw_impl`；gx 改用 gather-col2im（或 winograd bwd-data）；gW 接入 split-K wgrad kernel |
| `functions.py:1042-1227` | `winograd_conv2d_backward` | dM 变换、gW 逆变换换成 kernel；gx 分支换 gather kernel；顺手把 U 缓存键从指针改为 (id, version) |
| `functions.py:1229-1260` | `im2col_conv2d_forward` | 换 im2col kernel + 一次 GEMM；同时把 col 挂进 `_fw_workspace` 供 `Conv2DGradW` 复用（对齐现有 winograd V 复用机制） |
| `functions.py:1549-1583` | `conv2d_backward_input_array` | Python 双循环整体替换为 gather kernel（dilation 作 kernel 参数透传） |
| `functions.py:1789-1824` | `Conv2DGradW` | 优先消费 forward 缓存的 col；未命中则 im2col kernel + split-K wgrad |
| `functions.py:1838-1897` | `Pooling` / `Pooling2DGrad` | 前向：单 kernel（每输出线程循环 KH·KW 求 max+argmax，索引保持 int64 语义兼容）；反向：单 gather kernel 取代"大 gcol 散写 + col2im" |
| `functions.py:1931-1955` | `AveragePooling` | 反向改 scale + gather col2im；修 KW/KH 互换 |
| `functions.py:2229-2308` | `FusedConvReLU` | **先修 `np.tensordot` GPU bug**；前向尾部用 epilogue.cu 融合 bias+ReLU |
| `core.py:243-245` | `Tensor.backward` 累加 | 可选优化：`x.grad.data + gx.data` 每次全量新分配，可换就地加 kernel（低优先级） |
| `nn/module.py:154-261` | `Conv2d` Layer | 基本不动（分发在 Function 层）；仅透传 dtype/device，暴露 `ENNEURO_CONV_IMPL` 环境开关 |
| `ao/pattern.py`、`ao/graphoptimizer.py` | 融合图 | 模式不变；`FusedConvReLU` 换实现后节点参数增加 `impl` 字段 |
| `tests/` + `code/bench_conv_cache.py` | 测试与基准 | 新增正确性/确定性/性能三组（见 §7） |

---

## 五、关键 kernel 设计与修改方法

### 5.1 gather 型 col2im —— 一举解决 gx 反向与池化反向

每个输入像素一个线程，反解覆盖窗口（gather，无原子、确定性），直接读 `gy` 与 `W`：

```cuda
extern "C" __global__ void col2im_gather_nchw(
    const float* __restrict__ gy,       // (N,OC,OH,OW) 即上游梯度
    const float* __restrict__ W,        // (OC,C,KH,KW)
    float* __restrict__ gx,             // (N,C,H,W)
    int N,int C,int H,int Wd,int OC,int OH,int OW,
    int KH,int KW,int SH,int SW,int PH,int PW,int DH,int DW)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * C * H * Wd;
    if (idx >= total) return;
    int iw = idx % Wd, ih = (idx / Wd) % H;
    int ic = (idx / ((long)Wd * H)) % C, n = idx / ((long)Wd * H * C);
    float acc = 0.f;
    for (int kh = 0; kh < KH; ++kh) {
        int oh = ih + PH - kh * DH;              // 反解：oh*SH + kh*DH - PH == ih
        if (oh < 0 || oh % SH) continue;         // 步长不整除 → 该窗口无贡献
        int ohd = oh / SH; if (ohd >= OH) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int ow = iw + PW - kw * DW;
            if (ow < 0 || ow % SW) continue;
            int owd = ow / SW; if (owd >= OW) continue;
            for (int oc = 0; oc < OC; ++oc)
                acc += gy[n*OC*OH*OW + oc*OH*OW + ohd*OW + owd]
                     * W [oc*C*KH*KW + ic*KH*KW + kh*KW + kw];
        }
    }
    gx[idx] = acc;
}
```

要点：

- **不再构造 gcol**：内存从 ~300 MB/层（缺陷 #3）降到 gx 本身；
- 归约维 OC·KH·KW 对 GPU 并行友好（输出线程数 = N·C·H·W），`oc` 内循环连续访问 gy，二期可用 shared memory 缓存 W tile；
- **三处共用**：`Deconv2d.forward`（即 conv 的 gx 路径）、`conv2d_backward_input_array`（dilation 分支）、winograd 反向 gx——本子方向复用率最高的 kernel；
- pad 折叠进边界判断，不再分配 padded 大数组（对比 `col2im_array` 的 `xp.zeros(H+2PH+SH-1, ...)`）。

### 5.2 池化前向/反向（对照 PyTorch，反向采用 gather 版）

```cuda
// 反向：一线程一输入，检查覆盖窗口的 argmax 是否指回自己
extern "C" __global__ void maxpool_bwd(
    const long long* __restrict__ idxs,   // (N,C,OH,OW)，沿用现有 int64 语义
    const float* __restrict__ gy, float* __restrict__ gx,
    int N,int C,int H,int Wd,int OH,int OW,int KH,int KW,int SH,int SW,int PH,int PW)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)N*C*H*Wd) return;
    int iw = idx % Wd, ih = (idx / Wd) % H, c = (idx / ((long)Wd*H)) % C, n = idx / ((long)Wd*H*C);
    float acc = 0.f;
    for (int kh = 0; kh < KH; ++kh) {
        int oh = ih + PH - kh;  if (oh < 0 || oh % SH) continue;  int ohd = oh / SH;
        if (ohd >= OH) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int ow = iw + PW - kw;  if (ow < 0 || ow % SW) continue;  int owd = ow / SW;
            if (owd >= OW) continue;
            int kk = kh * KW + kw;
            long long oidx = ((long long)n * C + c) * OH * OW + ohd * OW + owd;
            if (idxs[oidx] == kk) acc += gy[oidx];  // argmax 平局取首个，与 numpy argmax 语义一致
        }
    }
    gx[idx] = acc;
}
```

要点：

- 与 PyTorch 的 atomicAdd 版相比：**无竞争、两次运行 bitwise 一致**——现有测试框架按 numpy 对拍，确定性是硬需求；
- 2×2 s2（LeNet/VGG 全部场景）每输入恰被 1 个窗口覆盖，gather 退化为纯直写，零额外开销；
- `self.indexes` 语义与 dtype 不变 → `Pooling2DWithIndexes`（二阶梯度路径）零改动；
- 前向 kernel：一线程一输出 `(n,c,oh,ow)`，循环 KH·KW 求 max + argmax，一次写出 y 与 idxs。

### 5.3 Winograd CUDA 版（三 pass 形态，两期走）

**一期（推荐先做）**：保留现有"变换 → GEMM → 逆变换"结构与全部缓存/workspace 机制，仅把 5 个变换阶段换成 kernel：

| 阶段 | 现状（NumPy/CuPy） | CUDA 一期 |
|---|---|---|
| 输入变换 V = BᵀdB | ~8 个逐元素 kernel + `sliding_window_view` | 1 个 kernel：线程一 tile，共享内存装 4×4 窗口，寄存器完成 16 次加减法，直接写出 (16, C, N·T) 布局对齐现有 matmul |
| 16 分量 GEMM | `matmul(U16, V16T)`（已走 cuBLAS） | **保留不动** |
| 输出逆变换 AᵀMA + 散写 | ~8 kernel + 4 次跨步散写 + `y_buffer` | 1 个 kernel：线程一 tile 写 2×2 输出，边界掩码直写 y（**消灭 y_buffer**） |
| 反向 dM 变换（gy → dM） | 显式公式 + `gy_buffer` 补零 | 1 个 kernel：线程一 tile，奇数 out 尺寸用边界判断置零（**消灭 gy_buffer**） |
| 反向 gW 逆变换（gU → gW） | S0/S1/S2 显式公式（OC×C，量小） | 1 个小 kernel，或保留 NumPy |

配套修改：

- `winograd_conv2d_backward` 的 gU einsum（`functions.py:1186`）先保留（cupy einsum 与 strided-batch GEMM 本质相同），二期再迁到空间域；
- `_fw_workspace` 复用机制原样保留——CUDA 版的 V 同样可被反向零拷贝复用；
- **二期**：升级 F(4×4,3×3)（每输出 MAC 从 4× 降到 2.25×，cuDNN 在大图上的主力 Winograd 形态）；GEMM 核心用 wmma / TF32（fp32 累加），变换保持 fp32；
- **明确不做**：Winograd 域 wgrad（业界不选，误差与收益都不划算）。

### 5.4 wgrad：implicit GEMM + 确定性 split-K（对照 cuDNN ALGO_0/ALGO_1）

- **v1（正确性优先，1–2 天）**：每线程一 `(oc, c, kh, kw)`，沿 N·OH·OW 归约。并行度仅 OC·C·9，仅适合小层；但相比现状（重跑 im2col + 全量 tensordot）已省一半开销；
- **v2（推荐落地）**：block tile (OC_tile × C·9_tile) + 共享内存缓存 gy / x_pad tile，沿 N·OH·OW 循环；部分和写 workspace，第二个 kernel 归约——即 cuDNN ALGO_1 形态，**确定性**。

必须配 split-K/两段归约，否则输出并行度不足（这正是 cuDNN 为 bwd filter 准备 6 种算法的原因）。同时为 `Conv2DGradW` 增加前向 col 复用：forward 的 `_fw_workspace` 挂载 col（对齐 winograd V 的复用模式），反向零拷贝消费。

### 5.5 编译器封装（对接全框架的统一入口）

```python
# eneuro/base/cuda/compiler.py
import cupy as cp
_cache = {}

def get_kernel(name: str, code: str, options=()):
    major, minor = cp.cuda.Device().compute_capability          # RTX 4060 → (8, 9)
    key = (name, (major, minor), options)
    if key not in _cache:
        mod = cp.RawModule(code=code, backend='nvrtc',
                           options=(f'--gpu-architecture=compute_{major}{minor}', *options))
        _cache[key] = mod.get_function(name)
    return _cache[key]
```

- NVRTC 后端 → 团队 Windows 机器**无需安装 nvcc**；编译开销（百 ms 级）按 kernel 首次调用摊销，可选 cubin 落盘；
- 所有 kernel 入口统一 try/except → 回退现有 NumPy/CuPy 路径（与 fft 路径的 OOM 回退 `functions.py:1299-1302` 同一模式）。

---

## 六、数值精度、确定性与缓存安全

1. **Winograd fp32 误差放大**是已知问题（cuDNN 因此在 fp16 下不选 Winograd）：对拍容差按"im2col 路径 vs winograd 路径"分层设定（单层 rtol ~1e-3，深层累积放宽）；TF32/fp16 仅允许进 GEMM 核心，禁止进变换；
2. **确定性**：全部采用 gather 写法，wgrad 采用 workspace 归约 → 端到端 bitwise 可复现；autotune 计时建议从 `perf_counter` 换成 `cp.cuda.Event`（`functions.py:942-953`），降低计时噪声；
3. **缓存键安全**：`_winograd_u_cache` 的指针键（缺陷 #2）必须修复；workspace 缓存的 LRU 上限机制（max 4/2 条）可沿用，kernel 版 workspace 仅增加 (N, OC, tile, dim, dtype) 维度。

---

## 七、测试与基准计划（对齐现有资产）

| 类型 | 内容 | 挂靠 |
|---|---|---|
| 正确性 | 各 kernel vs 现有 NumPy 路径逐元素对拍；尺寸覆盖：奇数 out、pad>0、非方形核、dilation、groups、fp16/fp64 | 新增 `code/tests/test_cuda_conv_backward.py`、`code/tests/test_cuda_pooling.py`（风格参照 `tests/test_advanced_convolutions.py`、`code/test_winograd_3x3_stride1.py`） |
| 确定性 | 同输入跑 2 次 bitwise 相等；反向梯度与数值差分对拍 | 同上 |
| 性能 | `bench_conv_cache.py` 扩展：CUDA kernel 路径 vs NumPy 路径 vs `cupyx.cudnn` 基线，三类形状（LeNet 28×28 / VGG / ResNet18 32×32），统计 kernel launch 次数（nvtx） | `code/bench_conv_cache.py` |
| 端到端 | `trainer.py` 全流程 GPU 训练不回归（`tests/test_donkey/test_resnet18.py` 等现有脚本回归跑通） | 现有脚本 |

---

## 八、分阶段路线与工作量估计

| 阶段 | 内容 | 预估 | 收益 |
|---|---|---|---|
| P0 | `cupyx.cudnn` 接成对照基线 + 修 `FusedConvReLU` GPU bug（`functions.py:2289-2290`）+ 修 U 缓存指针键 | 0.5–1 天 | 有标尺；排雷 |
| P1 | im2col/col2im + 池化 fwd/bwd kernel（§5.1/5.2），接入 `im2col_array`/`col2im_array`/`Pooling*` | 2–3 天 | **性价比最高**：消灭最多 launch 与最大临时张量，池化反向预计 2–5× |
| P2 | Winograd 三 pass CUDA 一期 + gather 型 gx 全线替换 + wgrad v1/v2，接入 `Conv2d.backward` | 3–5 天 | 反向链路完整 CUDA 化 |
| P3 | bias+ReLU epilogue、F(4×4,3×3)、wmma GEMM、cubin 落盘 | 3–5 天 | 二期优化 |

---

## 九、风险清单

| # | 风险 | 缓解 |
|---|---|---|
| 1 | NVRTC 语法限制：不能 `#include` 头文件、模板受限 | kernel 用 C 风格 + 宏参数化（dtype trait 手写 float/double 双版本，或用 CuPy 的 dtype→宏注入） |
| 2 | 自写 GEMM 打不过 cuBLAS | GEMM 继续走 `cupy.matmul`（cuBLAS），只自写变换/反传/池化 kernel——cuDNN 亦是如此分工 |
| 3 | 路径缓存键遗漏实现维度 | `_path_cache_key`（`functions.py:886`）追加 `impl`（cuda/numpy），避免 autotune 结果跨实现串用 |
| 4 | 设备推断与掩码残留 | kernel 输出 cupy 数组经 `as_Tensor` 自动得 `'cuda'`，core 无需改动；`FusedConvReLU` 的 `self.mask` 确认随实现切换仍在同一设备 |
| 5 | CUDA 驱动兼容（CLAUDE.md 记录的 12.6 切换问题） | kernel 运行时按设备架构编译，与工具包版本解耦；所有 kernel 入口保留 NumPy 回退路径 |

---

## 附：本报告引用的关键代码位置索引

| 主题 | 位置 |
|---|---|
| 路径选择 / autotune | `code/eneuro/base/functions.py:852`、`functions.py:898` |
| Winograd 前向 / 反向 | `functions.py:1365`、`functions.py:1042` |
| im2col / col2im | `functions.py:672`、`functions.py:767` |
| gx 反向（dilation） | `functions.py:1549` |
| gW 反向 | `functions.py:1789` |
| 池化前向 / 反向 | `functions.py:1838`、`functions.py:1861`、`functions.py:1931` |
| 融合算子 | `functions.py:2229`、`functions.py:2313` |
| 双后端抽象 | `code/eneuro/base/core.py:17`、`core.py:341` |
| Layer 层卷积 | `code/eneuro/nn/module.py:154` |
