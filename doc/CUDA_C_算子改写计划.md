# EnNeuro CUDA C 算子改写实施计划（GPT-5.6 Sol 交接版）

版本：v2，2026-09-09。本文完整替代此前追加式草案；配套阅读 [技术原理](CUDA_C_算子技术原理.md)。

> 当前交付仅为文档，尚未授权开始编码。将来用户明确授权实现后，Sol 按下面的阶段顺序编码、验证和交付。不能把存在设计文档、测试被跳过或执行了 CuPy 回退视为 CUDA C 改写已经完成。

>   【Claude 补充索引 · 2026-09-09 · 非原文】  
> 应用户要求，本次基于 `code/eneuro` 实际代码逐条核对后，以带 `【Claude 补充】` 标记的引用块补充以下内容；未改动任何原文。授权编码前建议逐条确认采纳与否：采纳的去除标记并入正文，未采纳的直接删除对应引用块即可。
> 索引：C1 同进程 `Optimizer._state` 污染防护（§8.2 末）；C2 梯度累加接入方式写死（§6.4 末）；C3 6D 布局与 DLL ABI 闭合（§7 op 表后）；C4 `--cudart=static` 备选预案（§7 末）；C5 create\_graph 分路信号（§6.4 末）；C6 S0 编译工具链预探测（§9.4）；C7 空 batch 行为收紧定性（§5.1 末）；C8 im2col dilation 对照基线（§9.1 末）；C9 CrossEntropyLoss one-hot 分支（§2 表后）；C10 勘误记录（文末）。

## 1. 目标、范围与仓库依据

保留 Python 自动求导和 CuPy 显存管理，新增可选择的 CUDA C 计算后端。最终完成内置 LeNet 的 MNIST 训练/测试，以及 CuPy、RawModule、DLL 三方数值和性能对比。DLL 仍借用 CuPy 管理数组、执行 cuBLAS；本阶段不移除 CuPy 依赖。

已核对的代码位置（实施前重新检查当前版本）：

| 位置                              | 当前实现与允许修改点                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `code/eneuro/base/core.py`      | Tensor、Function；Add/Sub/Mul/Div/Neg/Pow/Exp/Square。在对应算子内部接入 dispatch；公共接口保持不变  |
| `code/eneuro/base/functions.py` | 另一套 Exp、Log、Sigmoid、ReLU、MatMul、Linear、im2col/col2im、Conv2d、Pooling；两处 Exp 均需覆盖 |
| `code/eneuro/nn/loss.py`        | CrossEntropyLoss 使用稳定 log-softmax；主实验使用该类，不使用带 epsilon 的 SoftmaxWithLoss        |
| `code/eneuro/nn/module.py`      | 内置 LeNet：conv1 padding=2、conv2 padding=0、全部隐藏激活 ReLU、参数部分延迟初始化                  |
| `code/eneuro/nn/optim.py`       | 优化器用 CuPy 表达式；存在类属性 `_state`，不同实验必须进程隔离，避免共享状态污染                                |
| `code/eneuro/train/trainer.py`  | 已有训练入口；部分回调把验证值同时填入训练值，正式报告使用独立实验循环计算指标                                         |

已有 `code/tests/test_lenet.py` 定义另一套 LeNet（padding/激活不同），不可拿它替代主实验。已有 tests 中含脚本、外部框架和数据依赖，不要求不加区分地对整仓库执行 pytest。

不做：Winograd/FFT、分组/深度卷积、独立转置卷积 API、平均池化/BatchNorm、图融合优化器、FP16/FP64 kernel、自研 GEMM、优化器 CUDA kernel、训练服务器、无 CuPy Tensor 引擎。只修复阻碍本阶段数值正确性的具体问题，不做全仓库重构。

## 2. 必须交付的算子清单

“CUDA C”表示确实有手写设备代码；“组合”表示调用自研 kernel 加明确保留的 CuPy 运算。不是每个数学反向公式都需要单独编译一个同名 kernel。

| 算子                             | 本阶段计算实现                                   | 反向与接入要求                                      |
| ------------------------------ | ----------------------------------------- | -------------------------------------------- |
| add、sub                        | CUDA C 二元逐元素                              | gy / -gy，广播梯度用 sum\_to                       |
| mul、div                        | CUDA C 二元逐元素                              | 先在广播后的完整形状计算局部梯度，再分别 sum\_to 回原输入形状          |
| neg                            | CUDA C 一元逐元素                              | -gy                                          |
| pow、square                     | CUDA C 标量幂；square 调用 pow(x,2)             | c\*x^(c-1))gy；c=0 时直接零梯度，避免 0)x^-1           |
| exp                            | CUDA C expf；覆盖 core/functions 两个入口        | exp(x)\*gy                                   |
| log                            | CUDA C logf                               | gy/x                                         |
| relu                           | CUDA C 前向与专用 backward                     | x>0 时传递 gy；零点梯度为0                            |
| sigmoid                        | CUDA C；沿用现有 tanh(x/2)/2+1/2 公式            | gyyyy(1-y)，调用基础 kernel 组合                    |
| bias\_add                      | CUDA C，支持二维 Linear 和四维 NCHW               | gx=gy；gb 用 CuPy sum                          |
| im2col、col2im                  | CUDA C 展开与 gather 折回                      | 两者为互相转置的线性映射；支持 matrix/6D 两种现有 API 布局        |
| Conv2d forward                 | 自研 im2col + CuPy matmul/cuBLAS + bias     | 所有普通卷积测试形状均可走新路径                             |
| Conv2d backward gx             | CUDA C gather，直接计算输入梯度                    | 不生成 gcol、不用 atomicAdd                        |
| Conv2d backward gW/gb          | 重用本次 forward 的 col + cuBLAS；gb 用 CuPy sum | 不声明 gW/gb 归约为自研 kernel                       |
| MaxPool forward/backward       | CUDA C 窗口扫描、索引保存、输入梯度 gather              | 不能把 Python/CuPy scatter 循环冒充 CUDA C backward |
| Linear/MatMul                  | GEMM 保留 cuBLAS；bias 用自研 kernel            | gx=<gy@W.T>；gW=x.T\@gy；gb=sum(gy,0)          |
| Softmax、CrossEntropyLoss       | 稳定组合：CuPy max/sum/标签索引 + 自研 exp/log/基础运算  | 含上游标量 gy，标签无梯度，batch mean                    |
| sum/mean/sum\_to、broadcast\_to | 保留 CuPy 实现                                | 广播归约是保留项，不能遗漏                                |
| reshape/transpose/flatten      | 保留数组视图/必要复制                               | 不另写无计算意义的 kernel                             |

>   【Claude 补充 C9 · 2026-09-09 · 非原文】CrossEntropyLoss 的 one-hot 分支一并接入  
> 上表仅点名标签索引形式；`CrossEntropyLoss` 另有 one-hot 分支（loss.py:150-152 前向、loss.py:168 反向，`t.ndim==2`）。主实验只用标签索引，但既然接入该类，one-hot 分支应走同一条“CuPy max/sum + 自研 exp/log”组合路径，测试矩阵补一条 one-hot 小用例，避免严格模式下同一算子出现两套路径。

基础反向继续由现有 Function 和 Tensor 运算组合完成；只有 ReLU、池化、卷积采用专用数组级反向。只验收一阶反向。`create_graph=True` 时保留旧路径，不宣称修复现有框架的高阶求导。

## 3. 两条路线的严格先后

1.   快速验证路线  ：所有 CUDA C 源码先写成 `raw_kernels.py` 中的原始字符串，`cupy.RawModule(..., backend='nvrtc')` 编译，Python 发射 kernel。先完成全部算子和 LeNet 数值、训练、基准报告。
2.   正式交付路线  ：快速路线正确性门槛通过且报告已生成后，才把设备源码迁至 `.cu`，nvcc 编译 DLL，ctypes 调用；再重复同一验收。不得只提供 DLL 骨架并称为完成。
3.   统一源码  ：第二阶段以 `sources/kernels.cu` 为唯一设备代码源。RawModule 也读取该文件；`raw_kernels.py` 改为加载器。DLL 的 `library.cu` 包含这份代码并提供 C ABI wrapper，避免维护两份不同公式。

快速路线不调用 nvcc，但仍需要可用的 CuPy、NVRTC、驱动及匹配的运行时/头文件。DLL 路线额外需要 CUDA Toolkit 和兼容的 MSVC x64 编译器。

## 4. 文件职责与运行配置

将来编码时新增：

```text
code/eneuro/base/cuda/
  __init__.py       公共后端配置和数组 API 导出
  compiler.py       RawModule 延迟编译/缓存
  raw_kernels.py    路线一内嵌源码；路线二读取 kernels.cu
  dispatch.py       参数检查、后端选择、CuPy参考与诊断
  extension.py     路线二 ctypes/DLL 加载
  sources/kernels.cu   路线二唯一设备源码
  sources/library.cu   路线二 host launcher
  sources/api.h        路线二 ABI 声明
  build_cuda.ps1       路线二构建脚本
  bin/                 本机构建产物，忽略进版本控制
code/tests/test_cuda_kernels.py
code/tests/test_cuda_lenet.py
code/tests/test_cuda_backend_cpu.py
code/bench_cuda_lenet.py
requirements-cuda.txt
pytest.ini             已存在时合并 cuda 标记，不能覆盖其他配置
```

不新增第二套模型。算子源码只放 cuda 包，现有 core/functions/loss 只加接入代码；避免循环导入，cuda 包不能在模块顶层反向 import core/functions。

- 默认 `ENNEURO_CUDA_BACKEND=cupy`，保持旧行为；显式测试 rawmodule/extension。
- `auto` 依次选择 DLL、RawModule、CuPy。`rawmodule` 和 `extension` 是严格模式：支持范围内缺 kernel、编译/加载失败直接报错，以免验收实际测了回退。
- 不支持的 dtype/广播等在 auto 中回退；严格模式下报 NotImplementedError。单独的 fallback 测试使用 auto。
- 环境只在首次导入时读取；`set_backend(mode)` 随后优先，`using_backend(mode)` 用 contextvars 实现可恢复的上下文配置，不在一次 forward/backward 中途切换。
- `get_backend()` 返回请求模式。`is_available(backend='rawmodule')` 做显式懒探测；import 本身不初始化 CUDA。CuPy/GPU/NVRTC/DLL 探测分别写入 diagnostics。
- `ENNEURO_CUDA_SYNC=1` 用于调试同步，正式性能测量关闭。debug 同步针对当前 stream。
- 使用当前设备和 `cp.cuda.get_current_stream()`；不同设备数组混用报错，不暗中搬迁。

## 5. 数组 API 与语义

以下为新增 API 规格，保留原框架公共接口。数组通常为 CuPy float32；索引/标签例外明确如下。

```python
set_backend(mode: str) -> None
get_backend() -> str
using_backend(mode: str)  # 上下文管理器
is_available(backend='rawmodule') -> bool
get_diagnostics(reset=False) -> dict

add_forward(a,b); sub_forward(a,b); mul_forward(a,b); div_forward(a,b)
neg_forward(x); pow_forward(x, exponent); exp_forward(x); log_forward(x)
relu_forward(x); relu_backward(x, gy); sigmoid_forward(x)
bias_add_forward(x, bias)  # (N,F)+(F,) 或 (N,C,H,W)+(C,)
im2col_forward(x, kernel, stride=1, pad=0, dilation=1, to_matrix=True)
col2im_forward(col, input_shape, kernel, stride=1, pad=0, dilation=1, to_matrix=True)
conv2d_forward(x,w,b=None,stride=1,pad=0,dilation=1,return_context=False)
# 默认返回 y；return_context=True 返回 (y, context)
conv2d_backward(gy,x,w,b=None,stride=1,pad=0,dilation=1,context=None)
# 返回 (gx,gW,gb)，b=None 时 gb=None
maxpool_forward(x,kernel,stride=1,pad=0)  # 返回(y,indexes)
maxpool_backward(gy,indexes,input_shape,kernel,stride=1,pad=0)
```

### 5.1 dtype、标量、广播与布局

- CUDA C 数值 kernel 只计算 float32；不能把 float64 隐式降为 float32。布尔掩码/标签不属于普通算术 kernel 的 float32 输入。
- 二元 kernel 支持同 shape，以及任一操作数为标量（Python/NumPy 数值，或 size=1 的 CuPy 数组）。host 标量按现有 float32 运算语义传值；device 标量读指针0，禁止为取值调用 item/asnumpy。
- 其他合法广播保留 CuPy fallback；非法广播仍抛 ValueError。pow 的指数仅支持常量标量，不计算指数梯度；兼容现有 Pow wrapper 传入的零维 device 标量。
- 非连续输入通过 `cp.ascontiguousarray` 在设备端标准化，计入算子耗时和复制次数；正常输出 C-contiguous。该规则适用于 Linear 的转置视图和卷积的 transpose 输出。
- 空数组在元素级操作返回同形空输出、不 launch。空 batch 卷积、损失及非正 OH/OW 报 ValueError。
- rank 不限制基础元素操作；卷积/池化仅 NCHW。维度、stride、dilation、kernel 必须正整数，pad 必须非负。普通卷积仅 groups=1；group/depthwise API 保留旧路径。

>   【Claude 补充 C7 · 2026-09-09 · 非原文】空数组行为定性：显式收紧，非兼容项  
> 上文“空 batch 卷积、损失及非正 OH/OW 报 ValueError”是对现状的变更（当前 `im2col_array` 对 N=0 可产出空结果并继续执行），与 §5.2“数学语义不得更改”表面冲突。定性为：数值语义不变，输入域校验收紧；验收时 CuPy 基线对空 batch 的旧行为不参与对比，并在报告中将该差异记入 behavior\_change 清单。

### 5.2 数学语义不得更改

- 卷积是 cross-correlation，不翻卷积核；越界输入值为0。
- im2col 行按 `(n,oh,ow)`、列按 `(c,kh,kw)` 排列，kw 最快；矩阵形状 `(N*OH*OW,C*KH*KW)`。`to_matrix=False` 返回 `(N,C,KH,KW,OH,OW)`，转换在设备上完成。
-   MaxPool padding 为0，严格兼容当前 im2col 零填充基线。   不改为负无穷；负数边界窗口可能由 padding 的0胜出。索引 `kh*KW+kw`，int64，首次最大值；全 padding 窗口输出0、index=0，反向不给真实输入传梯度。
- ReLU 零点梯度0。普通正确性用有限输入；特殊 NaN/Inf 用 NumPy/CuPy oracle 单独测试相同 mask 和位置，不用 allclose 掩盖意外 NaN。MaxPool 以首次 NaN 和首次最大值对齐 oracle。
- 梯度归约固定先算逐元素导数再 sum\_to；尤其核查当前 Mul/Div 先归约再乘除造成的广播错误。这是必要的正确性修复，分别记录原始基线与修复后 CuPy基线，不能要求 CUDA 重现错误。

## 6. 算法、缓存与集成

### 6.1 算法固定

使用一维 grid、256线程/block和 int64\_t 索引的 grid-stride loop。所有 host 整数显式转 np.int32/np.int64，不依赖 Python int 推断；Windows 上不能用 C long 代替 int64\_t。没有元素时不 launch。

卷积：令 A=im2col(x)、B=W\.reshape(OC,K)。forward 为 `A @ B.T`，reshape/transpose 后复制到连续 NCHW，再 bias\_add。令 G=gy.transpose(0,2,3,1).reshape(M,OC)，gW=`G.T @ A` reshape 回 W；gb=CuPy sum(gy,(0,2,3))；gx 用专用 gather 累加合法 gy\*W。cuBLAS、reduction 均使用 float32，首版不启用 fast\_math/混合精度。

Linear 保留原来的 activation 位置：只用 CUDA bias，不把无激活的 Linear 改成 Linear+ReLU，也不改图优化器。Softmax/交叉熵先减每行最大值，复用 exp/log 和 sum，不在模型末尾额外加 Softmax。

### 6.2 workspace 与 stream

forward 的 col 放在所属 Function 的 context 中，供该次 backward 复用；无梯度模式不保留。禁止按 shape 全局缓存激活内容，防止两次 forward、参数变化或不同 stream 覆写。缺 context 时重算 col；完成 backward/图释放时释放引用。上下文存输入 shape/dtype/device/stride/pad/dilation及实际后端，不用裸地址推断数据没变。

只全局缓存编译模块（SHA256源码、编译选项、CuPy/NVRTC版本、设备ID/架构、dtype、进程），不缓存训练 Tensor/Function。CuPy memory pool负责内存复用。进程与 CUDA context 重建后丢弃编译缓存。

DLL 不分配、释放或缓存设备数组。标准化临时数组至少保留到当前 stream 中完成消费；同 stream 顺序管理，多 stream 显式事件依赖。OOM 报资源错误并终止当前步骤，不盲目重跑更大 CuPy gcol 路径。

### 6.3 错误策略

仅在 launch 前发现 DLL缺失、ABI不符、NVRTC不可用等“后端不可用”时，auto 才允许逐级回退，并缓存本进程失效原因。非法参数直接报错。非法访存、异步执行失败、未知运行异常直接传播并标记失败，不在可能已损坏的 CUDA context 中重新执行 CuPy。

诊断至少累计 `op/requested_backend/actual_backend/kernel_count/copy_count/fallback_reason/error`；CuPy/cuBLAS保留部分单独标识 shared\_cupy。严格 LeNet 的自研算子 fallback\_count 必须为0。

### 6.4 接入顺序与必要修复

先测试独立 cuda API，再接 core 的元素 Function、functions 的 ReLU/Pooling/Conv/Linear及loss。Conv 新路径在旧 Winograd/FFT autotune 前分流；强制 cupy 时不能经辅助函数绕回自研 kernel。

处理无 bias：Function...call.. 可能将 None 包成 Tensor(None)，backward 应检查 `b.data is None`。所有梯度包装成 `as_Tensor`，不直接返回 ndarray。不重新设计整个自动求导引擎；只修改本阶段所需的梯度累加计算路径，使核心 `x.grad.data + gx.data` 在启用后端时也能调用 add，并保留既有图释放行为。

>   【Claude 补充 C2 · 2026-09-09 · 非原文】梯度累加接入方式写死（细化上段“也能调用 add”一句）  
> `x.grad.data + gx.data`（core.py:245）在启用后端时必须走走走数组级直调走走（dispatch 到 add 的 CUDA C kernel，语义等价 add\_forward），，，不得，，走 Function `add()`：后者会在 backward 执行中途创建新节点、抬升 generation，破坏既有图释放行为。即此处“调用 add”仅指后端数组 API：不建图、不登记 creator/outputs、不改 Config。backward 路径其余裸数组归约（如 Linear.backward 的 `gb = gys.sum(axis=0)`，functions.py:459）按 §2 归约保留项处理，不强行替换。

>   【Claude 补充 C5 · 2026-09-09 · 非原文】create\_graph 分路信号明确化  
> “create\_graph=True 时保留旧路径”（§2）的判定信号直接使用 backward 执行时的 `Config.enable_backprop`：Tensor.backward 在 `Config.using_config('enable_backprop', create_graph)` 内调用各 Function.backward（core.py:231），故一阶反向执行时该值为 False（走新路径），create\_graph=True 的反向中该值为 True（走旧路径）。不得为此另设全局开关、实例标志或额外传参。

## 7. DLL 构建与 ABI（快速路线完成后执行）

目标 Windows x64；使用 `ctypes.CDLL`、C ABI、`extern "C" __declspec(dllexport)`。采用一个固定通用 launcher，避免每条路线不同参数顺序：

```c
int32_t eneuro_abi_version(void); // 固定返回1
int32_t eneuro_launch_v1(int32_t op, const void* a, const void* b,
    void* out, void* aux, const int64_t* meta, int32_t meta_len,
    float scalar, void* stream);
const char* eneuro_error_string(int32_t status);
```

a/b/out/aux 为 device pointer；meta 是 host int64 数组，wrapper 在函数返回前读出并按值传给 kernel，不把 meta host pointer 当作 device pointer。stream 必须传 `cp.cuda.get_current_stream().ptr`；0仅表示CUDA null stream，不表示任意“当前stream”。

ctypes 固定 argtypes：int32、四个c\_void\_p、POINTER(c\_int64)、int32、c\_float、c\_void\_p；restype=c\_int32。Windows pointer不能截断成int32；argmax用int64\_t。wrapper 不抛跨ABI异常，不同步正常执行，发射后检查即时 launch status；异步错误由显式同步处报告。

op/meta 协议（数组长度和值先由Python验证）：

| op                                         | meta及指针                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| 0 neg / 1 exp / 2 log / 3 relu / 4 sigmoid | meta=\[count]，a输入，out输出                                                 |
| 5 pow                                      | meta=\[count,exponent\_on\_device]，host指数用scalar，device指数用b\[0]         |
| 10 add / 11 sub / 12 mul / 13 div          | meta=\[count,a\_scalar,b\_scalar]，a/b均为device buffer（host标量包装1元素buffer） |
| 14 relu\_backward                          | meta=\[count]，a=x,b=gy,out=gx                                           |
| 20 bias                                    | meta=\[N,C,spatial]，二维spatial=1，a=x,b=bias                              |
| 30 im2col / 31 col2im                      | meta=\[N,C,H,W,KH,KW,OH,OW,SH,SW,PH,PW,DH,DW]，a输入,out输出；统一matrix布局      |
| 32 conv\_gx                                | 上述meta后追加OC，a=gy,b=W,out=gx                                             |
| 40 pool\_forward                           | 卷积meta的前12项，a=x,out=y,aux=indexes                                       |
| 41 pool\_backward                          | 同40，a=gy,b=indexes,out=gx                                               |

所有调用不使用的指针传NULL。status：0成功、1非法参数、2 ABI/op不支持、3 CUDA launch错误；error\_string给出文本，CUDA错误细节同时在报告保存。严格模式任意非零都失败，不能伪装成成功。

>   【Claude 补充 C3 · 2026-09-09 · 非原文】6D 布局与 DLL ABI 的闭合（消解 §2/§5.2 与本节“统一matrix布局”的表述冲突）  
> 定死如下：设备 kernel 只读写 matrix 布局（`(N*OH*OW, C*KH*KW)` 连续），ABI 不新增 layout 字段。6D 请求统一实现为：im2col 的输出由 matrix 缓冲 reinterpret 为 `(N,OH,OW,C,KH,KW)` 后经 CuPy transpose 至 `(N,C,KH,KW,OH,OW)`；col2im 的 6D 输入先经反向 transpose 归位 matrix 再进 kernel。该 transpose 为 CuPy 发射的设备端 copy kernel（即 §5.2“转换在设备上完成”），计入 copy\_count。RawModule 路线采用同一方案以保持两路线一致，不另写 6D 直写 kernel；extension 严格模式下 6D 因此可用，不触发回退。

`build_cuda.ps1 -Arch sm_XX -CudaPath ...`：未指定Arch时从CuPy设备读取；验证CUDA路径、nvcc和MSVC x64工具链。已有VS环境可直接用，否则由vswhere定位VsDevCmd；找不到时给出具体缺失项。构建 `library.cu`，包含 `kernels.cu`，使用 `-O2 --std=c++14 -shared -Xcompiler /MD --cudart=static` 和当前架构gencode，禁用fast\_math。输出 `bin/enneuro_cuda_smXX.dll` 及build\_manifest.json（ABI、源码hash、工具版本、架构、命令）。优先显式 `ENNEURO_CUDA_DLL`，否则bin对应架构；不全盘搜索DLL。

>   【Claude 补充 C4 · 2026-09-09 · 非原文】`--cudart=static`     与 CuPy 共存的备选预案  
> 静态 cudart 后，进程内并存 CuPy 的 `cudart64_12x.dll` 与 DLL 内嵌 runtime：二者共享驱动层 primary context，但 per-thread 错误状态互不共享。“DLL 不分配/释放设备数组”的原文约束已避开最大风险，裸指针 kernel launch 通常可行。若联调出现难以解释的 context/错误状态异常，按序降级：(a) 改动态链接 cudart（去掉 `--cudart=static`，确保 cudart64\_12x.dll 可被加载）；(b) 改 driver API（`cuLaunchKernel`），彻底脱离 cudart。两种备选均不改动 ABI 与 kernels.cu；实际采用哪条须写入 build\_manifest.json 与报告。

## 8. 固定模型、数据与实验

### 8.1 环境与数据

实施时检查git状态，保留已有用户改动；优先当前项目EnNeuro环境，记录解释器绝对路径。不因另一个Python缺CuPy就断言本机不可用。`requirements-cuda.txt` 声明 CuPy CUDA12x与pytest为可选开发依赖，记录实际通过验证的精确版本，不强制升级已工作的环境。

按AGENTS要求，CUDA新版本API失败时在当前PowerShell切换12.6再执行相同命令：

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH = "$env:CUDA_PATH\bin;" + $env:PATH
```

数据默认 `code/tests/testdata/MNIST_data/mnist.pkl`，CLI可用 `--data` 覆盖。使用原 train/test 划分，禁止合并后随机拆分。支持字典train\_img/train\_label/test\_img/test\_label（或x\_train/y\_train对应键）、两元组、三元组格式。图片支持(N,784)/(N,28,28)/(N,1,28,28)；检查范围：整数或max>1且<=255时除255，已经\[0,1]的float不再次除255；其他范围报错。标签转int32并检查0..9。

### 8.2 初始化与训练配置

主模型固定 `eneuro.nn.module.LeNet(in_channels=1,num_classes=10)`，28→28→14→10→5，flatten=400，FC 120/84/10，隐藏层ReLU。共61,706个参数（含bias）；初始化后断言形状/参数数目。

种子20260909；在CPU创建模型，no\_grad零输入forward一次完成lazy初始化，按 `_flatten_params` 的完整名字排序，将全部权重显式float32保存为init\_weights.npz。每个后端在独立子进程载入同一文件；不按set迭代顺序zip参数、不只依靠相同seed重建、不共享Adam状态。

实验配置：train前1024、test前256；batch64、shuffle=False、drop\_last=False、epochs=2；Adam(lr=0.001,beta1=0.9,beta2=0.999,eps=1e-10,l1/l2=0)，每batch zero\_grad→forward→CrossEntropyLoss→backward→step。各模式在各自完整进程中执行。测试用Config.test\_mode+no\_grad，关闭图优化/AMP/可视化/早停。

独立实验循环输出真实train/test指标（总loss按样本数加权）；另加Trainer短数据smoke确认框架入口兼容。不得仅输出脚本生成的“估计训练指标”。

>   【Claude 补充 C1 · 2026-09-09 · 非原文】同进程     `Optimizer._state`     污染防护（强制细化 8.2/8.3）  
> `Optimizer._state` 为类属性（optim.py:32），且 Adam 的 S/V/T 子状态仅在缺失时创建（optim.py:201-207）：同一进程内实例化第二个 Adam 会直接复用第一个实例的动量/二阶矩 dict，并从继续累加的 T\_KEY 起步。因此仅“每个后端一个进程”不足以隔离 8.2 的独立实验循环与 8.3 第3层的 Trainer smoke——二者若同进程先后运行，smoke 的 Adam 起步即被上一实验污染（t 继续累加、v/s 复用）。规定：同一 Python 进程内只允许实例化一个 Optimizer；Trainer smoke 必须独立子进程执行（复用 bench 脚本的子进程机制），与三个后端进程同级隔离。

### 8.3 对比层次

1. 算子固定输入与同一任意上游梯度：CuPy为主要oracle，CPU float64小规模朴素公式/有限差分为独立oracle。
2. 同一初始化、同一batch：比较conv1/relu1/pool1/conv2/relu2/pool2/flatten/fc1/relu3/fc2/relu4/logits、loss、输入梯度、全部参数梯度及Adam更新后参数。仅调试对比时保留中间梯度。
3. 独立训练：记录每一步loss/accuracy和每epoch测试指标。开始与结束对训练子集做同模式评估；各后端最终训练loss应下降，测试accuracy与CuPy差值<=0.02。两epoch只是训练闭环演示，不宣称达到经典LeNet最终准确率。

## 9. 测试、性能与完成定义

### 9.1 测试矩阵

pytest文件内部从文件位置把code加入sys.path，所有执行入口有main保护。注册cuda标记；CPU可用性测试不得模块级importorskip CuPy。无CuPy/GPU时只跳过GPU测试，报告not\_run；不能把跳过算通过。

固定测试：

- 元素shape：()、(1,)、(17,)、(257,)、(2,3,5)、空数组；同形、左/右host标量、device标量、非连续切片和transpose。广播(2,3)+(3,)在auto回退；Mul/Div广播反向用有限差分验正确。pow指数0/1/2/0.5，非整数幂只用正底数。
- 卷积具体案例(x/W,s,p,d)：(1,1,5,7)/(2,1,3,2),(1,1),(0,0),(1,1)；(2,3,7,5)/(4,3,3,3),(2,1),(1,2),(1,1)；(1,2,9,8)/(3,2,3,2),(2,2),(2,1),(2,1)；(1,1,1,1)/(1,1,1,1)；LeNet两层N=1/64。均测试bias有/无。
- im2col/col2im：matrix与6D，对照旧函数；测试 `<im2col(x),v>=<x,col2im(v)>`，不能要求col2im(im2col(x))等于x（有重叠计数）。
- pool：(2,2,s2)、(3,2,s1,p1)、非方形窗口、负值与0 padding、平局、全padding窗口、重叠梯度；argmax逐项相同，反向有相加。
- 非正kernel/stride/dilation、负pad、channels不匹配、跨设备输入报错。float64/一般broadcast/group是auto回退测试。测试缺DLL/编译失败的预启动回退与strict错误，不故意破坏GPU上下文制造非法访存。
- 分叉图累计梯度、重复两次forward后分别backward、不同权重同shape、两条非默认stream（显式事件依赖）、无bias、保存/加载参数、100次完整步骤释放图后live显存无持续增长。

>   【Claude 补充 C8 · 2026-09-09 · 非原文】im2col/col2im 的 dilation 对照基线说明  
> 现有 `Im2col` Function 包装不传 dilation（functions.py:615-616），而底层 `im2col_array` 直调支持 dilation 参数（functions.py:672）。故本节“对照旧函数”仅指直调 `im2col_array`/`col2im_array` 并显式传 dilation；不得以 Im2col/Col2im Function 包装作为 dilation≠1 的基线。dilation=1 时两者等价，任选。

### 9.2 数值门槛

记录 max\_abs 和 relative\_l2=norm(a-b)/max(norm(ref),1e-12)，另做 `abs(a-b)<=atol+rtol*abs(ref)` 全元素检查；不能只看均值。

| 场景                      | atol | rtol | relative\_l2上限（ref norm>=1e-8时） |
| ----------------------- | ---: | ---: | ------------------------------: |
| 元素/搬运/pool              | 2e-5 | 2e-5 |                            2e-5 |
| Conv/Linear前后向、小批逐层     | 5e-5 | 2e-4 |                            2e-4 |
| LeNet单步logits/全部梯度与更新参数 | 1e-3 | 1e-3 |                            1e-3 |

标量loss绝对误差<=1e-4；ref近零时以绝对容差为准。float64有限差分步长1e-5，小形状至少抽查每种输入20坐标，绝对/相对阈值1e-5/1e-3；避开ReLU零点/MaxPool平局，平局用显式oracle测试。超标定位首个不一致张量，不随意放宽阈值。

### 9.3 性能

固定N=1/16/64，主要验收N=64；GPU常驻输入，当前stream预热20次，计时100次，报告Event中位数/P95与同步墙钟step时间，两者分别命名，编译/首次上下文时间单列。算子计时包含必要复制/分配，GPU输入驻留完整训练step包含forward+loss+backward+Adam，数据读取和日志不计入；另报告真实epoch墙钟时间。

各后端独立进程、相同初始化。性能测试也核对实际后端，不得使用auto悄悄选择CuPy。目标明确定义：`median_step_ms_custom/median_step_ms_cupy <= 1.10`（允许最多10%变慢，不表示1.1倍加速）；同时报告speedup=CuPy/custom。未达到须写performance\_not\_met，不称为全面通过；可以继续DLL搬迁验证，但不得掩盖性能缺口。

显存报告CuPy pool used/total的阶段采样高水位及明确列出的col/gcol工作区字节。称为采样值，不称GPU精确峰值。真实kernel启动次数仅在Nsight等profile可用时报告；否则null/not\_measured，dispatch计数另列。

### 9.4 顺序与命令

从仓库根目录执行，后端对比入口 `code/bench_cuda_lenet.py` 同时支持profile=smoke/train/benchmark/all；程序以子进程运行指定后端并汇总。

```powershell
# S0：记录环境、原始CPU/CuPy基线，建立init_weights；无GPU不得声称GPU通过。
# S1：基础算子实现与接入（含必要的广播梯度修复）。
python -m pytest code/tests/test_cuda_backend_cpu.py code/tests/test_cuda_kernels.py -q -k 'elementwise or backend'
# S2：im2col/col2im/pool、卷积、Linear/loss接入。
python -m pytest code/tests/test_cuda_kernels.py -q
# S3：RawModule训练和报告；通过正确性后才开始S4源码迁移/DLL编译。
python -m pytest code/tests/test_cuda_lenet.py -q
python code/bench_cuda_lenet.py --backends cupy rawmodule --profile all --out artifacts/cuda_stage1
# S4：nvcc DLL编译（Arch由已探测GPU生成，不写死）。
powershell -NoProfile -ExecutionPolicy Bypass -File code/eneuro/base/cuda/build_cuda.ps1
# S5：三方复验，DLL不是仅有构建脚本。
python code/bench_cuda_lenet.py --backends cupy rawmodule extension --profile all --out artifacts/cuda_stage1_dll
```

>   【Claude 补充 C6 · 2026-09-09 · 非原文】S0 增加编译工具链预探测  
> 原 S0 只记录环境与基线。为避免 S3 全部通过后才发现无法进入 S4，S0 同时探测并记录：nvcc 可用性与版本（含按 AGENTS/CLAUDE 切到 CUDA 12.6 后复测）、MSVC x64 工具链（cl.exe 经 vswhere/VsDevCmd 定位结果）、Windows SDK 版本，写入 cuda\_lenet\_report.json 的 environment 字段。任一缺失仅标记 dll\_build=not\_ready，不阻塞 S1–S3。

CPU回归使用新增backend\_cpu、针对所修改算子的NumPy梯度测试；另运行 `python code/tests/test_special_convolutions.py`、`python code/tests/test_advanced_convolutions.py` 并保存前后结果。发现原有失败明确区分，不要求扩展修复外部框架、Web、数据依赖测试。标注缺数据/GPU/编译器时未完成的门槛。

## 10. 报告与交付给用户

输出目录由--out指定；默认artifacts/cuda\_stage1。除日志/环境/初始化npz外，必须输出：

- cuda\_operator\_compare.csv：backend,actual\_backend,op,phase,shape,dtype,status,max\_abs,relative\_l2,atol,rtol,median\_ms,p95\_ms,pool\_used\_sample\_high\_water\_bytes,fallback\_reason。
- cuda\_lenet\_compare.csv：backend,epoch,step,split,num\_samples,loss,accuracy,step\_gpu\_ms,step\_wall\_ms,status；accuracy为0..1。
- cuda\_lenet\_report.json：schema\_version=1、UTC时间、git commit及dirty、执行命令/解释器/依赖版本/GPU/驱动/Toolkit/NVRTC/架构、数据路径hash/归一化/划分、seed/超参、各阶段pass/fail/skip/not\_run、跳过原因、actual\_backend计数/回退、自研与shared\_cupy清单、误差/性能门槛及结论、产物路径。无测量用null，不能填0。
- 构建DLL、manifest、复现命令、依赖说明；二进制按产物交付，不强制提交git。

S0-S5只在实际验收通过后标记complete；缺GPU/数据/编译器时允许完成不依赖它的准备，必须准确列出尚未验证内容。最终答复说明改了什么、数值结果、性能是否达标、如何复现和剩余问题。禁止只写kernel骨架、只做前向或只跑CuPy fallback便声称完整交付。

## 11. 可直接交给 GPT-5.6 Sol 的任务说明

> 用户明确授权编码后：请先完整阅读本计划与技术原理文档，核对AGENTS和工作区状态。按S0→S5实现，先Python内嵌CUDA C+RawModule验证，再迁移.cu并通过nvcc DLL/ctypes正式交付。保留CuPy/NumPy兼容接口和已有用户改动。按算子清单完成一阶反向、LeNet训练/测试、三方数值和性能报告；严格模式不得隐藏回退。不能将未运行或跳过标为通过。若有阻碍，保存诊断和已完成阶段；不得自行把范围缩成仅文档、仅骨架或仅前向。

### 10.1 报告字段补充

`data_path_hash`、`init_weights_hash` 和源码 hash 统一使用 SHA-256 十六进制小写；路径不存在时填 `unknown`，不填空字符串。所有时间字段使用 UTC ISO-8601，所有耗时字段使用毫秒，显存字段使用字节。

>   【Claude 补充 C10 · 2026-09-09 · 非原文】勘误记录（仅记录，不改动原文）  
>
> - §1 表中“优化器用 CuPy 表达式”：实为 `xp`（np 或 cp）表达式（各 step 经 `get_array_module` 分派），CPU 基线与 GPU 基线共用同一代码，无实际影响。
> - §11 之后的“### 10.1 报告字段补充”编号应为 11.1；内容本身有效，按 11.1 理解即可。

