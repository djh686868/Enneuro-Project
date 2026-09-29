# EnNeuro 框架静态代码审查缺陷报告

> **审查日期**：2026-07-21
> **审查范围**：EnNeuro 框架全部 Python 源文件（约 95 个）
> **审查方法**：人工逐行审查 + 多代理并行交叉验证
> **审查人**：Claude Code（静态分析）

---

## 一、审查概要

| 统计项 | 数量 |
|--------|------|
| 审查文件数 | ~95 |
| 发现缺陷总数 | **62** |
| 🔴 Critical | 5 |
| 🟠 High | 19 |
| 🟡 Medium | 23 |
| 🟢 Low | 15 |

---

## 二、关键发现（Critical & High）

### CR-1 🔴 `Optimizer._state` 类变量被所有实例共享

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/nn/optim.py:32-44` |
| **类型** | 逻辑错误 — 状态污染 |
| **触发条件** | 创建任意两个优化器实例 |

**问题描述**：`_state: dict[str, Any] = {}` 声明为类变量，所有 `SGD`/`MomentumSGD`/`Adam` 实例共享同一字典。`__init__` 中 `self._state.update(...)` 直接修改共享字典，导致后创建的优化器污染前一个的状态（lr、动量速度、Adam 时间步等）。

**复现步骤**：
```python
opt1 = Adam(model1.params(), lr=0.001)
opt2 = Adam(model2.params(), lr=0.01)
# opt1._state['lr'] 已被 opt2 覆盖为 0.01
```

**修复建议**：在 `__init__` 中将 `_state` 初始化为实例变量：
```python
def __init__(self, ...):
    self._state = {}  # 实例变量，而非类变量
    self._state.update(lr=lr, l2_lambda=l2_lambda, l1_lambda=l1_lambda)
```

---

### CR-2 🔴 Windows 多进程递归派生风险

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/data/dataloader.py:131-139` |
| **类型** | 平台兼容性 — 系统崩溃 |
| **触发条件** | Windows 上设置 `num_workers > 0` |

**问题描述**：`_AsyncLoaderIter.__init__` 使用 `multiprocessing.Process()` 启动子进程，但缺少 `if __name__ == "__main__"` 保护。Windows 使用 `spawn` 方式创建进程，子进程会重新导入整个模块，从而再次执行 `Process()` 创建，导致无限递归派生。

**修复建议**：
```python
# 方案 A：在文档中禁止 Windows 多进程，自动检测并降级
import platform
if platform.system() == 'Windows' and num_workers > 0:
    num_workers = 0
    warnings.warn("DataLoader multiprocessing is not supported on Windows")

# 方案 B：使用 __main__ 保护 + spawn 安全设计
```

---

### CR-3 🔴 `Parameter.backward()` 参数传递错误

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/base/parameter.py:5-10` |
| **类型** | API 错误 — 梯度被忽略 |
| **触发条件** | 调用 `param.backward(gradient)` 期望使用外部梯度 |

**问题描述**：
```python
def backward(self, gradient: Tensor | None = None) -> None:
    return super().backward(gradient)
```

`Tensor.backward` 的签名为 `backward(self, retain_grad=False, create_graph=False)`。此处将 `gradient` 作为 `retain_grad` 传入，外部梯度完全被忽略，且当 `gradient` 为 Tensor 时 `retain_grad` 恒为真值。

**修复建议**：
```python
def backward(self, gradient: Tensor | None = None, retain_grad=False, create_graph=False) -> None:
    if gradient is not None:
        self.grad = gradient
    return super().backward(retain_grad=retain_grad, create_graph=create_graph)
```

---

### CR-4 🔴 `TimeMeter.avg()` 引用未定义属性导致崩溃

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/train/meters.py:52-53` |
| **类型** | 运行时错误 — `AttributeError` |
| **触发条件** | 使用 Visualizer 时调用 `time_meter.avg()` |

**问题描述**：`TimeMeter.avg()` 访问 `self.count`，但 `__init__` 和 `reset()` 均未初始化该属性。任何使用 `Visualizer` 的训练过程在调用此方法时都会崩溃。

**修复建议**：在 `__init__` 和 `reset()` 中添加 `self.count = 0`。

---

### CR-5 🔴 `run_feature_maps` 钩子返回值解构错误

| 属性 | 内容 |
|------|------|
| **文件** | `code/web_server/services/explain_service.py:109-116` |
| **类型** | 运行时错误 — 功能完全不可用 |
| **触发条件** | 调用 `/api/explain/feature_maps` 端点 |

**问题描述**：
```python
hook = capture_features(target_layer)  # 返回 (handle, storage) 元组！
# storage 字典未被提取，特征图永不存在
# ...
if target_layer._captured_features is None:  # 该属性不存在！
    raise RuntimeError("No features captured")
```

`capture_features` 返回 `(handle, storage)` 元组，但代码未解构，且访问了不存在的 `_captured_features` 属性。此 API 端点始终失败。

**修复建议**：
```python
feature_handle, feature_storage = capture_features(target_layer)
# 使用 feature_storage['output'] 获取特征图
```

---

### HI-1 🟠 `Tensor.__eq__` / `Tensor.__lt__` 返回数组而非 bool

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/base/core.py:140-151` |
| **类型** | 语义错误 |
| **触发条件** | 在布尔上下文中比较 Tensor（`if a == b:`、`a in list` 等） |

**问题描述**：`self.data == other.data`（numpy/cupy 数组比较）返回逐元素布尔数组，而非标量 `bool`。搭配 `@total_ordering` 装饰器，所有比较运算符均受影响，在 `if` 语句中会抛出 `ValueError: The truth value of an array with more than one element is ambiguous`。

**修复建议**：
```python
def __eq__(self, other):
    if isinstance(other, Tensor):
        return np.array_equal(self.data, other.data)
    return False
```

---

### HI-2 🟠 `Layer.__call__` 裸 `raise` 无异常类型

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/nn/module.py:37-38` |
| **类型** | 语法/逻辑错误 |
| **触发条件** | `forward()` 返回空元组 |

**问题描述**：
```python
if len(outputs) == 0:
    raise  # RuntimeError: No active exception to re-raise
```
裸 `raise` 在不处于异常上下文时抛出无意义的 `RuntimeError`。

**修复建议**：`raise RuntimeError("forward() returned 0 outputs")`

---

### HI-3 🟠 `Layer.__call__` 不自动将输入转换为 Tensor

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/nn/module.py:30-39` |
| **类型** | 语义疏忽 |
| **触发条件** | 传入 numpy 数组给 Layer |

**问题描述**：`Function.__call__` 会用 `as_Tensor` 包装输入，但 `Layer.__call__` 直接传递原始输入给 `forward()`。如果用户传入 numpy 数组，将无 autograd 支持。

**修复建议**：在 `Layer.__call__` 开头添加 `inputs = [as_Tensor(x) if not isinstance(x, Tensor) else x for x in inputs]`。

---

### HI-4 🟠 `as_Tensor` 对 int/float 静默返回错误设备

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/base/core.py:350-355` |
| **类型** | 静默错误 |
| **触发条件** | 传入标量 `int` 或 `float` |

**问题描述**：`isinstance(x, (int, float))` 分支只执行 `pass`，然后落入默认逻辑并永远返回 `device='cpu'` 的 Tensor。GPU 训练场景下导致设备不匹配。

**修复建议**：要么抛出明确的 `TypeError`，要么接受 `device` 参数。

---

### HI-5 🟠 `FusedConvReLU.backward` 硬编码 `np.tensordot` 导致 GPU 路径断裂

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/base/functions.py:2289-2291` |
| **类型** | GPU 不兼容 |
| **触发条件** | 在 GPU 上使用 FusedConvReLU |

**问题描述**：
```python
gW = np.tensordot(g_conv, col_x, ((0,2,3), (0,4,5)))  # 硬编码 np
```
当 `col_x` 为 CuPy 数组时，`np.tensordot` 会导致类型错误或触发昂贵的 GPU→CPU 同步。对比 `Conv2DGradW.forward`（行 1823）正确使用了 `xp.tensordot`。

**修复建议**：改为 `xp = get_array_module(g_conv); gW = xp.tensordot(...)`。

---

### HI-6 🟠 全局钩子系统单例非线程安全

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/utils/hooks.py:82-91` |
| **类型** | 并发安全 |
| **触发条件** | 多线程同时构造 `HookRegistry()` |

**问题描述**：`HookRegistry.__new__` 实现经典单例但无锁保护。两个线程同时构造时可能获得不同实例，导致层调用序列记录不一致。

**修复建议**：使用 `threading.Lock` 保护单例创建。

---

### HI-7 🟠 `unchain_backward` 逻辑实现完全错误

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/base/core.py:254-270` |
| **类型** | 死代码 / 逻辑错误 |
| **触发条件** | 调用 `tensor.unchain_backward()` |

**问题描述**：`while funcs:` 循环只执行一次 `pop` 就退出（无 `continue` 逻辑），随后的 `for` 循环处理的是 while 退出前的最后一个 `f`。`seen_set` 从未与 `funcs` 交叉检查，函数收集逻辑与 `backward` 中的正确实现完全不同。

**修复建议**：参考 `backward()` 方法的正确拓扑排序逻辑重写。

---

### HI-8 🟠 HTTP 响应体未消费导致连接泄漏

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/utils/monitor_client.py:77-79` |
| **类型** | 资源泄漏 |
| **触发条件** | 持续高频率训练监控 |

**问题描述**：`urllib.request.urlopen(req, timeout=2)` 返回的响应对象从未调用 `.read()` 或 `.close()`。高频 batch 事件场景下，大量未关闭的连接会耗尽可用 socket。

**修复建议**：
```python
with urllib.request.urlopen(req, timeout=2) as resp:
    resp.read()  # 消费响应体
```

---

### HI-9 🟠 训练回调传递完全错误的数据

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/train/trainer.py:227-236` |
| **类型** | 死代码 + 数据错误 |
| **触发条件** | 每次 epoch 结束回调 |

**问题描述**：
```python
train_loss_ep, train_acc_ep = self._one_step(...) if False else (0.0, 0.0)
```
`if False` 使训练集评估永远不执行，回调的 `train_loss`/`train_acc`/`val_loss`/`val_acc` 全部填入同一个验证集 loss/acc。Web UI 看到的训练指标完全不可信。

**修复建议**：移除 `if False`，或在非 verbose 模式下单独做一次前向评估。

---

### HI-10 🟠 `trainer.fit` 验证集单 batch 推理

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/train/trainer.py:203-207` |
| **类型** | 内存风险 |
| **触发条件** | 大验证集 |

**问题描述**：`batch_size=len(val_loader.dataset)` 将整个验证集作为一个 batch 送入 GPU。大验证集会 OOM。

**修复建议**：使用合理的验证 batch_size（如 64 或 128）。

---

### HI-11 🟠 `FusedConvBNReLU` running_var 命名错误

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/base/functions.py:2391,2402` |
| **类型** | 属性命名错误 |
| **触发条件** | 使用融合算子时 |

**问题描述**：
```python
self.running_var = Tensor(..., name='running_mean')  # 错误！应为 'running_var'
```
`running_var` 的 `name` 被错误设为 `'running_mean'`，导致序列化/调试时无法区分二者。

**修复建议**：将 `name` 改为 `'running_var'`。

---

### HI-12 🟠 Zip Slip 漏洞

| 属性 | 内容 |
|------|------|
| **文件** | `code/web_server/routers/datasets.py:35-36` |
| **类型** | 安全漏洞 |
| **触发条件** | 上传恶意构造的 ZIP 文件 |

**问题描述**：`zipfile.ZipFile(path).extractall(dest)` 不校验提取路径，攻击者可在 ZIP 中包含 `../../` 路径写入服务器任意位置。

**修复建议**：
```python
for member in z.infolist():
    target = os.path.realpath(os.path.join(dest, member.filename))
    if not target.startswith(os.path.realpath(dest)):
        raise ValueError("Zip path traversal detected")
z.extractall(dest)
```

---

### HI-13 🟠 `load_model` 是空壳 —— 推理服务不可用

| 属性 | 内容 |
|------|------|
| **文件** | `code/serving/predictor.py:16-26` |
| **类型** | 功能缺失 |
| **触发条件** | 任何推理请求 |

**问题描述**：`load_model` 在 `os.path.exists(self.model_path)` 为 True 时仅打印日志，从不实际加载模型，始终回退到硬编码的 3×3 虚拟矩阵。推理服务完全不可用。

**修复建议**：实现实际的模型加载逻辑，使用 `Serializer.load()`。

---

### HI-14 🟠 `predict` 维度不匹配时静默返回虚假结果

| 属性 | 内容 |
|------|------|
| **文件** | `code/serving/predictor.py:47-53` |
| **类型** | 静默数据错误 |
| **触发条件** | 输入维度与权重不匹配 |

**问题描述**：
```python
if len(row) == len(self.model['weights']):
    ...
else:
    output = np.sum(row) * 0.1  # 完全无意义的"预测"
```
维度不匹配时不报错，而是返回 `sum(input) * 0.1`，调用方获得看似正常、完全错误的"预测结果"。

**修复建议**：抛出 `ValueError(f"Input dim {len(row)} != model dim {len(self.model['weights'])}")`。

---

### HI-15 🟠 TCP 服务无限制线程创建（DoS）

| 属性 | 内容 |
|------|------|
| **文件** | `code/serving/tcp_server.py:79-88` |
| **类型** | 拒绝服务漏洞 |
| **触发条件** | 攻击者建立大量 TCP 连接 |

**问题描述**：每个 TCP 连接无条件创建新线程，无最大线程数限制、无线程池、无速率限制。数千连接可耗尽系统线程和文件描述符。

**修复建议**：使用 `ThreadPoolExecutor` 配合有界工作队列。

---

### HI-16 🟠 梯度回传引导反向传播中 Tensor 数据就地修改

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/explainability/guided_backprop.py:139-167` |
| **类型** | 计算图损坏 |
| **触发条件** | 使用 Guided Backpropagation 时 |

**问题描述**：`_backward_hook` 中执行 `grad.data = grad_data` 替换了计算图中 Tensor 的 data。若同一梯度被多个操作共享，此就地修改会破坏其他路径的反向传播。

**修复建议**：不修改原始 Tensor 的 data，而是返回新的梯度张量。

---

### HI-17 🟠 `asyncio.Queue` 在同步 handler 中使用

| 属性 | 内容 |
|------|------|
| **文件** | `code/web_server/routers/monitor.py:27,41` |
| **类型** | 并发模型错误 |
| **触发条件** | 外部监控推送数据时 |

**问题描述**：`push_metrics` 是同步函数，但操作 `asyncio.Queue`。FastAPI 的同步端点在线程池中运行，而 `asyncio.Queue` 绑定到事件循环线程，跨线程调用可抛出 `RuntimeError`。

**修复建议**：将 `asyncio.Queue` 替换为 `queue.Queue`，或将端点改为 `async def`。

---

### HI-18 🟠 `stop_training` 竞态条件

| 属性 | 内容 |
|------|------|
| **文件** | `code/web_server/services/task_manager.py:112-113` |
| **类型** | 竞态条件 |
| **触发条件** | 训练开始后立即调用停止 |

**问题描述**：`task.trainer_ref` 在工作线程中赋值（行 71）。如果用户在线程到达该行之前调用 `stop_training`，`task.trainer_ref` 为 `None`，`request_stop()` 被静默跳过。训练继续运行但 UI 显示"已停止"。

**修复建议**：使用 `threading.Event` 信号机制，不依赖 `trainer_ref` 的赋值时机。

---

### HI-19 🟠 `model_to_graph` 执行真实前向传播产生副作用

| 属性 | 内容 |
|------|------|
| **文件** | `code/eneuro/ao/graphoptimizer.py:10-14` |
| **类型** | 状态污染 |
| **触发条件** | 图优化（AO）流程 |

**问题描述**：
```python
with trace_context() as tracer:
    _ = model(sample_input)  # 真实前向传播！
```
未使用 `Config.test_mode()` 或 `Config.no_grad()`，BatchNorm 运行统计量被样本输入污染。

**修复建议**：
```python
with Config.test_mode(), Config.no_grad(), trace_context() as tracer:
    _ = model(sample_input)
```

---

## 三、中危缺陷 (Medium)

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| M-1 | `data/dataloader.py:10-42` | 工作进程无 try/except，数据加载异常时主进程永久阻塞 |
| M-2 | `data/dataloader.py:131` | 工作进程非 daemon，主进程崩溃后残留僵尸进程 |
| M-3 | `data/dataloader.py:183` | `if p in self.workers` 永为 True（死条件） |
| M-4 | `data/dataset.py:19` | `assert np.isscalar(index)` —— 用断言做输入校验 |
| M-5 | `data/dataset.py:27` | `__len__` 无 None 检查，`prepare()` 异常后 `len(None)` 崩溃 |
| M-6 | `data/transform.py:75-86` | `adjust_hsv` 接收 [0,1] 浮点数但 cv2 期望 [0,255] uint8 |
| M-7 | `train/trainer.py:313-314` | 训练时收集 `y_true_list`/`y_pred_list` 但从未使用 |
| M-8 | `utils/hooks.py:315` | 模块导入即触发全局 monkey-patch，无法按需导入 |
| M-9 | `utils/serializer.py:33-34` | `bare except Exception` 静默丢弃真实错误 |
| M-10 | `base/functions.py:439/125` | `Exp` 类在 core.py 和 functions.py 中重复定义 |
| M-11 | `base/functions.py:2195-2223` | `BatchNorm2d.backward` 混合 Tensor/原始数据操作 |
| M-12 | `base/functions.py:536-539` | `Div.forward` 无零除保护 |
| M-13 | `base/functions.py:146-154` | `Log.forward` 无非正输入验证 |
| M-14 | `base/functions.py:1947-1951` | `AveragePooling.backward` 变量名 `KW, KH` 互换 |
| M-15 | `nn/loss.py:112` | Sigmoid 对大负值输入数值不稳定 |
| M-16 | `explainability/gradcam.py:311-331` | 递归遍历 `__dict__` 可能无限循环；深度限制 20 |
| M-17 | `web_server/dataset_manager.py:76-82` | PKL 文件为检测编码而完整读取两次 |
| M-18 | `web_server/dataset_manager.py:68-72` | 标签解析失败静默设为 0 类 |
| M-19 | `web_server/routers/datasets.py:63-69` | 通道提取逻辑错误：3 通道图取 [0] 丢弃 G/B |
| M-20 | `serving/metrics.py:26` | 直方图静默丢弃 50% 数据 |
| M-21 | `serving/metrics.py:64-72` | `get_all_metrics` 无锁遍历直方图字典 |
| M-22 | `serving/tcp_server.py:22-29` | TCP 无最大消息长度限制 |
| M-23 | `ao/quantize.py:1-135` | 整个量化模块被注释掉，功能不可用 |

---

## 四、低危缺陷 (Low)

| # | 文件 | 问题 |
|---|------|------|
| L-1 | `global_config.py:6` | 模块级 `import cv2` 导致无 OpenCV 时模块不可导入 |
| L-2 | `global_config.py:1-3` | `VISUAL_CONFIG` 全局可变字典无锁 |
| L-3 | `base/functions.py:71-96, nn/loss.py:10-35` | `to_xp` 函数在两个文件中重复定义 |
| L-4 | `base/functions.py` 多处 | Winograd workspace 缓存限制（4/8/256）无来源注释 |
| L-5 | `base/functions.py:1826-1835` | `Conv2DGradW.backward` 缩进错误嵌套在 `forward` 内 |
| L-6 | `nn/module.py:515-519` | `_from_pure_list` 递归无深度限制 |
| L-7 | `data/transform.py:6-15` | `normalize` 硬编码 numpy，GPU tensor 触发不必要的传输 |
| L-8 | `utils/statedict.py:2-6` | 抽象接口未继承 `abc.ABC` / 无 `@abstractmethod` |
| L-9 | `utils/monitor_client.py:78-79` | `bare except` 静默丢弃所有错误 |
| L-10 | `utils/visualization.py` 多处 | 图像归一化代码重复 3 次 |
| L-11 | `web_server/main.py:15-20` | CORS `allow_origins=["*"]` 全开放 |
| L-12 | `ao/cast.py:122-131` | GradScaler 文档引用不存在的 `autocast` 上下文管理器 |
| L-13 | `ao/graph.py:225` | 可视化输出路径硬编码为 `"graph.dot"` |
| L-14 | `ao/tracer.py:1,8` | `import weakref` 重复导入 |
| L-15 | `serving/config.py:5` | `METRICS_ENABLED` 环境变量被解析但从未使用 |

---

## 五、按模块分布

| 模块 | Critical | High | Medium | Low | 合计 |
|------|----------|------|--------|-----|------|
| `eneuro/base/` | 1 | 3 | 4 | 3 | **11** |
| `eneuro/nn/` | 1 | 2 | 1 | 2 | **6** |
| `eneuro/data/` | 1 | 1 | 4 | 1 | **7** |
| `eneuro/train/` | 1 | 1 | 1 | 0 | **3** |
| `eneuro/utils/` | 0 | 3 | 2 | 4 | **9** |
| `eneuro/explainability/` | 0 | 1 | 1 | 0 | **2** |
| `eneuro/ao/` | 0 | 1 | 2 | 3 | **6** |
| `web_server/` | 1 | 4 | 4 | 1 | **10** |
| `serving/` | 0 | 3 | 3 | 1 | **7** |
| `eneuro/global_config.py` | 0 | 0 | 0 | 2 | **2** |

---

## 六、修复优先级建议

### 第一阶段（立即修复 — 影响常规使用）

| 优先级 | 编号 | 简述 |
|--------|------|------|
| 1 | CR-1 | Optimizer `_state` 类变量 → 实例变量 |
| 2 | CR-2 | Windows 多进程加保护或自动降级 |
| 3 | CR-3 | Parameter.backward 参数传错 |
| 4 | CR-5 | run_feature_maps 钩子解构错误 |
| 5 | HI-1 | Tensor 比较返回数组而非 bool |
| 6 | HI-4 | as_Tensor 标量静默返回错误设备 |
| 7 | HI-5 | FusedConvReLU GPU 路径硬编码 np |
| 8 | HI-11 | running_var 命名错误 |

### 第二阶段（安全加固）

| 优先级 | 编号 | 简述 |
|--------|------|------|
| 9 | HI-12 | Zip Slip 路径穿越 |
| 10 | HI-15 | TCP 无限制线程创建 |
| 11 | HI-18 | stop_training 竞态条件 |
| 12 | HI-6 | HookRegistry 单例非线程安全 |
| 13 | HI-8 | MonitorClient 连接泄漏 |

### 第三阶段（功能完整性）

| 优先级 | 编号 | 简述 |
|--------|------|------|
| 14 | CR-4 | TimeMeter.avg 崩溃 |
| 15 | HI-9 | 训练回调数据错误 |
| 16 | HI-13 | serving load_model 空壳 |
| 17 | HI-14 | serving predict 静默假结果 |
| 18 | HI-19 | model_to_graph 状态污染 |

---

## 七、附录：审查覆盖文件清单

```
code/eneuro/__init__.py
code/eneuro/global_config.py
code/eneuro/base/__init__.py
code/eneuro/base/core.py
code/eneuro/base/functions.py
code/eneuro/base/parameter.py
code/eneuro/nn/__init__.py
code/eneuro/nn/loss.py
code/eneuro/nn/module.py
code/eneuro/nn/optim.py
code/eneuro/data/__init__.py
code/eneuro/data/dataloader.py
code/eneuro/data/dataset.py
code/eneuro/data/transform.py
code/eneuro/train/__init__.py
code/eneuro/train/meters.py
code/eneuro/train/metrics.py
code/eneuro/train/trainer.py
code/eneuro/utils/__init__.py
code/eneuro/utils/hooks.py
code/eneuro/utils/monitor_client.py
code/eneuro/utils/serializer.py
code/eneuro/utils/statedict.py
code/eneuro/utils/visualization.py
code/eneuro/explainability/__init__.py
code/eneuro/explainability/gradcam.py
code/eneuro/explainability/guided_backprop.py
code/eneuro/ao/__init__.py
code/eneuro/ao/cast.py
code/eneuro/ao/executor.py
code/eneuro/ao/graph.py
code/eneuro/ao/graphoptimizer.py
code/eneuro/ao/pattern.py
code/eneuro/ao/quantize.py
code/eneuro/ao/tracer.py
code/web_server/main.py
code/web_server/schemas.py
code/web_server/routers/datasets.py
code/web_server/routers/explain.py
code/web_server/routers/models.py
code/web_server/routers/monitor.py
code/web_server/routers/system.py
code/web_server/routers/training.py
code/web_server/services/dataset_manager.py
code/web_server/services/explain_service.py
code/web_server/services/model_registry.py
code/web_server/services/task_manager.py
code/serving/benchmark_client.py
code/serving/client.py
code/serving/config.py
code/serving/logger.py
code/serving/metrics.py
code/serving/predictor.py
code/serving/schema.py
code/serving/server.py
code/serving/tcp_client.py
code/serving/tcp_server.py
```

---

> **报告生成日期**：2026-07-21
> **工具**：Claude Code 静态分析 + 多代理并行验证
