# EnNeuro 框架性能 Bug 修复报告

------

## 一、问题发现

**诱因**：对比 `train.py`（使用 `Trainer` 封装）与 `test_resnet18.py`（裸循环）训练同一 ResNet18 模型时，观测到显著的速度差异：

| 脚本               | batch_size | 耗时           |
| ------------------ | ---------- | -------------- |
| `test_resnet18.py` | 32         | ~340s          |
| `train.py`         | 16         | ~498s          |
| `train.py`         | 32         | ~1130s（推算） |

特别异常的是：`train.py` 使用 batch=32 比 batch=16 **慢 2.25 倍**（每 batch 慢 4.6 倍），与通常"大 batch 更快"的预期相反。

------

## 二、根因排查过程

### 第一层：表面差异分析

初步对比两脚本，发现几处不同：验证集额外推理、batch_size 差异、Adam vs SGD、图像加载方式。其中 **Adam 额外占用约 88MB 显存**（一阶矩 + 二阶矩），加上验证推理，推测是显存超限触发 GPU 换页。

**结论**：部分正确，但无法解释 batch=32 比 batch=16 慢 4.6 倍/batch。

------

### 第二层：计算图生命周期分析

深入分析 CPython 引用计数机制与 `Function.__call__` 的设计：

```python
# core.py Function.__call__
self.inputs = inputs        # 对所有输入 Tensor 的强引用
self.outputs = [weakref.ref(output) for output in outputs]  # 弱引用
```

计算图由输出 Tensor 向输入方向通过强引用链接，图的释放依赖 `loss` 变量被覆盖。在 `Trainer._one_step` 的 batch 循环中：

```
iteration N:   forward → 构建图N → backward → step
                                                 ↓
iteration N+1: y_hat = model(Xb)  ← 此时图N还活着（loss N 未被覆盖）
               loss = loss_fn(...) ← 图N在此刻才开始释放
```

**新旧两张图同时驻留显存**，峰值内存：

```
batch=16: peak = 2 × graph₁₆
batch=32: peak = 2 × graph₃₂ = 4 × graph₁₆
```

batch=32 的瞬时峰值是 batch=16 的 **4 倍**，触发 GPU 换页，解释了 4.6x 的每 batch 耗时差异。

------

### 第三层：`enable_backprop` 实际无效

为验证修复效果，检查 `enable_backprop` 的实际作用范围。发现 `Function.__call__` **完全不检查该 flag**：

```python
# core.py（修复前）
def __call__(self, *inputs):
    ...
    # ← 无任何 enable_backprop 检查
    self.generation = max([x.generation for x in inputs])
    for output in outputs:
        output.set_creator(self)   # 无论如何都建图
    self.inputs = inputs           # 无论如何都保留强引用
```

`enable_backprop=False` 在此实现中仅控制 `backward()` 内部是否构建**二阶梯度图**（用于高阶求导），对**前向计算图的构建完全无效**。这意味着：

- 验证阶段 `with Config.using_config('enable_backprop', False)` 的"保护"是假象
- 验证的完整计算图始终被构建，占用等量显存

------

### 第四层：自引入 Bug

为解决循环后 `_is_regression(y_hat, yb)` 需要访问 Tensor 的问题，引入了 `last_y_hat = y_hat`，导致修复失效：

```python
# 错误的修复
last_y_hat, last_yb = y_hat, yb   # ← Tensor 对象引用，图未释放
del y_hat, loss, y_target, y_true_cls  # ← 只删了局部名，图依然活着
```

`last_y_hat` 持有与 `y_hat` 相同的 Tensor 对象，`del y_hat` 仅减少了一个引用计数，整条图通过 `last_y_hat → y_hat.creator → Function.inputs → ...` 链路仍然存活。

------

## 三、最终修复方案

### Fix 1：`Function.__call__` 加入 `enable_backprop` 检查（`core.py`）

```python
# 修复后
if Config.enable_backprop:
    self.generation = max([x.generation for x in inputs])
    for output in outputs:
        output.set_creator(self)
    self.inputs = inputs
    self.outputs = [weakref.ref(output) for output in outputs]
```

`enable_backprop=False` 时不建图、不保留 `inputs` 强引用，所有中间激活值在 forward 完成后立即可被回收。

### Fix 2：验证阶段扩大 `no_grad` 范围（`trainer.py`）

```python
# 修复后：forward 和 loss 计算全部在无梯度上下文中
with Config.using_config('enable_backprop', False), Config.using_config('train', False):
    y_hat = self.model(Xb)
    ...
    loss = self.loss_fn(y_hat, y_target)
```

Fix 1 生效后，此处的 `enable_backprop=False` 真正使验证阶段不产生任何计算图，显存立即降回参数占用水平。

### Fix 3：训练循环末尾显式释放计算图根节点（`trainer.py`）

```python
# 保存形状信息而非 Tensor 对象，避免图被意外持有
last_y_hat_shape, last_yb_ndim = y_hat.shape, yb.ndim
del y_hat, loss, y_target, y_true_cls   # 真正释放所有图根节点
```

循环末尾所有持有计算图根节点的局部变量被显式删除，确保旧图在下一次 forward **开始之前**完全释放。

------

## 四、Bug 原理总结

本质上是**三个独立缺陷叠加**，在 batch=32 时共同触发 GPU OOM 换页：

```
缺陷①：Function.__call__ 从不检查 enable_backprop
         → 验证阶段无法关闭计算图，始终占用 ~350MB 显存

缺陷②：训练循环中计算图生命周期管理缺失
         → 每个 batch 开始时，新旧两张图同时驻留
         → 峰值 = 2 × graph，batch=32 时为 4 × graph₁₆

缺陷③：enable_backprop 语义与实现脱节
         → 框架提供了 no_grad() API，但 Function 层面从未实现
         → 开发者误以为可以用它节省显存，实际完全无效
```

三个缺陷叠加后，batch=32 时 GPU 峰值显存约为 batch=16 的 **4～6 倍**，超出 8GB VRAM 上限，触发 CuPy 内存池换页，导致每 batch 耗时从 1.6s 飙升至 7.3s。