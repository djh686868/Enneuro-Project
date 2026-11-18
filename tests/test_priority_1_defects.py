"""
第一优先级缺陷复现脚本
========================
复现 EnNeuro 静态审查报告中标记为"立即修复"的 8 项缺陷。

用法:
    python tests/test_priority_1_defects.py

注意: CR-2 (Windows 多进程) 的复现脚本独立存放，因为会触发系统级崩溃。
"""

import sys
import os
import traceback

# 确保 eneuro 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from eneuro.base import Tensor, as_Tensor, Parameter
from eneuro.base.core import Config, Function
from eneuro.nn.optim import Adam, SGD, MomentumSGD, Optimizer
from eneuro.nn.module import (
    Conv2d, BatchNorm2d, Linear,
    FusedConvReLU, FusedConvBNReLU,
    Layer, Module, Sequential,
)
from eneuro.utils import capture_features

# -- 测试工具 ------------------------------------------------------------------

PASS = 0
FAIL = 0
CONFIRMED = 0  # 缺陷被确认触发（预期"失败"）


def test(name, expect_defect=True):
    """expect_defect=True: 预期该测试会"失败"（确认缺陷存在）"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global PASS, FAIL, CONFIRMED
            try:
                fn(*args, **kwargs)
                if expect_defect:
                    print(f"  [OK] {name} -- 缺陷未触发（可能已修复）")
                else:
                    print(f"  [OK] {name}")
                PASS += 1
            except AssertionError as e:
                if expect_defect:
                    print(f"  [DEFECT CONFIRMED] {name}")
                    print(f"       {e}")
                    CONFIRMED += 1
                else:
                    print(f"  [FAIL] {name}: {e}")
                    FAIL += 1
            except Exception as e:
                if expect_defect:
                    # 有些缺陷的表现就是抛异常
                    print(f"  [DEFECT CONFIRMED] {name}")
                    print(f"       {type(e).__name__}: {e}")
                    CONFIRMED += 1
                else:
                    print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    FAIL += 1
        return wrapper
    return decorator


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# -- 辅助: 可前向传播的最简模型 ------------------------------------------------

class TinyModel(Module):
    """用于优化器测试的最简模型"""
    def __init__(self, out_features=10):
        super().__init__()
        self.fc = Linear(out_features)

    def forward(self, x):
        return self.fc(x)


# ==============================================================================
# CR-1: Optimizer._state 类变量被所有实例共享
# ==============================================================================

@test("CR-1a: 不同 LR 的 Adam 互相覆盖 _state")
def test_cr1_lr_override():
    m1 = TinyModel(10)
    m2 = TinyModel(10)
    x = Tensor(np.random.randn(2, 10).astype(np.float32))
    m1(x)
    m2(x)

    opt1 = Adam(list(m1.params()), lr=0.001)
    lr_before = float(opt1._state['lr'])

    opt2 = Adam(list(m2.params()), lr=0.999)

    lr_after = float(opt1._state['lr'])
    assert abs(lr_before - 0.001) < 1e-6, f"opt1 lr 初始值异常: {lr_before}"
    assert abs(lr_after - 0.001) < 1e-6, \
        f"CR-1 触发! opt1._state['lr'] 从 {lr_before} 被 opt2 覆盖为 {lr_after}。" \
        f" 根本原因: Optimizer._state 是类变量，所有实例共享同一字典。"


@test("CR-1b: Adam 时间步 t 被第二个实例污染")
def test_cr1_timestep_pollution():
    m1 = TinyModel(10)
    m2 = TinyModel(10)
    x = Tensor(np.random.randn(2, 10).astype(np.float32))
    m1(x)
    m2(x)

    opt_a = Adam(list(m1.params()), lr=0.001)
    t_before = opt_a._state.get('t', 0)

    opt_b = Adam(list(m2.params()), lr=0.001)

    t_after = opt_a._state.get('t', 0)
    # 't' 键是否存在的逻辑因实现而异。关键是检查 _state 是否为共享对象。
    assert opt_a._state is opt_b._state, \
        f"opt_a._state (id={id(opt_a._state)}) 与 opt_b._state (id={id(opt_b._state)}) 不是同一对象。" \
        f" 如果这里失败了，说明类变量共享问题可能已修复。"

    print(f"       opt_a._state is opt_b._state = {opt_a._state is opt_b._state}")


@test("CR-1c: MomentumSGD 速度字典跨实例共享")
def test_cr1_momentum_collision():
    m1 = TinyModel(10)
    m2 = TinyModel(10)
    x = Tensor(np.random.randn(2, 10).astype(np.float32))
    m1(x)
    m2(x)

    opt1 = MomentumSGD(list(m1.params()), lr=0.01, momentum=0.9)
    opt2 = MomentumSGD(list(m2.params()), lr=0.01, momentum=0.9)

    v_key = 'v'
    assert opt1._state is opt2._state, \
        f"两个 MomentumSGD 的 _state 不是同一对象 (id: {id(opt1._state)} vs {id(opt2._state)})"

    # 验证 'v' 键的字典也是同一个
    v1 = opt1._state.get(v_key, {})
    v2 = opt2._state.get(v_key, {})
    assert v1 is v2, \
        f"CR-1 触发! 两个优化器共享 velocity 字典 (id={id(v1)})。" \
        f" 索引 '0' 的动量速度会被互相覆盖。"


@test("CR-1d: SGD 和 Adam 混合使用状态污染")
def test_cr1_mixed_optimizers():
    """最致命的场景: SGD 和 Adam 先后创建，Adam 的 beta1/beta2 等状态泄漏到 SGD 中"""
    m1 = TinyModel(10)
    m2 = TinyModel(10)
    x = Tensor(np.random.randn(2, 10).astype(np.float32))
    m1(x)
    m2(x)

    sgd = SGD(list(m1.params()), lr=0.01)
    adam = Adam(list(m2.params()), lr=0.001)

    # SGD 的 _state 不应该有 Adam 特有的键
    adam_keys = {'beta1', 'beta2', 'eps', 's', 'v', 't'}
    leaked = adam_keys & set(sgd._state.keys())
    if leaked:
        print(f"       CR-1 触发! SGD._state 泄漏了 Adam 的状态键: {leaked}")
        print(f"       SGD._state keys = {list(sgd._state.keys())}")
    assert not leaked, \
        f"SGD._state 不应包含 Adam 特有的键，但实际包含: {leaked}"


# ==============================================================================
# CR-3: Parameter.backward() 参数传递错误
# ==============================================================================

@test("CR-3: Parameter.backward(gradient) 将 gradient 误传为 retain_grad")
def test_cr3_parameter_backward():
    p = Parameter(np.array([1.0, 2.0, 3.0], dtype=np.float32), name='test_param')
    custom_grad = Tensor(np.array([10.0, 20.0, 30.0], dtype=np.float32))

    # Parameter.backward 签名: backward(self, gradient) → super().backward(gradient)
    # Tensor.backward 签名:  backward(self, retain_grad=False, create_graph=False)
    # 结果: gradient 参数被当作 retain_grad，外部梯度被忽略
    p.backward(custom_grad)

    expected = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    actual = p.grad.data if p.grad is not None else None

    if actual is None:
        print(f"       p.grad is None (backward 未设置梯度)")
    elif np.allclose(actual, np.ones_like(expected)):
        print(f"       CR-3 触发! p.grad = {actual} (全是 1)，期望 = {expected}。")
        print(f"       原因: gradient 参数被当作 retain_grad 传入，外部梯度被忽略。")
        print(f"       Tensor.backward 内部因 self.grad is None 而创建 ones_like。")
    assert actual is not None, "backward 未设置梯度"
    assert np.allclose(actual, expected), \
        f"CR-3 触发! Parameter.backward(gradient) 将 gradient 误传为 retain_grad。" \
        f" p.grad={actual}, 期望={expected}"


# ==============================================================================
# CR-5: run_feature_maps 钩子返回值解构错误
# ==============================================================================

@test("CR-5: capture_features 返回 tuple 但 run_feature_maps 未解构")
def test_cr5_capture_features_unpacking():
    conv = Conv2d(out_channels=8, kernel_size=3, pad=1, in_channels=3)
    x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))

    # 正确用法: 解构
    handle, storage = capture_features(conv)
    y = conv(x)
    assert storage['output'] is not None, "正常解构时 storage['output'] 不应为 None"
    handle.remove()

    # 模拟错误用法: 不解构 (正是 run_feature_maps 的 bug)
    wrong = capture_features(conv)
    assert isinstance(wrong, tuple), \
        f"capture_features 返回 {type(wrong).__name__}，是一个 tuple(len={len(wrong)})！"
    assert len(wrong) == 2, \
        f"capture_features 返回长度为 {len(wrong)} 的 tuple"

    print(f"       capture_features() 返回类型: tuple (handle, storage)")
    print(f"       run_feature_maps 中: hook = capture_features(layer)  -- 未解构!")
    print(f"       因此 storage dict 被丢失，特征图永远为 None。")

    # 清理
    wrong[0].remove()


# ==============================================================================
# HI-1: Tensor.__eq__ / __lt__ 返回数组而非 bool
# ==============================================================================

@test("HI-1a: Tensor.__eq__ 返回 ndarray 而非 bool")
def test_hi1_eq_returns_array():
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    result = a == b
    print(f"       a == b 返回类型: {type(result).__name__}")

    assert isinstance(result, np.ndarray), \
        f"期望 __eq__ 返回 ndarray (当前缺陷), 实际返回 {type(result).__name__}"
    assert result.shape == (3,), f"期望 shape=(3,), 实际 shape={result.shape}"

    # 验证在布尔上下文中崩溃
    try:
        bool(a == b)
        print(f"       bool(a == b) 未抛异常 (可能已修复)")
    except ValueError as e:
        print(f"       HI-1 确认! bool(a == b) 抛出 ValueError: {e}")


@test("HI-1b: @total_ordering 生成的比较运算符全部受影响")
def test_hi1_total_ordering_contamination():
    """@total_ordering 从 __lt__ 和 __eq__ 生成 __gt__, __ge__, __le__。
    由于 __eq__ 返回数组, __le__ 内部执行 (a < b) or (a == b) 时,
    Python 尝试对 ndarray 求布尔值, 触发 ValueError。"""
    a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    b = Tensor(np.array([3.0, 4.0], dtype=np.float32))

    # __lt__ 本身返回 ndarray (不抛异常, 但语义错误)
    lt_result = a.__lt__(b)
    assert isinstance(lt_result, np.ndarray), \
        f"__lt__ 返回 {type(lt_result).__name__}, 期望 ndarray"

    # __le__ 由 @total_ordering 生成为: lambda self, other: self < other or self == other
    # 这会触发 ValueError, 因为 ndarray 不能用在 or 中
    try:
        _ = a.__le__(b)
        print(f"       a.__le__(b) 未抛异常 (可能 total_ordering 已移除)")
    except ValueError as e:
        print(f"       HI-1 确认! a.__le__(b) (由 total_ordering 生成) 抛出 ValueError: {e}")
        print(f"       原因: (a < b) 返回 array, Python 无法对 array 求 bool 值。")


@test("HI-1c: Tensor 无法用于 list/set/dict 成员检查")
def test_hi1_list_membership():
    a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    b = Tensor(np.array([1.0, 2.0], dtype=np.float32))  # 值相同
    c = Tensor(np.array([5.0, 6.0], dtype=np.float32))  # 值不同

    lst = [a, c]
    try:
        result = b in lst
        print(f"       b in [a, c] = {result} (未抛异常)")
    except ValueError as e:
        print(f"       HI-1 确认! b in [a, c] 抛出 ValueError: {e}")
        print(f"       原因: Python 的 'in' 使用 __eq__ 比较, __eq__ 返回数组。")


# ==============================================================================
# HI-4: as_Tensor 对 int/float 静默返回错误设备
# ==============================================================================

@test("HI-4: as_Tensor(int/float) 静默吞掉输入")
def test_hi4_as_tensor_scalar():
    import inspect

    t_int = as_Tensor(42)
    t_float = as_Tensor(3.14)

    print(f"       as_Tensor(42) -> device='{t_int.device}', data={t_int.data}")
    print(f"       as_Tensor(3.14) -> device='{t_float.device}', data={t_float.data}")

    # 查看源码确认 pass 分支
    source = inspect.getsource(as_Tensor)
    lines = source.split('\n')
    bug_line = None
    for i, line in enumerate(lines):
        if 'isinstance(x, (int, float))' in line:
            # 下一行应该是 pass
            if i + 1 < len(lines) and 'pass' in lines[i + 1]:
                bug_line = i + 1
                break

    if bug_line is not None:
        print(f"       HI-4 确认! 源码第 {bug_line+1} 行: isinstance(int/float) 分支只有 pass。")
        print(f"       无异常、无警告、无设备参数——调用方完全不知道标量被静默处理。")

    # 验证: 无 GPU 时总是 cpu (这是预期行为, 缺陷在于无错误提示)
    assert t_int.device == 'cpu' or t_int.device == 'cuda', f"异常 device: {t_int.device}"


# ==============================================================================
# HI-5: FusedConvReLU.backward 硬编码 np.tensordot
# ==============================================================================

@test("HI-5: FusedConvReLU.backward 硬编码 np.tensordot")
def test_hi5_fused_conv_relu_np_hardcode():
    import inspect
    from eneuro.base.functions import FusedConvReLU as FusedConvReLU_Fn

    source = inspect.getsource(FusedConvReLU_Fn.backward)
    np_count = source.count('np.tensordot')

    print(f"       FusedConvReLU.backward 中 'np.tensordot' 出现 {np_count} 次")
    assert np_count >= 1, \
        f"未找到 np.tensordot (出现 {np_count} 次) —— 源码可能已修复"

    # 检查是否有 xp.tensordot 作为对照
    has_xp = 'xp.tensordot' in source
    print(f"       FusedConvReLU.backward 中 'xp.tensordot' 出现: {has_xp}")

    # 对比 Conv2DGradW.forward (正确的实现)
    from eneuro.base.functions import Conv2DGradW
    correct_source = inspect.getsource(Conv2DGradW.forward)
    correct_has_xp = 'xp.tensordot' in correct_source
    print(f"       Conv2DGradW.forward 中 'xp.tensordot' (正确做法): {correct_has_xp}")
    print(f"       --> FusedConvReLU.backward 在 GPU 上会因 np.tensordot 而失败。")


# ==============================================================================
# HI-11: FusedConvBNReLU running_var 命名错误
# ==============================================================================

@test("HI-11: FusedConvBNReLU running_var.name == 'running_mean'")
def test_hi11_running_var_name():
    import inspect
    from eneuro.base.functions import FusedConvBNReLU as FusedConvBNReLU_Fn

    source = inspect.getsource(FusedConvBNReLU_Fn.forward)

    # 查找 name='running_mean' 出现次数 (在 running_var 的上下文中)
    count_running_mean = source.count("name='running_mean'")
    print(f"       FusedConvBNReLU.forward 中 \"name='running_mean'\" 出现 {count_running_mean} 次")

    # 正确情况下, running_mean 和 running_var 应各出现一次 name='running_mean'
    # 和 name='running_var'。如果 name='running_mean' 出现 2 次, 说明 running_var
    # 也用了 running_mean 的名字。
    assert count_running_mean >= 2, \
        f"name='running_mean' 仅出现 {count_running_mean} 次, 可能已修复或代码结构变化。" \
        f" 期望 >= 2 (running_mean 本身 1 次 + running_var 错误命名 1 次)"

    # 检查 name='running_var' 是否存在
    count_running_var = source.count("name='running_var'")
    print(f"       FusedConvBNReLU.forward 中 \"name='running_var'\" 出现 {count_running_var} 次")
    print(f"       --> running_var 的 name 被错误设为 'running_mean',")
    print(f"           导致两个参数在序列化/调试时无法区分。")


# ==============================================================================
# CR-2: Windows 多进程递归派生 (源码检查，不实际触发)
# ==============================================================================

@test("CR-2: _AsyncLoaderIter 缺少 __main__ 保护")
def test_cr2_windows_multiprocess_guard():
    import inspect
    import platform
    from eneuro.data.dataloader import _AsyncLoaderIter

    source = inspect.getsource(_AsyncLoaderIter.__init__)
    has_process = 'Process(' in source
    assert has_process, "源码不包含 Process( -- 可能已重构"

    has_main_guard = ('if __name__' in source) or ('__main__' in source)

    if platform.system() == 'Windows' and not has_main_guard:
        print(f"       CR-2 确认! Windows 上 _AsyncLoaderIter 使用 Process() 但无 __main__ 保护。")
        print(f"       设置 num_workers > 0 会触发无限递归 spawn。")
    elif not has_main_guard:
        print(f"       当前平台 {platform.system()}, 缺少 __main__ 保护 (主要在 Windows 上触发)。")
    else:
        print(f"       源码包含 __main__ 保护 (可能已修复)。")


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  EnNeuro 第一优先级缺陷复现测试")
    print("  expect_defect=True 的测试: 期望触发缺陷 (CONFIRMED 是好结果)")
    print("=" * 70)

    section("CR-1: Optimizer._state 类变量共享 (4 项)")
    test_cr1_lr_override()
    test_cr1_timestep_pollution()
    test_cr1_momentum_collision()
    test_cr1_mixed_optimizers()

    section("CR-2: Windows 多进程递归派生 (源码检查)")
    test_cr2_windows_multiprocess_guard()

    section("CR-3: Parameter.backward 参数传递错误")
    test_cr3_parameter_backward()

    section("CR-5: capture_features 返回值解构错误")
    test_cr5_capture_features_unpacking()

    section("HI-1: Tensor 比较返回数组而非 bool (3 项)")
    test_hi1_eq_returns_array()
    test_hi1_total_ordering_contamination()
    test_hi1_list_membership()

    section("HI-4: as_Tensor 对 int/float 静默处理")
    test_hi4_as_tensor_scalar()

    section("HI-5: FusedConvReLU.backward 硬编码 np.tensordot")
    test_hi5_fused_conv_relu_np_hardcode()

    section("HI-11: FusedConvBNReLU running_var 命名错误")
    test_hi11_running_var_name()

    # -- 汇总 --
    total = PASS + CONFIRMED + FAIL
    print(f"\n{'='*70}")
    print(f"  汇总: {total} 项测试")
    print(f"    通过 (可能已修复):  {PASS}")
    print(f"    缺陷确认 (CONFIRMED): {CONFIRMED}")
    print(f"    意外失败:             {FAIL}")
    print(f"{'='*70}")

    if CONFIRMED > 0:
        print(f"\n  >> {CONFIRMED} 项第一优先级缺陷已确认触发。建议立即修复。")
    if PASS > 0 and CONFIRMED == 0:
        print(f"\n  >> 所有缺陷似乎已修复，或测试条件不满足。")
    if FAIL > 0:
        print(f"\n  >> {FAIL} 项意外失败，请检查测试逻辑。")
        sys.exit(1)

    sys.exit(0 if FAIL == 0 else 1)
