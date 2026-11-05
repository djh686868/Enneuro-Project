"""
带调试信息的钩子系统
"""
import weakref
from typing import Callable, Dict, Any, Optional, List
import numpy as np


class HookHandle:
    """钩子句柄，用于管理钩子的生命周期"""

    def __init__(self, hook_id: int, remove_fn: Callable[[int], None]):
        self.hook_id = hook_id
        self._remove_fn = remove_fn
        self._removed = False

    def remove(self):
        """移除钩子"""
        if not self._removed:
            self._remove_fn(self.hook_id)
            self._removed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class HookManager:
    """钩子管理器 - 管理前向和反向传播钩子"""

    def __init__(self):
        self._forward_hooks: Dict[int, Callable] = {}
        self._backward_hooks: Dict[int, Callable] = {}
        self._hook_counter = 0

    def register_forward_hook(self, hook_fn: Callable) -> HookHandle:
        """注册前向传播钩子"""
        hook_id = self._hook_counter
        self._hook_counter += 1
        self._forward_hooks[hook_id] = hook_fn

        def remove_func(id_to_remove: int):
            if id_to_remove in self._forward_hooks:
                del self._forward_hooks[id_to_remove]

        return HookHandle(hook_id, remove_func)

    def register_backward_hook(self, hook_fn: Callable) -> HookHandle:
        """注册反向传播钩子"""
        hook_id = self._hook_counter
        self._hook_counter += 1
        self._backward_hooks[hook_id] = hook_fn

        def remove_func(id_to_remove: int):
            if id_to_remove in self._backward_hooks:
                del self._backward_hooks[id_to_remove]

        return HookHandle(hook_id, remove_func)

    def trigger_forward_hooks(self, *args, **kwargs):
        """触发所有前向钩子"""
        for hook in self._forward_hooks.values():
            hook(*args, **kwargs)

    def trigger_backward_hooks(self, grad_inputs, grad_outputs):
        """触发所有反向钩子"""
        for hook in self._backward_hooks.values():
            hook(grad_inputs, grad_outputs)

    def clear_all_hooks(self):
        """清除所有钩子"""
        self._forward_hooks.clear()
        self._backward_hooks.clear()


class HookRegistry:
    """全局钩子注册表"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._function_to_layer = weakref.WeakKeyDictionary()
            cls._instance._current_layer = None
            # 前向传播层调用顺序追踪
            cls._instance._layer_call_sequence = []   # 本轮 forward 中层 id 的调用序列
            cls._instance._id_to_layer_ref = {}       # id -> weakref(layer)
            cls._instance._recording_sequence = False
        return cls._instance

    def register_function_layer_mapping(self, function, layer):
        self._function_to_layer[function] = weakref.ref(layer)

    def get_layer_for_function(self, function):
        if function in self._function_to_layer:
            return self._function_to_layer[function]()
        return None

    def set_current_layer(self, layer):
        self._current_layer = layer

    def get_current_layer(self):
        return self._current_layer

    # ── 层调用顺序追踪 ──────────────────────────────────────────────────────
    def start_recording_sequence(self):
        """开始记录本轮前向传播的层调用顺序"""
        self._layer_call_sequence = []
        self._id_to_layer_ref = {}
        self._recording_sequence = True

    def stop_recording_sequence(self):
        """停止记录"""
        self._recording_sequence = False

    def record_layer_call(self, layer):
        """在 layer_call_with_hooks 中调用，记录每次层调用"""
        if self._recording_sequence:
            lid = id(layer)
            self._layer_call_sequence.append(lid)
            if lid not in self._id_to_layer_ref:
                self._id_to_layer_ref[lid] = weakref.ref(layer)

    def get_successor_layer(self, target_layer):
        """
        返回本轮前向传播中紧跟在 target_layer 之后调用的层对象。
        若 target_layer 不在序列中或已是最后一层，则返回 None。
        """
        target_id = id(target_layer)
        seq = self._layer_call_sequence
        try:
            idx = seq.index(target_id)
            if idx + 1 < len(seq):
                successor_id = seq[idx + 1]
                ref = self._id_to_layer_ref.get(successor_id)
                if ref is not None:
                    return ref()
        except ValueError:
            pass
        return None


def add_hooks_to_module():
    """向现有的 Layer 和 Module 类添加钩子功能"""
    from ..nn.module import Layer
    from ..base.core import Function

    original_layer_call = Layer.__call__
    original_function_call = Function.__call__
    original_function_backward = Function.backward

    hook_registry = HookRegistry()

    def layer_call_with_hooks(self, *inputs):
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
            self._saved_inputs = None
            self._saved_outputs = None

        self._saved_inputs = [weakref.ref(x) for x in inputs]

        # 记录本层调用顺序，供 GradCAM 等可解释性工具使用
        hook_registry.record_layer_call(self)

        hook_registry.set_current_layer(self)

        outputs = original_layer_call(self, *inputs)

        hook_registry.set_current_layer(None)

        if not isinstance(outputs, tuple):
            output_tuple = (outputs,)
        else:
            output_tuple = outputs
        self._saved_outputs = [weakref.ref(y) for y in output_tuple]

        self._hook_manager.trigger_forward_hooks(self, inputs, outputs)

        return outputs

    Layer.__call__ = layer_call_with_hooks

    def register_forward_hook(self, hook_fn: Callable) -> HookHandle:
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
        return self._hook_manager.register_forward_hook(hook_fn)

    def register_backward_hook(self, hook_fn: Callable) -> HookHandle:
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()
        return self._hook_manager.register_backward_hook(hook_fn)

    Layer.register_forward_hook = register_forward_hook
    Layer.register_backward_hook = register_backward_hook

    def function_call_with_hooks(self, *inputs):
        if not hasattr(self, '_hook_manager'):
            self._hook_manager = HookManager()

        current_layer = hook_registry.get_current_layer()
        if current_layer is not None:
            hook_registry.register_function_layer_mapping(self, current_layer)

        self._hook_manager.trigger_forward_hooks(self, inputs, None, phase='pre')

        outputs = original_function_call(self, *inputs)

        self._hook_manager.trigger_forward_hooks(self, inputs, outputs, phase='post')

        # 将每个函数实例的 backward 包装为实例属性。
        # 实例属性优先于类方法，所以无论子类如何覆盖 backward，
        # 这里的包装都能被正确拦截，而 type(self).backward 则
        # 精确调用该子类自身的真实 backward 实现。
        #
        # ⚠ 内存泄漏修复：
        #   原来闭包直接捕获 self（强引用），同时 self.backward 又持有该闭包，
        #   形成 "Function → closure → Function" 的引用环。
        #   Python 引用计数无法自动回收环形引用，导致整条计算图（Function +
        #   其 inputs Tensor 链）在 backward 结束后仍驻留内存。
        #   修复方案：闭包只持有 weakref.ref(self)，需要时再解引用。
        #   弱引用不增加引用计数，引用环因此被打断，计算图可被正常回收。
        real_backward = type(self).backward
        ref = weakref.ref(self)  # 弱引用，避免循环引用

        def instance_backward_with_hooks(gy):
            fn = ref()  # ref
            if fn is None:
                # 对象已被 GC（正常 backward 调用栈中不应发生）
                raise RuntimeError("Function 实例已被 GC，无法执行 backward")

            grad_inputs = real_backward(fn, gy)

            if not isinstance(grad_inputs, tuple):
                grad_inputs_tuple = (grad_inputs,)
            else:
                grad_inputs_tuple = grad_inputs

            grad_outputs = (gy,) if not isinstance(gy, tuple) else gy

            if fn._hook_manager._backward_hooks:
                fn._hook_manager.trigger_backward_hooks(grad_inputs_tuple, grad_outputs)

            layer = hook_registry.get_layer_for_function(fn)
            if layer is not None and hasattr(layer, '_hook_manager') and layer._hook_manager._backward_hooks:
                layer._hook_manager.trigger_backward_hooks(grad_inputs_tuple, grad_outputs)

            return grad_inputs

        self.backward = instance_backward_with_hooks

        return outputs

    Function.__call__ = function_call_with_hooks

    Function.register_forward_hook = register_forward_hook
    Function.register_backward_hook = register_backward_hook


def _to_numpy(arr):
    """将 numpy / cupy 数组统一转换为 numpy ndarray，供后处理使用。"""
    if arr is None:
        return None
    if hasattr(arr, 'get'):          # cupy ndarray
        return arr.get()
    if hasattr(arr, 'data'):         # Tensor
        return _to_numpy(arr.data)
    return np.asarray(arr)


def capture_features(layer):
    """捕获特征图（前向输出），始终以 numpy ndarray 存储"""
    storage = {'input': None, 'output': None}

    def hook(module, inputs, outputs):
        storage['input'] = [
            _to_numpy(x.data if hasattr(x, 'data') else x)
            for x in inputs
        ]
        storage['output'] = _to_numpy(outputs.data if hasattr(outputs, 'data') else outputs)

    handle = layer.register_forward_hook(hook)
    return handle, storage


def capture_gradients(layer):
    """捕获反向梯度，始终以 numpy ndarray 存储"""
    storage = {'grad_output': None, 'grad_input': None}

    def hook(grad_inputs, grad_outputs):
        if grad_outputs and len(grad_outputs) > 0 and grad_outputs[0] is not None:
            storage['grad_output'] = _to_numpy(
                grad_outputs[0].data if hasattr(grad_outputs[0], 'data') else grad_outputs[0]
            )

        if grad_inputs:
            storage['grad_input'] = [
                _to_numpy(g.data if hasattr(g, 'data') else g) if g is not None else None
                for g in grad_inputs
            ]

    handle = layer.register_backward_hook(hook)
    return handle, storage


def backward_hook(layer):
    """装饰器：为层添加反向钩子"""
    def decorator(hook_fn: Callable):
        handle = layer.register_backward_hook(hook_fn)
        return handle
    return decorator


add_hooks_to_module()
