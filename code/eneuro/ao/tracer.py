# tracer.py
import weakref
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from ..base.core import Tensor, Function, Config
from .graph import Graph, Node, NodeType

class Tracer:
    """
    图记录器：在 Function.__call__ 中被调用，逐步构建计算图。
    使用全局 Config.record_graph 和 Config.current_tracer 控制激活状态。
    """
    def __init__(self):
        self.graph = Graph()
        # 可选：缓存 Function 的参数提取结果
        self._func_params_cache: Dict[int, Dict] = {}

    def record(self, func: Function, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        """
        记录一次 Function 调用，构建节点和边。
        参数：
            func   : 当前调用的 Function 实例
            inputs : 输入 Tensor 列表（已转为 Tensor）
            outputs: 输出 Tensor 列表
        """
        # 1. 获取或创建输入 Tensor 节点
        input_nodes = [self._get_or_create_tensor_node(t) for t in inputs]
        # 2. 获取或创建输出 Tensor 节点
        output_nodes = [self._get_or_create_tensor_node(t) for t in outputs]
        # 3. 获取或创建 Function 节点（附带参数）
        func_node = self._get_or_create_func_node(func)

        # 4. 建立数据流边：输入 -> 函数
        for inp_node in input_nodes:
            self.graph.add_edge(inp_node, func_node)
        # 5. 建立数据流边：函数 -> 输出
        for out_node in output_nodes:
            self.graph.add_edge(func_node, out_node)

    def _get_or_create_tensor_node(self, tensor: Tensor) -> Node:
        """返回 Tensor 对应的节点（自动去重）"""
        return self.graph.add_node(tensor)

    def _get_or_create_func_node(self, func: Function) -> Node:
        """返回 Function 对应的节点，并附加参数信息"""
        node = self.graph.add_node(func)
        # 如果尚未提取参数，则提取并缓存
        func_id = id(func)
        if func_id not in self._func_params_cache:
            self._func_params_cache[func_id] = self._extract_params(func)
        node.params = self._func_params_cache[func_id]   # 附加到节点上，方便模式匹配
        return node

    def _extract_params(self, func: Function) -> Dict[str, Any]:
        """
        提取 Function 的关键参数，用于后续的模式匹配。
        可根据需要扩展提取的属性列表。
        """
        params = {'class_name': func.__class__.__name__}
        # 常见于卷积、池化等算子的参数
        common_attrs = ['stride', 'pad', 'kernel_size', 'axis', 'keepdims',
                        'momentum', 'eps', 'outsize', 'to_matrix']
        for attr in common_attrs:
            if hasattr(func, attr):
                val = getattr(func, attr)
                # 若为 pair 转换后的元组，直接存储
                params[attr] = val
        return params

    def get_graph(self) -> Graph:
        """返回构建完成的计算图"""
        return self.graph

    def clear(self) -> None:
        """清空当前记录的数据（重新开始）"""
        self.graph = Graph()
        self._func_params_cache.clear()


@contextmanager
def trace_context():
    """
    上下文管理器：在此上下文中执行的前向传播会被自动记录。
    用法：
        with trace_context() as tracer:
            output = model(sample_input)
            graph = tracer.get_graph()
    """
    old_record_flag = Config.record_graph
    old_tracer = Config.current_tracer
    tracer = Tracer()
    Config.record_graph = True
    Config.current_tracer = tracer
    try:
        yield tracer
    finally:
        Config.record_graph = old_record_flag
        Config.current_tracer = old_tracer