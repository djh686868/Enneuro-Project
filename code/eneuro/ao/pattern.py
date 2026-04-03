# pattern.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Type, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from .graph import Graph, Node, NodeType
from ..base.core import Function


class MatchStrategy(Enum):
    """匹配策略：EXACT 要求节点类型完全一致，SUBTYPE 允许子类匹配"""
    EXACT = auto()
    SUBTYPE = auto()


@dataclass
class NodeMatcher:
    """
    单个节点的匹配条件。
    """
    # 匹配的函数类型（可以是类或类名）
    func_type: Union[Type[Function], str, None] = None
    # 可选的参数匹配条件，例如 {'stride': (1,1), 'pad': (0,0)}
    params: Dict[str, Any] = field(default_factory=dict)
    # 匹配策略
    strategy: MatchStrategy = MatchStrategy.EXACT

    def matches(self, node: Node) -> bool:
        """检查节点是否满足匹配条件"""
        if node.type != NodeType.FUNCTION:
            return False

        obj = node.obj
        # 类型匹配
        if self.func_type is not None:
            if self.strategy == MatchStrategy.EXACT:
                if type(obj) != self.func_type:
                    return False
            else:  # SUBTYPE
                if not isinstance(obj, self.func_type):
                    return False

        # 参数匹配
        node_params = getattr(node, 'params', {})
        for key, expected in self.params.items():
            actual = node_params.get(key)
            if actual != expected:
                return False
        return True


@dataclass
class FusionPattern:
    """
    描述一个可融合的算子序列模式。
    目前支持线性链式结构：A -> B -> C ...，每个节点为 Function。
    """
    name: str                                    # 模式名称，如 "conv_relu"
    node_matchers: List[NodeMatcher]             # 顺序匹配的 Function 节点条件
    # 可选：额外的边约束（例如需要输入 Tensor 只被当前子图使用等，暂不实现）
    
    # 对应的融合算子类（必须是 Function 子类）
    fused_class: Optional[Type[Function]] = None

    def __post_init__(self):
        # 验证至少有一个节点
        if not self.node_matchers:
            raise ValueError("FusionPattern must have at least one node matcher")

    def match_start(self, graph: Graph, start_node: Node) -> Optional[MatchResult]:
        """
        从 start_node 开始尝试匹配模式。
        start_node 必须是 Function 节点，且是模式的第一个节点。
        如果匹配成功，返回 MatchResult；否则返回 None。
        """
        if start_node.type != NodeType.FUNCTION:
            return None

        matched_nodes: List[Node] = []
        current = start_node
        for i, matcher in enumerate(self.node_matchers):
            if not matcher.matches(current):
                return None
            matched_nodes.append(current)

            # 如果不是最后一个节点，则沿着输出边找到下一个 Function 节点
            if i < len(self.node_matchers) - 1:
                # 获取当前节点的输出 Tensor 节点
                succ_tensors = [graph.nodes[sid] for sid in graph.output_edges[current.id]
                                if graph.nodes[sid].type == NodeType.TENSOR]
                if not succ_tensors:
                    return None
                # 通常取第一个输出 Tensor（对于单输出算子）
                # 然后找到该 Tensor 的下一个 Function 节点
                next_func = None
                for tensor_node in succ_tensors:
                    func_succ = [graph.nodes[succ_id] for succ_id in graph.output_edges[tensor_node.id]
                                 if graph.nodes[succ_id].type == NodeType.FUNCTION]
                    if func_succ:
                        # 如果多个输出连接到不同 Function，这里取第一个（简单线性假设）
                        next_func = func_succ[0]
                        break
                if next_func is None:
                    return None
                current = next_func

        # 匹配成功，计算子图的输入和输出 Tensor 节点
        # 输入：第一个节点的输入 Tensor 节点（前驱 Tensor 节点，且不在子图内）
        first_node = matched_nodes[0]
        input_tensors = []
        for pred_id in graph.input_edges[first_node.id]:
            pred_node = graph.nodes[pred_id]
            if pred_node.type == NodeType.TENSOR:
                input_tensors.append(pred_node)

        # 输出：最后一个节点的输出 Tensor 节点
        last_node = matched_nodes[-1]
        output_tensors = []
        for succ_id in graph.output_edges[last_node.id]:
            succ_node = graph.nodes[succ_id]
            if succ_node.type == NodeType.TENSOR:
                output_tensors.append(succ_node)

        # 如果输入/输出有多个，可能需要更复杂的处理，目前按单输入单输出处理
        if not input_tensors or not output_tensors:
            return None

        return MatchResult(
            pattern=self,
            matched_nodes=matched_nodes,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )


@dataclass
class MatchResult:
    """存储一个模式匹配的结果"""
    pattern: FusionPattern
    matched_nodes: List[Node]          # 匹配到的 Function 节点列表（按顺序）
    input_tensors: List[Node]          # 子图外流入的 Tensor 节点
    output_tensors: List[Node]         # 子图流出的 Tensor 节点

    def __post_init__(self):
        # 确保所有 matched_nodes 都是 FUNCTION 类型
        for node in self.matched_nodes:
            if node.type != NodeType.FUNCTION:
                raise ValueError("matched_nodes must contain only FUNCTION nodes")
    
    def get_subgraph_nodes(self, graph: Graph) -> List[Node]:
        """返回子图中所有节点（包括中间 Tensor 节点）"""
        nodes = set(self.matched_nodes)
        # 收集 matched_nodes 之间的中间 Tensor
        # 遍历每对相邻的 Function 节点，获取它们之间的 Tensor
        for i in range(len(self.matched_nodes) - 1):
            func_a = self.matched_nodes[i]
            func_b = self.matched_nodes[i + 1]
            # 查找从 func_a 输出到 func_b 输入的 Tensor 节点
            for out_tensor_id in graph.output_edges[func_a.id]:
                out_tensor = graph.nodes[out_tensor_id]
                if out_tensor.type == NodeType.TENSOR:
                    # 检查该 Tensor 是否连接到 func_b
                    if func_b.id in graph.output_edges[out_tensor.id]:
                        nodes.add(out_tensor)
        return list(nodes)

    def replace(self, graph: Graph, fused_func: Function) -> Node:
        """
        使用融合后的 Function 节点替换当前匹配的子图。
        返回新创建的 Function 节点。
        """
        # 调用 Graph.replace_subgraph 方法
        fused_node = graph.replace_subgraph(
            subgraph_nodes=self.get_subgraph_nodes(graph),  # 需要完整节点集合
            inputs=self.input_tensors,
            outputs=self.output_tensors,
            fused_func=fused_func
        )
        return fused_node


# 预定义常用模式，供 FusionRegistry 使用
def create_conv_relu_pattern() -> FusionPattern:
    from ..base.functions import Conv2d, ReLU
    return FusionPattern(
        name="conv_relu",
        node_matchers=[
            NodeMatcher(func_type=Conv2d),
            NodeMatcher(func_type=ReLU),
        ],
        fused_class=None  # 外部设置，例如 FusedConvReLU
    )


def create_conv_bn_relu_pattern() -> FusionPattern:
    from ..base.functions import Conv2d, BatchNorm2d, ReLU
    return FusionPattern(
        name="conv_bn_relu",
        node_matchers=[
            NodeMatcher(func_type=Conv2d),
            NodeMatcher(func_type=BatchNorm2d),
            NodeMatcher(func_type=ReLU),
        ],
        fused_class=None
    )