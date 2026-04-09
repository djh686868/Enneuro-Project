# graph.py
from __future__ import annotations
from typing import List, Set, Dict, Optional, Any, Tuple, Union
from enum import Enum, auto
import weakref
from collections import deque

from ..base.core import Tensor, Function

class NodeType(Enum):
    TENSOR = auto()
    FUNCTION = auto()

class Node:
    """计算图中的节点，包装 Tensor 或 Function 对象"""
    def __init__(self, obj: Any, node_id: int, node_type: NodeType):
        self.obj = obj                      # 原始 Tensor 或 Function 实例
        self.id = node_id                   # 唯一标识
        self.type = node_type               # 节点类型
        self.name = getattr(obj, 'name', None) or obj.__class__.__name__
        self.params = None

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type.name}, name={self.name})"

class Graph:
    """
    计算图数据结构，存储节点和边，支持子图匹配和替换。
    节点: Tensor 或 Function
    边: 从 Tensor 节点指向 Function 节点（数据流入），以及从 Function 节点指向 Tensor 节点（数据流出）。
    """
    def __init__(self):
        self.nodes: Dict[int, Node] = {}          # id -> Node
        self.tensor_nodes: Dict[int, Node] = {}   # 仅 Tensor 节点
        self.func_nodes: Dict[int, Node] = {}     # 仅 Function 节点
        self.input_edges: Dict[int, List[int]] = {}   # node_id -> list of predecessor node ids (输入边)
        self.output_edges: Dict[int, List[int]] = {}  # node_id -> list of successor node ids (输出边)
        self.next_id: int = 0
        self.obj_to_node: Dict[int, Node] = {}    # 对象 id 到节点的映射，避免重复添加

    def _new_id(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid

    def add_node(self, obj: Any) -> Node:
        """添加节点，如果对象已存在则返回已有节点"""
        obj_id = id(obj)
        if obj_id in self.obj_to_node:
            return self.obj_to_node[obj_id]

        if isinstance(obj, Tensor):
            ntype = NodeType.TENSOR
        elif isinstance(obj, Function):
            ntype = NodeType.FUNCTION
        else:
            raise TypeError(f"Cannot add node of type {type(obj)}")

        nid = self._new_id()
        node = Node(obj, nid, ntype)
        self.nodes[nid] = node
        self.obj_to_node[obj_id] = node
        if ntype == NodeType.TENSOR:
            self.tensor_nodes[nid] = node
        else:
            self.func_nodes[nid] = node
        # 初始化边字典
        self.input_edges[nid] = []
        self.output_edges[nid] = []
        return node

    def add_edge(self, from_node: Node, to_node: Node):
        """添加从 from_node 到 to_node 的有向边"""
        if from_node.type == to_node.type:
            raise ValueError("Edges must connect different node types (Tensor <-> Function)")
        # 避免重复添加边
        if to_node.id not in self.output_edges[from_node.id]:
            self.output_edges[from_node.id].append(to_node.id)
            self.input_edges[to_node.id].append(from_node.id)

    def get_predecessors(self, node: Node) -> List[Node]:
        """获取节点的所有前驱节点"""
        return [self.nodes[pid] for pid in self.input_edges[node.id]]

    def get_successors(self, node: Node) -> List[Node]:
        """获取节点的所有后继节点"""
        return [self.nodes[sid] for sid in self.output_edges[node.id]]

    def topological_order(self) -> List[Node]:
        """返回拓扑排序的节点列表（从输入到输出）"""
        in_degree = {nid: len(self.input_edges[nid]) for nid in self.nodes}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            nid = queue.popleft()
            order.append(self.nodes[nid])
            for succ_id in self.output_edges[nid]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return order

    def replace_subgraph(self, subgraph_nodes: List[Node], 
                         inputs: List[Node], outputs: List[Node],
                         fused_func: Function) -> Node:
        """
        将子图替换为融合函数节点。
        subgraph_nodes: 子图中的所有节点（Tensor 和 Function）
        inputs: 子图外流入子图的 Tensor 节点（子图的输入）
        outputs: 子图流出到外部的 Tensor 节点（子图的输出）
        fused_func: 融合后的 Function 实例
        """
        subgraph_set = set(subgraph_nodes)   # 转换为集合，便于快速判断

        for inp in inputs:
            assert inp not in subgraph_set, "Input node must be outside subgraph"
        for out in outputs:
            assert out not in subgraph_set, "Output node must be outside subgraph"

        # 1. 创建新的 Function 节点
        fused_node = self.add_node(fused_func)
        
        # 2. 连接输入边：每个输入 Tensor -> 新节点
        for inp_node in inputs:
            if inp_node.type != NodeType.TENSOR:
                raise ValueError("Subgraph input must be Tensor node")
            self.add_edge(inp_node, fused_node)
            # 移除子图内从 inp_node 出发的旧边
            self._remove_edges_from_node(inp_node, subgraph_set)
        
        # 3. 连接输出边：新节点 -> 每个输出 Tensor
        for out_node in outputs:
            if out_node.type != NodeType.TENSOR:
                raise ValueError("Subgraph output must be Tensor node")
            self.add_edge(fused_node, out_node)
            # 移除指向 out_node 的旧边（来自子图内部）
            self._remove_edges_to_node(out_node, subgraph_set)
        
        # 4. 删除子图中的所有节点
        for node in subgraph_nodes:
            self._remove_node(node)
        
        return fused_node

    def _remove_edges_from_node(self, node: Node, keep_set: Set[Node]):
        """移除 node 指向 keep_set 中节点的边（仅当目标在子图内）"""
        new_succ = []
        for succ_id in self.output_edges[node.id]:
            succ_node = self.nodes[succ_id]
            if succ_node in keep_set:
                # 移除边：从 node 的 output_edges 中删除，并从 succ 的 input_edges 中删除
                self.input_edges[succ_id].remove(node.id)
                continue
            new_succ.append(succ_id)
        self.output_edges[node.id] = new_succ

    def _remove_edges_to_node(self, node: Node, keep_set: Set[Node]):
        """移除 keep_set 中节点指向 node 的边"""
        new_pred = []
        for pred_id in self.input_edges[node.id]:
            pred_node = self.nodes[pred_id]
            if pred_node in keep_set:
                self.output_edges[pred_id].remove(node.id)
                continue
            new_pred.append(pred_id)
        self.input_edges[node.id] = new_pred

    def _remove_node(self, node: Node):
        """完全删除节点及其所有关联边"""
        # 删除所有出边
        for succ_id in self.output_edges[node.id]:
            self.input_edges[succ_id].remove(node.id)
        # 删除所有入边
        for pred_id in self.input_edges[node.id]:
            self.output_edges[pred_id].remove(node.id)
        # 从字典中删除
        del self.nodes[node.id]
        if node.type == NodeType.TENSOR:
            del self.tensor_nodes[node.id]
        else:
            del self.func_nodes[node.id]
        del self.input_edges[node.id]
        del self.output_edges[node.id]
        # 从对象映射中删除（注意：可能有多个对象指向同一个 obj？这里假设每个 obj 只出现一次）
        obj_id = id(node.obj)
        if obj_id in self.obj_to_node:
            del self.obj_to_node[obj_id]

    def clone(self) -> Graph:
        """深拷贝图（不拷贝原始对象，只拷贝结构和引用）"""
        new_graph = Graph()
        old_to_new = {}
        for nid, node in self.nodes.items():
            new_node = new_graph.add_node(node.obj)
            if hasattr(node, 'params'):
                new_node.params = node.params   # 复制参数
            old_to_new[nid] = new_node
        for nid, node in self.nodes.items():
            for succ_id in self.output_edges[nid]:
                new_graph.add_edge(old_to_new[nid], old_to_new[succ_id])
        return new_graph

    def visualize(self, filename: str = "graph.dot"):
        """导出为 Graphviz dot 格式用于可视化"""
        lines = ["digraph G {"]
        for node in self.nodes.values():
            label = f"{node.name}\\nid:{node.id}"
            shape = "ellipse" if node.type == NodeType.TENSOR else "box"
            lines.append(f'  n{node.id} [label="{label}", shape={shape}];')
        for nid, succ_list in self.output_edges.items():
            for succ_id in succ_list:
                lines.append(f"  n{nid} -> n{succ_id};")
        lines.append("}")
        with open(filename, "w") as f:
            f.write("\n".join(lines))