# executor.py
from typing import List, Union, Dict, Any, Optional
from ..base.core import Tensor, Function
from ..base.parameter import Parameter
from .graph import Graph, Node, NodeType

class GraphExecutor:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.topo_order = graph.topological_order()
        
        # 1. 识别参数节点和数据输入节点
        self.param_nodes: List[Node] = []      # Parameter 节点
        self.data_input_nodes: List[Node] = [] # 数据输入节点（非 Parameter）
        
        for node in self.graph.nodes.values():
            if node.type != NodeType.TENSOR:
                continue
            # 没有入边的 Tensor 节点
            if not self.graph.input_edges[node.id]:
                if isinstance(node.obj, Parameter):
                    self.param_nodes.append(node)
                else:
                    self.data_input_nodes.append(node)

    def forward(self, *inputs: Tensor) -> Union[Tensor, List[Tensor]]:
        """
        执行优化图的前向传播
        inputs: 仅对应数据输入节点（按 self.data_input_nodes 顺序）
        """
        # 校验输入数量
        if len(inputs) != len(self.data_input_nodes):
            raise ValueError(
                f"Expected {len(self.data_input_nodes)} data inputs, got {len(inputs)}"
            )
        
        cache: Dict[int, Tensor] = {}
        
        # 1. 将参数节点填入缓存（直接使用 Parameter 对象，其 data 属性会自动更新）
        for node in self.param_nodes:
            cache[node.id] = node.obj   # node.obj 就是 Parameter 实例
        
        # 2. 将数据输入节点填入缓存
        for node, tensor in zip(self.data_input_nodes, inputs):
            cache[node.id] = tensor
        
        # 3. 按拓扑序执行所有 Function 节点
        for node in self.topo_order:
            if node.type == NodeType.TENSOR:
                continue  # Tensor 节点的值已由上游 Function 产生
            
            func: Function = node.obj
            # 获取输入 Tensor 节点（前驱）
            input_nodes = [
                self.graph.nodes[pred_id] for pred_id in self.graph.input_edges[node.id]
            ]
            input_tensors = [cache[in_node.id] for in_node in input_nodes]
            
            # 调用 Function（自动建立计算图，支持反向传播）
            outputs = func(*input_tensors)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            
            # 获取输出 Tensor 节点（后继）
            output_nodes = [
                self.graph.nodes[succ_id] for succ_id in self.graph.output_edges[node.id]
            ]
            if len(outputs) != len(output_nodes):
                raise RuntimeError(
                    f"Function {func} returned {len(outputs)} outputs, "
                    f"graph expects {len(output_nodes)}"
                )
            
            # 缓存输出
            for out_node, out_tensor in zip(output_nodes, outputs):
                cache[out_node.id] = out_tensor
        
        # 4. 收集最终输出（无出边的 TensorNode）
        output_nodes = [
            n for n in self.graph.nodes.values()
            if n.type == NodeType.TENSOR and not self.graph.output_edges[n.id]
        ]
        results = [cache[n.id] for n in output_nodes]
        return results[0] if len(results) == 1 else results