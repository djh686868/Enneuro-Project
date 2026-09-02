# executor.py
import numpy as np
from typing import List, Union, Dict, Any, Optional
from ..base.core import Tensor, Function
from ..base.parameter import Parameter
from .graph import Graph, Node, NodeType
from ..utils import StateDict

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False


class GraphExecutor(StateDict):
    def __init__(self, graph: Graph):
        self.graph = graph
        self.topo_order = graph.topological_order()
        
        # 1. 识别参数节点和数据输入节点
        self.param_nodes: List[Node] = []      # Parameter 节点
        self.data_input_nodes: List[Node] = [] # 数据输入节点（非 Parameter）
        
        for node in self.graph.topological_order():
            if node.type != NodeType.TENSOR:
                continue
            # 没有入边的 Tensor 节点
            if not self.graph.input_edges[node.id]:
                if isinstance(node.true_obj, Parameter):
                    self.param_nodes.append(node)
                else:
                    self.data_input_nodes.append(node)

    def params(self) -> List[Parameter]:
        return [node.true_obj for node in self.param_nodes]

    def _to_pure_list(self, tensor_like):
        if tensor_like is None:
            return None
        if isinstance(tensor_like, np.ndarray):
            return tensor_like.tolist()
        if has_cupy and isinstance(tensor_like, cp.ndarray):
            return cp.asnumpy(tensor_like).tolist()
        if isinstance(tensor_like, (list, tuple)):
            return [self._to_pure_list(item) for item in tensor_like]
        if isinstance(tensor_like, (int, float, bool, str)):
            return tensor_like
        try:
            return float(tensor_like)
        except (TypeError, ValueError):
            return str(tensor_like)

    def _from_pure_list(self, pure_list):
        if isinstance(pure_list, list):
            return [self._from_pure_list(item) for item in pure_list]
        return pure_list

    def to_dict(self) -> dict:
        """序列化当前 executor 的参数状态。"""
        params_dict = {}
        for node in self.param_nodes:
            param = node.true_obj
            if not isinstance(param, Parameter):
                continue

            data = param.data
            grad = param.grad.data if param.grad is not None and hasattr(param.grad, 'data') else None

            params_dict[str(node.id)] = {
                'node_id': node.id,
                'name': getattr(param, 'name', None),
                'data': self._to_pure_list(data),
                'grad': self._to_pure_list(grad) if grad is not None else None,
                'requires_grad': getattr(param, 'requires_grad', True),
                'shape': list(data.shape) if data is not None and hasattr(data, 'shape') else None,
                'dtype': str(data.dtype) if data is not None and hasattr(data, 'dtype') else None,
            }

        return {
            'metadata': {
                'graph_class': self.graph.__class__.__name__,
                'version': '1.0',
                'param_count': len(self.param_nodes),
                'data_input_count': len(self.data_input_nodes),
            },
            'graph': {
                'topo_order': [n.id for n in self.topo_order],
                'param_node_ids': [n.id for n in self.param_nodes],
                'data_input_node_ids': [n.id for n in self.data_input_nodes],
                'edges': [
                    [src_id, dst_id]
                    for src_id, succs in self.graph.output_edges.items()
                    for dst_id in succs
                ],
            },
            'params': params_dict,
        }
        
    def from_dict(self, d: dict) -> None:
        """按现有图中的参数节点，还原参数值和梯度状态。"""
        params_data = d.get('params', {})
        current_by_id = {node.id: node.true_obj for node in self.param_nodes}

        if 'graph' in d:
            graph_meta = d['graph']
            expected_ids = set(graph_meta.get('param_node_ids', []))
            if expected_ids and expected_ids != {node.id for node in self.param_nodes}:
                # 兼容性校验：图参数集合不一致时仅忽略，不重建图对象
                pass

        for raw_key, param_data in params_data.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                key = raw_key

            param = None
            if key in current_by_id:
                param = current_by_id[key]
            else:
                for node in self.param_nodes:
                    p = node.true_obj
                    if getattr(p, 'name', None) == param_data.get('name'):
                        param = p
                        break

            if param is None:
                continue

            xp = np
            if hasattr(param, 'data') and param.data is not None:
                xp = np
                if has_cupy and isinstance(param.data, cp.ndarray):
                    xp = cp

            if param_data.get('data') is not None:
                param.data = xp.array(self._from_pure_list(param_data['data']))

            grad_value = param_data.get('grad')
            if grad_value is not None:
                grad_tensor = Tensor(
                    xp.array(self._from_pure_list(grad_value)),
                    requires_grad=False,
                    name=getattr(param, 'name', None),
                    device='cuda' if xp is cp else 'cpu'
                )
                param.grad = grad_tensor
            else:
                param.grad = None

            param.requires_grad = param_data.get('requires_grad', getattr(param, 'requires_grad', True))
            if 'name' in param_data:
                param.name = param_data['name']

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
            cache[node.id] = node.true_obj   # node.true_obj 就是 Parameter 实例
        
        # 2. 将数据输入节点填入缓存
        for node, tensor in zip(self.data_input_nodes, inputs):
            cache[node.id] = tensor
        
        # 3. 按拓扑序执行所有 Function 节点
        for node in self.topo_order:
            if node.type == NodeType.TENSOR:
                continue  # Tensor 节点的值已由上游 Function 产生
            
            func: Function = node.true_obj
            # 获取输入 Tensor 节点（前驱）
            input_nodes = [
                self.graph.nodes[pred_id] for pred_id in self.graph.input_edges[node.id]
            ]
            input_tensors = [cache[in_node.id] for in_node in input_nodes]
            
            # 调用 Function（自动建立计算图，支持反向传播）
            outputs = func(*input_tensors)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            #print(f"func={func} outputs={outputs}")
            
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