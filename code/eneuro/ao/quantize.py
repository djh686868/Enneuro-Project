from __future__ import annotations
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import numpy as np
import weakref

from .graph import Graph, Node, NodeType
from .executor import GraphExecutor

from ..base.core import Tensor, Function, Config
from ..nn.optim import Optimizer
from ..base.functions import to_xp, get_array_module
from ..base import functions as f

class QuantizeManager:
    @staticmethod
    def apply_quantize(executor: GraphExecutor) -> GraphExecutor:
        graph = executor.graph
        # 按拓扑序遍历所有 Function 节点
        for node in graph.topological_order():
            if node.type != NodeType.FUNCTION: # 排除tensor
                continue

            # 遇到FQ节点
            if isinstance(node.true_obj, f.FakeQuantize):
                scale, zero_point, dtype = node.true_obj.scale, node.true_obj.zero_point, node.true_obj.dtype

                # 查找量化子图
                between_func_nodes, end_output_nodes, end_fdq_nodes = QuantizeManager.get_sub_quantize_nodes(graph=graph, start_fq_node=node)
                if len(end_output_nodes) > 0:
                    raise RuntimeError(f"存在未闭合的FakeQuantize: node.id={node.id}")
                
                # FQ替换为Quantize
                new_func = f.Quantize(scale, zero_point, dtype)
                graph.replace_subgraph([node], graph.get_predecessors(node), graph.get_successors(node), new_func)
                # FDQ替换为Dequantize
                for fdq_node in end_fdq_nodes:
                    new_func = f.Dequantize(scale, zero_point, dtype)
                    graph.replace_subgraph([fdq_node], graph.get_predecessors(fdq_node), graph.get_successors(fdq_node), new_func)

                for between_func_node in between_func_nodes:
                    # 可量化
                    if between_func_node.true_obj.__class__ in f.QuantizeRegistry.can_quantize:
                        # func_node <- pre_node(Tensor)
                        pre_nodes = graph.get_predecessors(between_func_node)
                        for pre_node in pre_nodes:
                            # 参数节点直接量化
                            if pre_node in executor.param_nodes:
                                pre_node.dtype = dtype
                                assert isinstance(pre_node.true_obj, Tensor)
                                pre_node.true_obj.data = pre_node.true_obj.data.astype(dtype)
                            # 其他节点忽略（不处理 未量化节点->func_node 的情况）
                            else:
                                pass
                    # 不可量化
                    else:
                        '''
                        pre_node -> 
                        between_func_node -> suc_node ->
                        suc_func
                        变为
                        pre_node -> 
                        [Dequantize -> Tensor ->] 
                        between_func_node -> suc_node ->
                        [Quantize -> Tensor ->]
                        suc_func
                        '''
                        pre_nodes = graph.get_predecessors(between_func_node)
                        # 去除原来的边
                        graph._remove_edges_to_node(between_func_node, keep_set=set(pre_nodes))
                        for pre_node in pre_nodes:
                            # 添加节点
                            dequantize_func = f.Dequantize(scale=scale, zero_point=zero_point, dtype=dtype)
                            tensor = dequantize_func(pre_node.true_obj)
                            dequantize_node = graph.add_node(dequantize_func)
                            tensor_node = graph.add_node(weakref.ref(tensor))
                            # 添加边
                            graph.add_edge(pre_node, dequantize_node)
                            graph.add_edge(dequantize_node, tensor_node)
                            graph.add_edge(tensor_node, between_func_node)

                        suc_nodes = graph.get_successors(between_func_node)
                        for suc_node in suc_nodes:
                            suc_func_nodes = graph.get_successors(suc_node)
                            # 去除原来的边
                            graph._remove_edges_from_node(suc_node, keep_set=set(suc_func_nodes))
                            for suc_func in suc_func_nodes:
                                # 添加节点
                                quantize_func = f.Quantize(scale=scale, zero_point=zero_point, dtype=dtype)
                                tensor = quantize_func(suc_node.true_obj)
                                quantize_node = graph.add_node(quantize_func)
                                tensor_node = graph.add_node(weakref.ref(tensor))
                                # 添加边
                                graph.add_edge(suc_node, quantize_node)
                                graph.add_edge(quantize_node, tensor_node)
                                graph.add_edge(tensor_node, suc_func)
        executor.__init__(graph)
        return executor

    @staticmethod
    def get_sub_quantize_nodes(graph: Graph, start_fq_node: Node):
        between_func_nodes = set() # FunctionNodes
        end_output_nodes = set() # TensorNodes: output
        end_fdq_nodes = set() # FuctionNodes: FDQ

        nodes = graph.get_successors(start_fq_node) # fq_node -> node(Tensor)
        fq_depth = 1
        node_stack = [(node, fq_depth) for node in nodes]
        # DFS
        while len(node_stack) > 0:
            cur_node, fq_depth = node_stack.pop() # TensorNode

            # cur_node -> suc_node(Function)
            suc_nodes = graph.get_successors(cur_node)
            # 无出边：是output，且没有遇到FDQ
            if len(suc_nodes) == 0:
                end_output_nodes.add(cur_node)

            # DFS搜索
            for suc_node in suc_nodes:
                # FDQ
                if isinstance(suc_node.true_obj, f.FakeDequantize):
                    if fq_depth == 1:
                        end_fdq_nodes.add(suc_node)
                        continue
                    else:
                        fq_depth -= 1
                else:
                    between_func_nodes.add(suc_node)

                # FQ
                if isinstance(suc_node.true_obj, f.FakeQuantize):
                    fq_depth += 1
                # cur_node -> suc_node -> suc_tensor_node(Tensor)
                suc_tensor_nodes = graph.get_successors(suc_node) 
                node_stack.extend([(suc_tensor_node, fq_depth) for suc_tensor_node in suc_tensor_nodes])

        return between_func_nodes, end_output_nodes, end_fdq_nodes
