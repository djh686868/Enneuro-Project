from __future__ import annotations
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import numpy as np
import weakref

from .graph import Graph, Node, NodeType

from ..base.core import Tensor, Function, Config
from ..nn.optim import Optimizer
from ..base.functions import to_xp, get_array_module, compare_dtype_greater
from ..base import functions as f


class AutoCastManager:
    @staticmethod
    def apply_cast(graph: Graph, dtype: str='float16') -> Graph:
        # 按拓扑序遍历所有 Function 节点
        for node in graph.topological_order():
            if node.type != NodeType.FUNCTION: # 排除tensor
                continue

            func_cls = node.true_obj.__class__
            # 是Cast/UpCast/DownCast节点，直接标记后继节点的dtype
            if func_cls in (f.Cast, f.UpCast, f.DownCast):
                func_dtype = node.true_obj.dtype

                suc_nodes = graph.get_successors(node)
                for suc_node in suc_nodes:
                    if func_cls == f.Cast:
                        suc_node.dtype = func_dtype
                    elif func_cls == f.UpCast:
                        if compare_dtype_greater(func_dtype, suc_node.dtype):
                            suc_node.dtype = func_dtype
                    else:
                        if compare_dtype_greater(suc_node.dtype, func_dtype):
                            suc_node.dtype = func_dtype
                continue
                    
            # 其他Function节点
            else:
                pre_nodes = graph.get_predecessors(node)
                suc_nodes = graph.get_successors(node)

                # 可以低精度的Function
                if func_cls in f.CastRegistry.cancast:
                    for pre in pre_nodes:
                        # 若输入高精度，则添加DownCast
                        if compare_dtype_greater(pre.dtype, dtype):
                            '''
                            pre(Tensor) -> node
                            变为
                            pre(Tensor) -> DownCast -> Tensor -> node 
                            '''
                            # 去除原来的边
                            graph._remove_edges_to_node(node, keep_set=set([pre]))
                            # 添加节点
                            downcast_func = f.DownCast(dtype=dtype)
                            tensor = downcast_func(pre.true_obj)
                            downcast_node = graph.add_node(downcast_func)
                            tensor_node = graph.add_node(weakref.ref(tensor))
                            # 添加边
                            graph.add_edge(pre, downcast_node)
                            graph.add_edge(downcast_node, tensor_node)
                            graph.add_edge(tensor_node, node)
                        # 重新连接一次边，以保证输入的顺序
                        else:
                            graph._remove_edges_to_node(node, keep_set=set([pre]))
                            graph.add_edge(pre, node)

                    
                    # 标记后继节点
                    for suc in suc_nodes:
                        if compare_dtype_greater(suc.dtype, dtype):
                            suc.dtype = dtype 

                # 需要高精度的Function
                elif func_cls in f.CastRegistry.resist_cast:
                    UPPER_DTYPE = 'float32'
                    for pre in pre_nodes:
                        # 若输入是低精度，则添加UpCast
                        if compare_dtype_greater(UPPER_DTYPE, pre.dtype):
                            '''
                            pre(Tensor) -> node
                            变为
                            pre(Tensor) -> UpCast -> Tensor -> node 
                            '''
                            # 去除原来的边
                            graph._remove_edges_to_node(node, keep_set=set([pre]))
                            # 添加节点
                            upcast_func = f.UpCast(dtype=UPPER_DTYPE)
                            tensor = upcast_func(pre.true_obj)
                            upcast_node = graph.add_node(upcast_func)
                            tensor_node = graph.add_node(weakref.ref(tensor))
                            # 添加边
                            graph.add_edge(pre, upcast_node)
                            graph.add_edge(upcast_node, tensor_node)
                            graph.add_edge(tensor_node, node)
                        # 重新连接一次边，以保证输入的顺序
                        else:
                            graph._remove_edges_to_node(node, keep_set=set([pre]))
                            graph.add_edge(pre, node)
                    
                    # 标记后继节点
                    for suc in suc_nodes:
                        if compare_dtype_greater(UPPER_DTYPE, suc.dtype):
                            suc.dtype = dtype 
                        
                # 没有要求的Function
                else:
                    if len(pre_nodes) > 0:
                        xp = get_array_module(pre_nodes[0])
                        suc_dtype = xp.result_type(*[pre.dtype for pre in pre_nodes])

                        for suc in suc_nodes:
                            suc.dtype = suc_dtype

        return graph

class GradScaler:
    '''
    动态损失缩放器，使梯度落入FP16的可表示范围内。
    用例：
    scaler = GradScaler()  # 损失缩放器
    for input, target in data:
        output = model(input)
        loss = loss_fn(output, target)       # 前向：低精度

        scaler.scale(loss).backward()         # 反向：缩放后的低精度梯度计算
        scaler.step(optimizer)               # 反缩放梯度并更新参数
    '''
    def __init__(self, init_scale=32768.0, growth_factor=2.0, backoff_factor=0.5, 
                 growth_interval=2000, min_scale=1.0) -> None:
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale

        self.step_count = 0

    # 缩放增大梯度，防止下溢
    def scale(self, loss: Tensor) -> Tensor:
        return self.scale_factor * loss

    def step(self, optimizer: Optimizer) -> None:
        # 检查溢出
        has_overflow = False
        for param in optimizer.params:
            if param.grad is None:
                continue

            xp = get_array_module(param.grad.data)
            if (xp.isinf(param.grad.data) | xp.isnan(param.grad.data)).any():
                has_overflow = True
                break

        # 溢出则本次梯度无效
        if has_overflow:
            optimizer.zero_grad() # 清除梯度

            self.scale_factor *= self.backoff_factor # 缩小缩放倍数
            self.scale_factor = max(self.scale_factor, self.min_scale) # 不小于min_scale

            self.step_count = 0

        # 未溢出则梯度下降
        else:
            # 还原梯度大小，正常更新
            for param in optimizer.params:
                if param.grad is None:
                    continue
                xp = get_array_module(param.grad.data)
                param.grad.data /= xp.asarray(self.scale_factor)
            optimizer.step()

            # 连续未溢出则放大缩放倍数
            self.step_count += 1
            if self.step_count >= self.growth_interval:
                self.scale_factor *= self.growth_factor
                self.step_count = 0
        
