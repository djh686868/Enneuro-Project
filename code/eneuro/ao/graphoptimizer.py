from .tracer import trace_context
from .pattern import PatternMatcher, FusionRegistry
from .graph import Graph
from .executor import GraphExecutor
from .cast import AutoCastManager
from .quantize import QuantizeManager

class GraphOptimizer:
    @staticmethod
    def model_to_graph(model, sample_input) -> Graph:
        # 记录一次前向，得到计算图
        with trace_context() as tracer:
            _ = model(sample_input)
            graph = tracer.get_graph()
        return graph

    @staticmethod
    def graph_to_executor(graph) -> GraphExecutor:
        return GraphExecutor(graph)

    @staticmethod
    def graph_apply_fuse(graph) -> Graph:
        # 匹配所有可融合模式
        matcher = PatternMatcher(graph, FusionRegistry)
        matches = matcher.find_all_matches()

        # 依次替换子图
        for match in matches:
            match.replace(graph)

        return graph
    
    @staticmethod
    def graph_apply_cast(graph, dtype='float16') -> Graph:
        graph = AutoCastManager.apply_cast(graph=graph, dtype=dtype)
        return graph
    
    @staticmethod
    def executor_apply_quantize(executor: GraphExecutor) -> GraphExecutor:
        return QuantizeManager.apply_quantize(executor=executor)
    
    @staticmethod
    def auto_optimize(model, sample_input) -> GraphExecutor:
        graph = GraphOptimizer.model_to_graph(model=model, sample_input=sample_input)
        graph = GraphOptimizer.graph_apply_fuse(graph)
        graph = GraphOptimizer.graph_apply_cast(graph)
        return GraphOptimizer.graph_to_executor(graph)