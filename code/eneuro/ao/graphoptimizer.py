from .tracer import trace_context
from .pattern import PatternMatcher, FusionRegistry
from .graph import Graph
from .executor import GraphExecutor

class GraphOptimizer:
    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input

    def optimize(self) -> Graph:
        # 1. 记录一次前向，得到计算图
        with trace_context() as tracer:
            _ = self.model(self.sample_input)
            graph = tracer.get_graph()

        # 2. 匹配所有可融合模式
        matcher = PatternMatcher(graph, FusionRegistry)
        matches = matcher.find_all_matches()

        # 3. 依次替换子图
        for match in matches:
            match.replace(graph)

        return graph

    def optimize_to_executor(self) -> GraphExecutor:
        optimized_graph = self.optimize()
        return GraphExecutor(optimized_graph)