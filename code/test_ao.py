import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eneuro.base import Tensor
from eneuro.nn.module import CNNWithPooling
from eneuro.nn.loss import CrossEntropyLoss
from eneuro.nn.optim import SGD
from eneuro.base.functions import *
from eneuro.ao.tracer import trace_context

# 创建简单的测试数据
X = np.random.randn(2, 3, 32, 32).astype(np.float32)  # 2张32x32的3通道图像
y = np.array([0, 5], dtype=np.int32)                   # 2个样本的标签，范围[0, 9]

# 创建模型
model = CNNWithPooling()

# 开始记录
with trace_context() as tracer:
    output = model(Tensor(X))
    graph = tracer.get_graph()

# 此时 graph 包含完整的前向计算图，可用于后续的模式匹配和融合
graph.visualize("my_graph.dot")   # 导出为 dot 文件

from eneuro.ao.pattern import FusionPattern, NodeMatcher, MatchStrategy
from eneuro.base.functions import Conv2d, ReLU, BatchNorm2d

conv_relu_pattern = FusionPattern(
    name="conv_relu",
    node_matchers=[
        NodeMatcher(func_type=Conv2d),
        NodeMatcher(func_type=ReLU),
    ],
    fused_class=FusedConvReLU   # 需要从 functions.py 导入
)

matches = []
for node in graph.func_nodes.values():
    match = conv_relu_pattern.match_start(graph, node)
    if match:
        matches.append(match)

print(f"Found {len(matches)} conv+relu patterns")

for i, match in enumerate(matches):
    print(f"Match {i}:")
    print(f"  Input tensor ids: {[n.id for n in match.input_tensors]}")
    print(f"  Output tensor ids: {[n.id for n in match.output_tensors]}")
    print(f"  Matched function ids: {[n.id for n in match.matched_nodes]}")

from eneuro.base.functions import FusedConvReLU
for match in matches:
    conv_node = match.matched_nodes[0]
    conv_func = conv_node.obj
    fused_func = FusedConvReLU(stride=conv_func.stride, pad=conv_func.pad)
    match.replace(graph, fused_func)

# 检查新节点是否出现在图中
fused_nodes = [n for n in graph.func_nodes.values() 
               if n.obj.__class__.__name__ == 'FusedConvReLU']
assert len(fused_nodes) == 2

# 检查是否存在孤立的旧节点（应已被删除）
old_conv_nodes = [n for n in graph.func_nodes.values() 
                  if isinstance(n.obj, Conv2d)]
old_relu_nodes = [n for n in graph.func_nodes.values() 
                  if isinstance(n.obj, ReLU)]
#assert len(old_conv_nodes) == 0 and len(old_relu_nodes) == 0

graph.visualize("optimized_graph.dot")

# 创建损失函数和优化器
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.params(), lr=0.1)

# 测试前向传播
print("测试前向传播...")
y_hat = model(Tensor(X))
print(f"前向传播结果: {y_hat}")

# 测试损失计算
print("\n测试损失计算...")
loss = loss_fn(y_hat, Tensor(y))
print(f"损失值: {loss}")

# 测试反向传播
print("\n测试反向传播...")
loss.backward()
print("反向传播完成")

# 测试参数更新
print("\n测试参数更新...")
optimizer.step()
print("参数更新完成")

print("\n所有测试完成！")
