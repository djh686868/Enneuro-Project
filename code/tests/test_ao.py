import sys
import os
import numpy as np
import time
from pathlib import Path

# 添加code目录到Python搜索路径，这样就能找到eneuro模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import Tensor,Config
from eneuro.base import functions as F
from eneuro.nn.module import CNNWithPooling,Sequential,Conv2d,BatchNorm2d,Linear
from eneuro.nn.loss import CrossEntropyLoss
from eneuro.nn.optim import SGD
from eneuro.ao import GraphOptimizer,GraphExecutor,trace_context
from eneuro.utils import save_checkpoint,load_checkpoint

# 创建简单的测试数据
size = 32
X = np.random.randn(2, 3, size, size).astype(np.float32)  # 2张3通道图像
y = np.array([0, 5], dtype=np.int32)                   # 2个样本的标签，范围[0, 9]

num_fuse = 10
sequential_content = []
for i in range(num_fuse):
    sequential_content.append(Conv2d(3,3))
    sequential_content.append(BatchNorm2d(3))
    sequential_content.append(F.relu)
sequential_content.append(F.flatten)
sequential_content.append(Linear(10))

def test_normal(epoch_num = 10):
    # 创建模型
    model = Sequential(*sequential_content)

    # 创建损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params(), lr=0.1)

    tic = time.time()
    for epoch in range(epoch_num):
        #print(f"epoch {epoch}")
        y_hat = model(Tensor(X))

        loss = loss_fn(y_hat, Tensor(y))
        loss.backward()

        optimizer.step()
    toc = time.time()
    duration = toc - tic

    save_checkpoint(model, optimizer, num_epoch, "normal_checkpoint.json")
    
    print(f"normal training complete in {duration:.4f}s  loss = {loss}")
    return duration

def test_executor(epoch_num = 10):
    # 创建模型
    model = Sequential(*sequential_content)

    # 执行一次前向，记录计算图
    sample_input = Tensor(X)
    with trace_context() as tracer:
        _ = model(sample_input)
        graph = tracer.get_graph()
    graph.visualize('origin_graph.dot')
    executor = GraphExecutor(graph)

    # 创建损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params(), lr=0.1)
    
    tic = time.time()
    for epoch in range(epoch_num):
        #print(f"epoch {epoch}")
        y_hat = executor.forward(Tensor(X))

        loss = loss_fn(y_hat, Tensor(y))
        loss.backward()

        optimizer.step()
    toc = time.time()
    duration = toc - tic
    print(f"executor training complete in {duration:.4f}s  loss = {loss}")
    return duration

def test_ao(epoch_num = 10):

    from eneuro.ao import GraphOptimizer
    # 创建模型
    model = Sequential(*sequential_content)

    # 执行一次前向，记录并优化计算图（得到优化后的 executor）
    sample_input = Tensor(X) # 样例输入
    op = GraphOptimizer(model, sample_input) # 图优化器
    graph = op.optimize() # 优化后的图
    graph.visualize('optimized_graph.dot') # 保存为.dot文件便于查看
    executor = op.optimize_to_executor() # 优化后的执行器

    # 创建损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params(), lr=0.1)
    
    tic = time.time()
    for epoch in range(epoch_num):
        y_hat = executor.forward(Tensor(X)) # 使用执行器进行前向传播

        loss = loss_fn(y_hat, Tensor(y))
        loss.backward() # 正常反向传播

        optimizer.step()
    toc = time.time()
    duration = toc - tic

    save_checkpoint(model, optimizer, num_epoch, "ao_checkpoint.json") # 正常保存
    print(f"ao training complete in {duration:.4f}s  loss = {loss}")
    
    return duration

def test_load_normal(path, epoch_num = 10):
    # 创建模型
    model = Sequential(*sequential_content)

    # 创建损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params(), lr=0.1)

    load_checkpoint(path, model, optimizer)

    save_checkpoint(model, optimizer, num_epoch, "normal_checkpoint_loaded.json")

    y_hat = model(Tensor(X))
    loss = loss_fn(y_hat, Tensor(y))
    print(f"immeadiate y_hat = {y_hat}  loss = {loss}")

    tic = time.time()
    for epoch in range(epoch_num):
        #print(f"epoch {epoch}")
        y_hat = model(Tensor(X))

        loss = loss_fn(y_hat, Tensor(y))
        loss.backward()

        optimizer.step()
    toc = time.time()
    duration = toc - tic
    
    save_checkpoint(model, optimizer, num_epoch, "normal_checkpoint_loaded_train.json")

    print(f"load training complete in {duration:.4f}s  loss = {loss}")
    return duration

def test_load_ao(path,epoch_num = 10):
    # 创建模型
    model = Sequential(*sequential_content)

    # 创建损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params(), lr=0.1)

    load_checkpoint(path, model, optimizer)

    # 执行一次前向，记录并优化计算图（得到优化后的 executor）
    sample_input = Tensor(X)
    op = GraphOptimizer(model, sample_input)
    #graph = op.optimize()
    #graph.visualize()
    executor = op.optimize_to_executor()
    
    save_checkpoint(model, optimizer, num_epoch, "ao_checkpoint_loaded.json")
    
    y_hat = executor.forward(Tensor(X))
    loss = loss_fn(y_hat, Tensor(y))
    print(f"immeadiate y_hat = {y_hat}  loss = {loss}")

    tic = time.time()
    for epoch in range(epoch_num):
        #print(f"epoch {epoch}")
        y_hat = executor.forward(Tensor(X))

        loss = loss_fn(y_hat, Tensor(y))
        loss.backward()

        optimizer.step()
    toc = time.time()
    duration = toc - tic

    save_checkpoint(model, optimizer, num_epoch, "ao_checkpoint_loaded_train.json")

    print(f"load ao training complete in {duration:.4f}s  loss = {loss}")
    return duration

def pattern_registry():
    from eneuro.ao import FusionRegistry,FusionPattern,NodeMatcher

    # 自定义匹配模式
    fusion_pattern = FusionPattern(
        name="conv_relu",
        node_matchers=[
            NodeMatcher(func_type=F.Conv2d),
            NodeMatcher(func_type=F.ReLU),
        ]
    )

    # 注册模式与替换的融合算子
    FusionRegistry.register(fusion_pattern, F.FusedConvReLU)

if __name__ == "__main__":
    num_epoch = 100
    normal_time = test_normal(num_epoch)
    executor_time = test_executor(num_epoch)
    ao_time = test_ao(num_epoch)

    #'''
    sub = normal_time - ao_time
    print(f"normal training complete in {normal_time:.4f}s")
    print(f"executor training complete in {executor_time:.4f}s")
    print(f"ao training complete in {ao_time:.4f}s")
    print(f"融合算子节约了 {sub * 100 / normal_time:.2f}% 的时间")
    print(f"本次测试中，平均每次forward每个conv_batchnorm_relu融合节约{sub / num_epoch / num_fuse}s")
    #'''
    
    test_load_normal("ao_checkpoint.json",num_epoch)
    test_load_ao("ao_checkpoint.json",num_epoch)

    print("\n所有测试完成！")
