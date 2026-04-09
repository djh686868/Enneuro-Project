import sys
from pathlib import Path
import numpy as np

# 添加code目录到Python搜索路径，这样就能找到eneuro模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.nn.module import Sequential, Linear, Module
from eneuro.nn.optim import MomentumSGD
from eneuro.utils.serializer import Serializer
from eneuro.base import Tensor

# 创建一个简单的测试模型
class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(5)
        self.fc2 = Linear(3)
        self.fc3 = Linear(1)
    
    def forward(self, x):
        from eneuro.base import functions as F
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 1. 创建模型和优化器
print("="*50)
print("Testing MomentumSGD Save/Load Functionality")
print("="*50)

print("Creating model and MomentumSGD optimizer...")
model = SimpleModel()
params = model.get_params_list()
optimizer = MomentumSGD(params, lr=0.01, momentum=0.9)

# 2. 初始化动量为特定值（用于验证）
print("Initializing optimizer state...")
# 手动设置一些动量值，以便后续验证
for idx, param in enumerate(params):
    if param.data is not None:
        # 为每个参数设置初始动量
        optimizer._state['v'][str(idx)] = np.ones_like(param.data) * 0.5
        print(f"  Param {idx} momentum initialized to: {optimizer._state['v'][str(idx)][0]}")

# 3. 创建测试数据
input_data = Tensor(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32))
target = Tensor(np.array([[10.0]], dtype=np.float32))

# 4. 模拟一次前向传播和反向传播，计算梯度
print("\nPerforming forward and backward pass...")
output = model(input_data)
print(f"  Model output: {output}")

# 计算损失（简单的MSE）
lost = (output - target) ** 2

# 5. 保存优化器状态
optimizer_path = "test_momentum_sgd_optimizer.json"
print(f"\nSaving optimizer state to {optimizer_path}...")
Serializer.save(optimizer, optimizer_path)

# 6. 创建新模型和优化器用于加载
print("\nCreating new model and optimizer for loading...")
loaded_model = SimpleModel()
loaded_params = loaded_model.get_params_list()
loaded_optimizer = MomentumSGD(loaded_params, lr=0.01, momentum=0.9)

# 7. 加载优化器状态
print(f"Loading optimizer state from {optimizer_path}...")
Serializer.load(loaded_optimizer, optimizer_path)

# 8. 验证加载的动量值
print("\nVerifying loaded momentum values...")
for idx, param in enumerate(loaded_params):
    if param.data is not None:
        loaded_momentum = loaded_optimizer._state['v'].get(str(idx), None)
        if loaded_momentum is not None:
            print(f"  Param {idx} loaded momentum: {loaded_momentum[0]}")
            print(f"  Momentum correctly loaded: {np.allclose(loaded_momentum, np.ones_like(loaded_momentum) * 0.5)}")

# 9. 测试使用加载的动量进行参数更新
print("\nTesting parameter update with loaded momentum...")

# 对加载的模型进行前向传播和反向传播
loaded_output = loaded_model(input_data)
loaded_loss = (loaded_output - target) ** 2

# 模拟梯度计算（这里直接设置梯度值以便验证）
for idx, param in enumerate(loaded_params):
    if param.data is not None:
        param.grad = Tensor(np.ones_like(param.data) * 0.1)
        print(f"  Param {idx} gradient set to: {param.grad.data[0]}")

# 保存更新前的参数值
pre_update_params = [param.data.copy() for param in loaded_params]

# 执行优化器步骤
print("  Executing optimizer step with loaded momentum...")
loaded_optimizer.step()

# 10. 验证参数更新是否正确使用了动量
print("\nVerifying parameter update with momentum...")
for idx, (param, pre_param) in enumerate(zip(loaded_params, pre_update_params)):
    if param.data is not None:
        # 计算预期的更新：w = w + v，其中 v = v*m - lr*grad
        expected_v = 0.5 * 0.9 - 0.01 * 0.1  # 初始动量*动量系数 - 学习率*梯度
        expected_param = pre_param + expected_v
        actual_param = param.data
        print(f"  Param {idx}:")
        print(f"    Pre-update: {pre_param[0]}")
        print(f"    Expected post-update: {expected_param[0]}")
        print(f"    Actual post-update: {actual_param[0]}")
        print(f"    Update matches expected: {np.allclose(actual_param, expected_param, atol=1e-6)}")

# 11. 保存和加载完整检查点测试
print("\n" + "="*50)
print("Testing Checkpoint Save/Load")
print("="*50)

checkpoint_path = "test_momentum_sgd_checkpoint.json"
print(f"Saving checkpoint to {checkpoint_path}...")
Serializer.save_checkpoint(model, optimizer, epoch=2, path=checkpoint_path)

# 创建新模型和优化器
new_model = SimpleModel()
new_optimizer = MomentumSGD(new_model.get_params_list(), lr=0.01, momentum=0.9)

# 加载检查点
print(f"Loading checkpoint from {checkpoint_path}...")
loaded_epoch = Serializer.load_checkpoint(checkpoint_path, new_model, new_optimizer)

print(f"Loaded epoch: {loaded_epoch}")
print(f"Checkpoint epoch correctly loaded: {loaded_epoch == 2}")

# 验证加载的动量
print("\nVerifying momentum from checkpoint...")
new_params = new_model.get_params_list()
for idx, param in enumerate(new_params):
    if param.data is not None:
        loaded_momentum = new_optimizer._state['v'].get(str(idx), None)
        if loaded_momentum is not None:
            print(f"  Param {idx} momentum from checkpoint: {loaded_momentum[0]}")
            print(f"  Momentum correctly loaded: {np.allclose(loaded_momentum, np.ones_like(loaded_momentum) * 0.5)}")

print("\n" + "="*50)
print("All tests completed!")
print("="*50)
