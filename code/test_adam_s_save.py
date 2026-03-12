from eneuro.nn.module import Sequential, Linear, Module
from eneuro.nn.optim import Adam
from eneuro.utils.serializer import Serializer
from eneuro.base import Tensor
import numpy as np

# 创建一个简单的测试模型
class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(3)
        self.fc2 = Linear(1)
    
    def forward(self, x):
        from eneuro.base import functions as F
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 1. 创建模型和Adam优化器
print("="*50)
print("Testing Adam Optimizer Save Functionality")
print("="*50)

print("Creating model and Adam optimizer...")
model = SimpleModel()
params = model.get_params_list()
optimizer = Adam(params, lr=0.001, beta1=0.9, beta2=0.999)

# 2. 创建测试数据并进行前向传播
input_data = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
target = Tensor(np.array([[5.0]], dtype=np.float32))

print("Performing forward pass...")
output = model(input_data)
print(f"Model output: {output}")

# 3. 执行优化器步骤，生成状态变量
print("\nExecuting optimizer step to generate state variables...")

# 手动设置梯度（模拟反向传播结果）
for param in params:
    if param.data is not None:
        param.grad = Tensor(np.ones_like(param.data) * 0.1)

# 执行优化器步骤，生成v和s状态
optimizer.step()

# 4. 查看优化器状态
print("\nOptimizer state after step:")
print(f"Keys in optimizer state: {list(optimizer._state.keys())}")

if 'v' in optimizer._state:
    print(f"v (first moment) exists with {len(optimizer._state['v'])} entries")
    for idx, v_val in optimizer._state['v'].items():
        print(f"  Param {idx} v: {v_val[0]}")

if 's' in optimizer._state:
    print(f"s (second moment) exists with {len(optimizer._state['s'])} entries")
    for idx, s_val in optimizer._state['s'].items():
        print(f"  Param {idx} s: {s_val[0]}")
else:
    print("s (second moment) does not exist in optimizer state")

# 5. 保存优化器状态
optimizer_path = "test_adam_optimizer.json"
print(f"\nSaving Adam optimizer state to {optimizer_path}...")
Serializer.save(optimizer, optimizer_path)

# 6. 读取保存的JSON文件内容，查看是否包含s
print(f"\nReading saved optimizer state from {optimizer_path}...")
import json
with open(optimizer_path, 'r', encoding='utf-8') as f:
    saved_state = json.load(f)

print(f"Saved state keys: {list(saved_state.keys())}")
if 's' in saved_state:
    print(f"s (second moment) is saved in the JSON file with {len(saved_state['s'])} entries")
    for idx, s_val in saved_state['s'].items():
        print(f"  Param {idx} saved s: {s_val[0]}")
else:
    print("s (second moment) is NOT saved in the JSON file")

# 7. 对比MomentumSGD和Adam的状态差异
print("\n" + "="*50)
print("Comparison: MomentumSGD vs Adam")
print("="*50)
print("MomentumSGD optimizer:")
print("  - Uses only 'v' (first moment/momentum)")
print("  - No 's' (second moment) parameter")
print("  - Suitable for general optimization tasks")
print("\nAdam optimizer:")
print("  - Uses both 'v' (first moment) and 's' (second moment)")
print("  - 'v' tracks exponential moving average of gradients")
print("  - 's' tracks exponential moving average of squared gradients")
print("  - Provides adaptive learning rates")
print("  - Better convergence for complex models")

print("\n" + "="*50)
print("Conclusion")
print("="*50)
print("The 's' parameter is specific to Adam optimizer, not MomentumSGD.")
print("When you save a MomentumSGD optimizer, you won't see 's' in the saved state.")
print("When you save an Adam optimizer, both 'v' and 's' will be saved.")
