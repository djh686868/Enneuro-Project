from eneuro.nn.module import MLP, Sequential, Linear, Module
from eneuro.nn.optim import SGD, Adam
from eneuro.utils.serializer import Serializer
from eneuro.base import Tensor
import numpy as np
from pathlib import Path
import pickle
import gzip

# ------------------------------
# 测试1：简单Sequential模型
# ------------------------------
print("="*50)
print("Test 1: Simple Sequential Model")
print("="*50)

# 1. 创建一个简单的MLP模型
print("Creating model...")
model = Sequential(
    Linear(10),
    Linear(5),
    Linear(2)
)

# 2. 初始化优化器
params = model.get_params_list()
optimizer = SGD(params, lr=0.01)

# 3. 使用随机数据进行一次前向传播（为了生成一些参数值）
print("Testing model with random data...")
input_data = Tensor(np.random.randn(2, 10).astype(np.float32))
output = model(input_data)
print(f"Model output: {output}")

# 4. 保存模型
model_path = "test_model.json"
checkpoint_path = "test_checkpoint.json"
print(f"Saving model to {model_path}...")
Serializer.save(model, model_path)

# 5. 保存完整检查点
print(f"Saving checkpoint to {checkpoint_path}...")
Serializer.save_checkpoint(model, optimizer, epoch=5, path=checkpoint_path)

# 6. 创建新模型用于载入
print("Creating new model for loading...")
loaded_model = Sequential(
    Linear(10),
    Linear(5),
    Linear(2)
)
loaded_optimizer = SGD(loaded_model.get_params_list(), lr=0.01)

# 7. 载入模型
print(f"Loading model from {model_path}...")
Serializer.load(loaded_model, model_path)

# 8. 测试载入的模型
print("Testing loaded model...")
loaded_output = loaded_model(input_data)
print(f"Loaded model output: {loaded_output}")

# 9. 比较原始模型和载入模型的输出是否一致
print(f"Outputs match: {np.allclose(output.data, loaded_output.data, atol=1e-6)}")

# 10. 载入完整检查点
print(f"Loading checkpoint from {checkpoint_path}...")
loaded_epoch = Serializer.load_checkpoint(checkpoint_path, loaded_model, loaded_optimizer)
print(f"Loaded epoch: {loaded_epoch}")

# 11. 测试载入检查点后的模型
print("Testing model from checkpoint...")
checkpoint_output = loaded_model(input_data)
print(f"Checkpoint model output: {checkpoint_output}")

# 12. 比较所有输出是否一致
print(f"All outputs match: {np.allclose(output.data, checkpoint_output.data, atol=1e-6)}")

print("\nTest 1 completed successfully!")

# ------------------------------
# 测试2：LeNet模型
# ------------------------------
print("\n" + "="*50)
print("Test 2: LeNet Model with MNIST Data")
print("="*50)

# LeNet模型定义
class LeNet(Module):
    """LeNet模型"""
    
    def __init__(self, num_classes=10, input_channels=1, input_size=28):
        super().__init__()
        
        # 导入必要的模块
        from eneuro.nn.module import Conv2d, Linear
        from eneuro.base import functions as F
        
        # 保存函数引用
        self.F = F

        # 卷积层参数
        self.conv1 = Conv2d(6, kernel_size=5, stride=1)
        self.conv2 = Conv2d(16, kernel_size=5, stride=1)

        # 计算全连接层输入维度
        feature_size = (input_size // 4) - 3
        
        # 全连接层
        self.fc1 = Linear(120)
        self.fc2 = Linear(84)
        self.fc3 = Linear(num_classes)
        
        # 保存参数用于前向传播
        self.feature_size = feature_size
    
    def forward(self, x):
        # 第一卷积块: Conv -> Sigmoid -> MaxPool
        x = self.conv1(x)
        x = self.F.sigmoid(x)
        x = self.F.pooling(x, 2, 2)
        
        # 第二卷积块: Conv -> Sigmoid -> MaxPool
        x = self.conv2(x)
        x = self.F.sigmoid(x)
        x = self.F.pooling(x, 2, 2)
        
        # 展平特征图
        x = self.F.flatten(x)
        
        # 全连接层
        x = self.F.sigmoid(self.fc1(x))
        x = self.F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        
        return x

# 数据加载函数
def load_mnist_from_pkl(pkl_path):
    """从mnist.pkl文件加载数据"""
    print(f"Loading MNIST data from {pkl_path}...")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except:
        with gzip.open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    
    if isinstance(data, tuple):
        if len(data) == 3:
            train_data, valid_data, test_data = data
            X_train, y_train = train_data
            X_test, y_test = test_data
        elif len(data) == 2:
            (X_train, y_train), (X_test, y_test) = data
        else:
            raise ValueError(f"Unknown data format, tuple length: {len(data)}")
    elif isinstance(data, dict):
        X_train = data.get('train_img', data.get('x_train'))
        y_train = data.get('train_label', data.get('y_train'))
        X_test = data.get('test_img', data.get('x_test'))
        y_test = data.get('test_label', data.get('y_test'))
    else:
        raise ValueError(f"Unknown data type: {type(data)}")
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    print(f"Loaded: train {X_train.shape}, test {X_test.shape}")
    return X_train, y_train, X_test, y_test

def prepare_mnist_data(X_train, y_train, X_test, y_test):
    """准备MNIST数据"""
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    if X_train.ndim == 2:
        num_samples_train = X_train.shape[0]
        num_samples_test = X_test.shape[0]
        X_train = X_train.reshape(num_samples_train, 28, 28)
        X_test = X_test.reshape(num_samples_test, 28, 28)
    
    if X_train.ndim == 3:
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]
    
    print(f"Processed data shape:")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# 1. 加载MNIST数据
data_dir = "testdata/MNIST_data"
pkl_path = Path(data_dir) / "mnist.pkl"

if pkl_path.exists():
    X_train, y_train, X_test, y_test = load_mnist_from_pkl(pkl_path)
    X_train, y_train, X_test, y_test = prepare_mnist_data(X_train, y_train, X_test, y_test)
else:
    print(f"MNIST data not found at {pkl_path}, skipping LeNet test")
    exit(0)

# 2. 创建LeNet模型
print("Creating LeNet model...")
lenet_model = LeNet(
    num_classes=10, 
    input_channels=X_train.shape[1], 
    input_size=X_train.shape[2]
)

# 3. 初始化优化器
lenet_params = lenet_model.params()
lenet_optimizer = Adam(lenet_params, lr=0.001)
print(f"Created Adam optimizer with lr={lenet_optimizer._state.get('lr')}")

# 4. 统计参数数量
total_params = 0
for param in lenet_params:
    if hasattr(param, 'data') and param.data is not None:
        total_params += np.prod(param.data.shape)
print(f"LeNet total parameters: {total_params:,}")

# 5. 测试LeNet模型
print("Testing LeNet model...")
# 使用少量MNIST数据进行测试
test_batch = X_train[:2]
test_input = Tensor(test_batch)
test_output = lenet_model(test_input)
print(f"LeNet output: {test_output}")

# 6. 保存LeNet模型
lenet_model_path = "test_lenet_model.json"
lenet_checkpoint_path = "test_lenet_checkpoint.json"
lenet_optimizer_path = "test_lenet_optimizer.json"
print(f"Saving LeNet model to {lenet_model_path}...")
Serializer.save(lenet_model, lenet_model_path)

# 7. 单独保存优化器
print(f"Saving optimizer to {lenet_optimizer_path}...")
Serializer.save(lenet_optimizer, lenet_optimizer_path)

# 8. 保存LeNet完整检查点
print(f"Saving LeNet checkpoint to {lenet_checkpoint_path}...")
Serializer.save_checkpoint(lenet_model, lenet_optimizer, epoch=3, path=lenet_checkpoint_path)

# 9. 创建新的LeNet模型和优化器用于载入
print("Creating new LeNet model and optimizer for loading...")
loaded_lenet_model = LeNet(
    num_classes=10, 
    input_channels=X_train.shape[1], 
    input_size=X_train.shape[2]
)
loaded_lenet_optimizer = Adam(loaded_lenet_model.params(), lr=0.01)  # 使用不同的初始学习率
print(f"Created new optimizer with initial lr={loaded_lenet_optimizer._state.get('lr')}")

# 10. 载入LeNet模型
print(f"Loading LeNet model from {lenet_model_path}...")
Serializer.load(loaded_lenet_model, lenet_model_path)

# 11. 测试载入的LeNet模型
print("Testing loaded LeNet model...")
loaded_lenet_output = loaded_lenet_model(test_input)
print(f"Loaded LeNet output: {loaded_lenet_output}")

# 12. 比较原始LeNet模型和载入模型的输出是否一致
print(f"LeNet outputs match: {np.allclose(test_output.data, loaded_lenet_output.data, atol=1e-6)}")

# 13. 单独载入优化器
print(f"Loading optimizer from {lenet_optimizer_path}...")
Serializer.load(loaded_lenet_optimizer, lenet_optimizer_path)
print(f"Loaded optimizer lr: {loaded_lenet_optimizer._state.get('lr')}")

# 14. 载入LeNet完整检查点
print(f"Loading LeNet checkpoint from {lenet_checkpoint_path}...")
loaded_lenet_epoch = Serializer.load_checkpoint(lenet_checkpoint_path, loaded_lenet_model, loaded_lenet_optimizer)
print(f"Loaded LeNet epoch: {loaded_lenet_epoch}")
print(f"Optimizer lr after checkpoint load: {loaded_lenet_optimizer._state.get('lr')}")

# 15. 测试载入检查点后的LeNet模型
print("Testing LeNet model from checkpoint...")
checkpoint_lenet_output = loaded_lenet_model(test_input)
print(f"Checkpoint LeNet output: {checkpoint_lenet_output}")

# 16. 比较所有LeNet输出是否一致
print(f"All LeNet outputs match: {np.allclose(test_output.data, checkpoint_lenet_output.data, atol=1e-6)}")

# 17. 验证优化器状态保存和恢复
print("\n=== Optimizer State Verification ===")
print(f"Original optimizer lr: 0.001")
print(f"Loaded optimizer lr from file: {loaded_lenet_optimizer._state.get('lr')}")
print(f"Optimizer lr matches: {abs(loaded_lenet_optimizer._state.get('lr') - 0.001) < 1e-6}")
print(f"Checkpoint epoch correctly loaded: {loaded_lenet_epoch == 3}")

print("\nTest 2 completed successfully!")
print("\n" + "="*50)
print("All tests completed successfully!")
print("="*50)
