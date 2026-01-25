# file name: train_lenet_local_mnist_fixed.py
import numpy as np
from pathlib import Path
import pickle
import gzip
from eneuro.nn.module import Module
from eneuro.utils import Visualizer

# 从 train_lenet_local_mnist.py 复制并修复
def load_mnist_from_pkl(pkl_path):
    """从mnist.pkl文件加载数据"""
    print(f"从 {pkl_path} 加载MNIST数据...")
    
    try:
        # 尝试直接读取
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except:
        # 如果文件是gzip压缩的
        with gzip.open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    
    # 不同格式的pkl可能有不同的数据结构
    if isinstance(data, tuple):
        if len(data) == 3:
            # 格式1: (train_data, valid_data, test_data)
            train_data, valid_data, test_data = data
            X_train, y_train = train_data
            X_test, y_test = test_data
        elif len(data) == 2:
            # 格式2: ((X_train, y_train), (X_test, y_test))
            (X_train, y_train), (X_test, y_test) = data
        else:
            raise ValueError(f"未知的数据格式，元组长度为: {len(data)}")
    elif isinstance(data, dict):
        # 格式3: 字典格式
        X_train = data.get('train_img', data.get('x_train'))
        y_train = data.get('train_label', data.get('y_train'))
        X_test = data.get('test_img', data.get('x_test'))
        y_test = data.get('test_label', data.get('y_test'))
    else:
        raise ValueError(f"未知的数据类型: {type(data)}")
    
    # 转换为numpy数组
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    print(f"加载完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    return X_train, y_train, X_test, y_test

def prepare_mnist_data(X_train, y_train, X_test, y_test):
    """准备MNIST数据 - 修复版本"""
    
    print(f"原始数据形状:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # 1. 归一化到 [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # 2. 检查数据是否被展平了
    if X_train.ndim == 2:
        # 数据已被展平，需要重塑为图像格式
        # 假设是MNIST，每个样本784维 = 28*28
        num_samples_train = X_train.shape[0]
        num_samples_test = X_test.shape[0]
        
        if X_train.shape[1] == 784:  # 展平的MNIST
            # 重塑为 (num_samples, 28, 28)
            X_train = X_train.reshape(num_samples_train, 28, 28)
            X_test = X_test.reshape(num_samples_test, 28, 28)
            print(f"数据已重塑为图像格式: {X_train.shape}")
        else:
            # 尝试自动推断图像尺寸
            # 假设是正方形图像
            img_size = int(np.sqrt(X_train.shape[1]))
            if img_size * img_size == X_train.shape[1]:
                X_train = X_train.reshape(num_samples_train, img_size, img_size)
                X_test = X_test.reshape(num_samples_test, img_size, img_size)
                print(f"数据已重塑为 {img_size}x{img_size} 图像")
            else:
                # 无法重塑，可能不是图像数据
                print("警告: 数据无法重塑为正方形图像，保持展平格式")
    
    # 3. 添加通道维度 (batch, channel, height, width)
    if X_train.ndim == 3:
        X_train = X_train[:, np.newaxis, :, :]  # 添加通道维度，变为 (batch, 1, 28, 28)
        X_test = X_test[:, np.newaxis, :, :]
    
    print(f"\n数据预处理完成:")
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

class MNISTDataset:
    """简单的MNIST数据集类"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.images)

class SimpleDataLoader:
    """简单的数据加载器"""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        # 创建索引列表
        indices = np.arange(len(self.dataset))
        
        # 如果需要打乱
        if self.shuffle:
            np.random.shuffle(indices)
        
        # 按批次生成数据
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # 收集批次数据
            batch_x = []
            batch_y = []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            # 转换为numpy数组
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            
            yield batch_x, batch_y
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# 从 lenet_custom.py 复制 LeNet 定义
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
        self.conv1 = Conv2d(6, kernel_size=5, stride=1, visualize=True)  # 输出通道: 6
        self.conv2 = Conv2d(16, kernel_size=5, stride=1) # 输出通道: 16

        # 计算全连接层输入维度
        # 经过两次池化后，特征图尺寸为 input_size // 4 - 3
        feature_size = (input_size // 4) - 3  # 对于28: (28//4)-3 = 7-3 = 4
        fc_input_dim = 16 * feature_size * feature_size
        
        # 全连接层
        self.fc1 = Linear(120)                   # 输入: fc_input_dim, 输出: 120
        self.fc2 = Linear(84)                    # 输入: 120, 输出: 84
        self.fc3 = Linear(num_classes)           # 输入: 84, 输出: num_classes
        
        # 保存参数用于前向传播
        self.feature_size = feature_size
        
        # 注意：这里我们需要手动收集参数
        self._params = []
        self._collect_params()
    
    def _collect_params(self):
        """收集所有参数"""
        self._params = []
        for attr_name in ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']:
            layer = getattr(self, attr_name)
            if hasattr(layer, 'params'):
                for param in layer.params():
                    self._params.append(param)
    
    def forward(self, x):
        # 第一卷积块: Conv -> Sigmoid -> MaxPool
        x = self.conv1(x)
        x = self.F.sigmoid(x)
        x = self.F.pooling(x, 2, 2, visualize=True)  # kernel_size=2, stride=2
        
        # 第二卷积块: Conv -> Sigmoid -> MaxPool
        x = self.conv2(x)
        x = self.F.sigmoid(x)
        x = self.F.pooling(x, 2, 2)  # kernel_size=2, stride=2
        
        # 展平特征图
        x = self.F.flatten(x)
        
        # 全连接层
        x = self.F.sigmoid(self.fc1(x))
        x = self.F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        """返回参数列表"""
        return self._params
    
    def cleargrads(self):
        """清除所有梯度"""
        for param in self.params():
            param.cleargrad()
    
    def using_config(self, param, value):
        """模拟Config上下文管理器"""
        # 简化版本，实际使用时需要完整实现
        from eneuro.base import Config
        return Config.using_config(param, value)

def train_lenet_local_fixed():
    """使用本地MNIST文件训练LeNet - 修复版本"""
    print("开始使用本地MNIST文件训练LeNet...")
    
    # 1. 加载数据
    data_dir = "D:\\eneuro\\Enneuro-Project\\code\\testdata\\MNIST_data"
    pkl_path = Path(data_dir) / "mnist.pkl"
    
    if pkl_path.exists():
        X_train, y_train, X_test, y_test = load_mnist_from_pkl(pkl_path)
    else:
        print("未找到mnist.pkl，尝试加载原始IDX格式...")
        # 这里可以添加IDX格式的加载代码
        return
    
    # 2. 预处理数据
    X_train, y_train, X_test, y_test = prepare_mnist_data(
        X_train, y_train, X_test, y_test
    )
    
    # 3. 创建数据集
    train_dataset = MNISTDataset(X_train, y_train)
    test_dataset = MNISTDataset(X_test, y_test)
    
    print(f"\n数据集统计:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"类别数: {len(np.unique(y_train))}")
    print(f"图像形状: {X_train.shape[1:]}")
    
    # 4. 创建数据加载器
    batch_size = 64
    train_loader = SimpleDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = SimpleDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. 创建LeNet模型
    input_channels = X_train.shape[1]  # 1 (灰度图)
    input_size = X_train.shape[2]      # 28
    
    model = LeNet(
        num_classes=10, 
        input_channels=input_channels, 
        input_size=input_size
    )
    
    print(f"\nLeNet模型信息:")
    print(f"输入尺寸: {input_channels}x{input_size}x{input_size}")
    print(f"输出类别: 10")
    
    # 6. 统计参数数量
    params = list(model.params())
    total_params = 0
    for param in params:
        if hasattr(param, 'data') and param.data is not None:
            total_params += np.prod(param.data.shape)
    
    print(f"模型总参数量: {total_params:,}")
    
    # 7. 导入训练器
    from eneuro.nn.optim import Adam
    from eneuro.nn.loss import crossEntropyError
    from eneuro.train import Trainer, Evaluator
    
    # 8. 设置优化器和损失函数
    optimizer = Adam(model.params(), lr=0.001)
    loss_fn = crossEntropyError

    visualizer = Visualizer(num_classes=10)
    
    # 9. 训练模型
    trainer = Trainer(model, loss_fn, optimizer, visualizer)
    
    print("\n开始训练...")
    trainer.fit(
        train_loader, 
        test_loader, 
        epochs=1,
        batch_size=batch_size,
        verbose=True
    )
    
    # 10. 最终评估
    # from eneuro.global_config import VISUAL_CONFIG
    # VISUAL_CONFIG["ENABLE_ALL_LAYERS"] = True  # 修改字典的值，所有模块都能读到
    from eneuro.global_config import visualize_model_first_batch
    visualize_model_first_batch(model, test_loader)

    print("\n最终评估...")
    evaluator = Evaluator(model, loss_fn, visualizer)
    test_loss, test_acc = evaluator.evaluate(
        test_loader, 
        batch_size=batch_size, 
        verbose=True
    )
    
    print(f"\n训练完成!")
    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"测试损失: {test_loss:.4f}")

    visualizer.plot_all()
    
    return model, test_acc

if __name__ == "__main__":
    t = train_lenet_local_fixed()
    if (t is not None):
        model, accuracy = t
    else:
        print("训练未完成。")