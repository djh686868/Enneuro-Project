# file name: lenet_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import gzip
from sklearn.metrics import confusion_matrix
import seaborn as sns

class LeNetPyTorch(nn.Module):
    """PyTorch版本的LeNet，与我们的框架保持一致"""
    
    def __init__(self, num_classes=10):
        super(LeNetPyTorch, self).__init__()
        
        # 卷积层定义
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 输入1通道，输出6通道
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 输入6通道，输出16通道
        
        # 全连接层定义
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 对于28x28输入，经过两次池化后是4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 使用Sigmoid激活，与原始LeNet和我们的框架保持一致
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # 卷积块1: Conv -> Sigmoid -> MaxPool
        x = self.activation(self.conv1(x))
        x = F.max_pool2d(x, 2)  # kernel_size=2, stride=2
        
        # 卷积块2: Conv -> Sigmoid -> MaxPool
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2)  # kernel_size=2, stride=2
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x

def load_mnist_data_pytorch(data_dir):
    """为PyTorch加载MNIST数据"""
    # 尝试多种路径，确保无论从哪里运行都能找到数据
    possible_paths = [
        Path(data_dir) / "mnist.pkl",
        Path(".") / data_dir / "mnist.pkl",
        Path("..") / data_dir / "mnist.pkl",
        Path("code") / "testdata" / "MNIST_data" / "mnist.pkl",
        Path("..") / "code" / "testdata" / "MNIST_data" / "mnist.pkl"
    ]
    
    pkl_path = None
    for path in possible_paths:
        if path.exists():
            pkl_path = path
            break
    
    if pkl_path is None:
        raise FileNotFoundError(f"找不到mnist.pkl文件，尝试了以下路径:\n{chr(10).join(str(p) for p in possible_paths)}")
    
    print(f"使用数据文件: {pkl_path}")
    
    # 加载数据
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except:
        with gzip.open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    
    # 提取数据
    if isinstance(data, tuple) and len(data) == 2:
        (X_train, y_train), (X_test, y_test) = data
    elif isinstance(data, tuple) and len(data) == 3:
        (X_train, y_train), _, (X_test, y_test) = data
    elif isinstance(data, dict):
        # 格式3: 字典格式
        X_train = data.get('train_img', data.get('x_train'))
        y_train = data.get('train_label', data.get('y_train'))
        X_test = data.get('test_img', data.get('x_test'))
        y_test = data.get('test_label', data.get('y_test'))
    else:
        raise ValueError(f"未知的数据格式: {type(data)}")
    
    # 转换为numpy数组
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    # 重塑为图像格式 (batch, channel, height, width)
    if X_train.ndim == 2 and X_train.shape[1] == 784:
        X_train = X_train.reshape(-1, 1, 28, 28)  # (60000, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)    # (10000, 1, 28, 28)
    
    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print(f"PyTorch数据加载完成:")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def train_pytorch_lenet(data_dir, epochs=10, batch_size=64, lr=0.001, device='cuda'):
    """使用PyTorch训练LeNet"""
    
    # 设置设备
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用GPU进行训练")
    else:
        device = torch.device('cpu')
        print("使用CPU进行训练")
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data_pytorch(data_dir)
    
    # 转换为PyTorch Tensor
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    
    # 创建Dataset和DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 创建模型
    model = LeNetPyTorch(num_classes=10).to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch LeNet参数数量: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'time_per_epoch': []
    }
    
    # 训练循环
    print(f"\n开始PyTorch训练 ({epochs}个epoch)...")
    print("="*60)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_accuracy = 100. * train_correct / train_total
        test_accuracy = 100. * test_correct / test_total
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)
        history['time_per_epoch'].append(epoch_time)
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs} 结果:")
        print(f"训练损失: {avg_train_loss:.4f} | 训练准确率: {train_accuracy:.2f}%")
        print(f"测试损失: {avg_test_loss:.4f} | 测试准确率: {test_accuracy:.2f}%")
        print(f"时间: {epoch_time:.2f}秒")
        print("-"*60)
    
    # 最终结果
    print("\nPyTorch训练完成!")
    print(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")
    print(f"平均每epoch时间: {np.mean(history['time_per_epoch']):.2f}秒")
    
    return model, history

def plot_all_visualization(model, history, X_test, y_test, device, framework="PyTorch", save_path=None):
    """绘制与EnNeuro.Visualizer.plot_all()相同格式的训练可视化"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 2x2子图布局，与EnNeuro格式一致
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    fig.suptitle('Training Visualization', fontsize=16)
    
    # 1. 左上角：损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, history['test_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 右上角：准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[0, 1].plot(epochs, history['test_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 左下角：时间消耗曲线
    axes[1, 0].plot(epochs, history['time_per_epoch'])
    axes[1, 0].set_title('Time Consumption per Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True)
    
    # 4. 右下角：混淆矩阵
    # 获取预测结果
    model.eval()
    X_test_tensor = torch.from_numpy(X_test).to(device)
    
    # 分批处理以避免内存问题
    all_predictions = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            output = model(batch)
            predictions = output.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, all_predictions)
    
    # 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{framework}_training_visualization.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def test_pytorch_model(model, history, data_dir, device='cuda'):
    """测试PyTorch模型"""
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data_pytorch(data_dir)
    
    # 测试集评估
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f"PyTorch模型测试准确率: {accuracy:.2f}%")
    
    # 单个样本预测示例
    print("\n随机样本预测示例:")
    random_idx = np.random.randint(0, len(X_test))
    sample = X_test[random_idx:random_idx+1]
    sample_tensor = torch.from_numpy(sample).to(device)
    
    with torch.no_grad():
        output = model(sample_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
    
    true_class = y_test[random_idx]
    confidence = probabilities[0, predicted_class].item() * 100
    
    print(f"真实标签: {true_class}")
    print(f"预测标签: {predicted_class}")
    print(f"置信度: {confidence:.2f}%")
    print(f"{'正确' if predicted_class == true_class else '错误'}")
    
    return accuracy

if __name__ == "__main__":
    # 训练参数
    data_dir = "code/testdata/MNIST_data"
    epochs = 10
    batch_size = 64
    lr = 0.001
    
    # 训练PyTorch模型
    model, history = train_pytorch_lenet(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device='cpu'  # 可以改为'gpu'如果有GPU
    )
    
    # 加载数据用于可视化
    X_train, y_train, X_test, y_test = load_mnist_data_pytorch(data_dir)
    
    # 使用与EnNeuro相同格式的可视化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    plot_all_visualization(model, history, X_test, y_test, device, framework="PyTorch")
    
    # 测试模型
    test_pytorch_model(model, history, data_dir)
    
    # 保存模型
    #torch.save(model.state_dict(), "lenet_pytorch.pth")
    #print("模型已保存为 lenet_pytorch.pth")