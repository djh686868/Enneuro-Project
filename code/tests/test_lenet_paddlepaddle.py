# file name: test_lenet_paddlepaddle.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import gzip
from sklearn.metrics import confusion_matrix
import seaborn as sns

class LeNetPaddlePaddle(nn.Layer):
    """PaddlePaddle版本的LeNet，与其他框架保持一致"""
    
    def __init__(self, num_classes=10):
        super(LeNetPaddlePaddle, self).__init__()
        
        # 卷积层定义
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        
        # 全连接层定义
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        # 卷积块1: Conv -> Sigmoid -> MaxPool
        x = F.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 卷积块2: Conv -> Sigmoid -> MaxPool
        x = F.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        
        # 全连接层
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        
        return x

def load_mnist_data_paddlepaddle(data_dir):
    """为PaddlePaddle加载MNIST数据"""
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
    
    print(f"PaddlePaddle数据加载完成:")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def train_paddlepaddle_lenet(data_dir, epochs=10, batch_size=64, lr=0.001, device='gpu'):
    """使用PaddlePaddle训练LeNet"""
    
    # 设置设备
    if device == 'gpu' and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
        print("使用GPU进行训练")
    else:
        paddle.set_device('cpu')
        print("使用CPU进行训练")
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data_paddlepaddle(data_dir)
    
    # 创建数据集
    train_dataset = paddle.io.TensorDataset([X_train, y_train])
    train_loader = paddle.io.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows下只能使用单进程模式
        use_shared_memory=False
    )
    
    test_dataset = paddle.io.TensorDataset([X_test, y_test])
    test_loader = paddle.io.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Windows下只能使用单进程模式
        use_shared_memory=False
    )
    
    # 创建模型
    model = LeNetPaddlePaddle(num_classes=10)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PaddlePaddle LeNet参数数量: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'time_per_epoch': []
    }
    
    # 训练循环
    print(f"\n开始PaddlePaddle训练 ({epochs}个epoch)...")
    print("="*60)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_id, (data, target) in enumerate(train_loader):
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            # 统计
            train_loss += loss.numpy()
            predicted = paddle.argmax(output, axis=1)
            train_correct += (predicted == target).sum().numpy()
            train_total += target.shape[0]
            
            # 打印进度
            if (batch_id + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_id+1}/{len(train_loader)} | '
                      f'Loss: {loss.numpy():.4f}')
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with paddle.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.numpy()
                predicted = paddle.argmax(output, axis=1)
                test_correct += (predicted == target).sum().numpy()
                test_total += target.shape[0]
        
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
    print("\nPaddlePaddle训练完成!")
    print(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")
    print(f"平均每epoch时间: {np.mean(history['time_per_epoch']):.2f}秒")
    
    return model, history

def plot_all_visualization(model, history, X_test, y_test, framework="PaddlePaddle", save_path=None):
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
    with paddle.no_grad():
        # 分批处理以避免内存问题
        all_predictions = []
        batch_size = 128
        for i in range(0, len(X_test), batch_size):
            batch = paddle.to_tensor(X_test[i:i+batch_size])
            output = model(batch)
            predictions = paddle.argmax(output, axis=1).numpy()
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

def test_paddlepaddle_model(model, history, data_dir):
    """测试PaddlePaddle模型"""
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data_paddlepaddle(data_dir)
    
    # 创建测试数据加载器
    test_dataset = paddle.io.TensorDataset([X_test, y_test])
    test_loader = paddle.io.DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=2
    )
    
    # 计算测试准确率
    model.eval()
    correct = 0
    total = 0
    
    with paddle.no_grad():
        for data, target in test_loader:
            output = model(data)
            predicted = paddle.argmax(output, axis=1)
            correct += (predicted == target).sum().numpy()
            total += target.shape[0]
    
    accuracy = 100. * correct / total
    print(f"PaddlePaddle模型测试准确率: {accuracy:.2f}%")
    
    # 单个样本预测示例
    print("\n随机样本预测示例:")
    random_idx = np.random.randint(0, len(X_test))
    sample = X_test[random_idx:random_idx+1]
    sample_tensor = paddle.to_tensor(sample)
    
    with paddle.no_grad():
        output = model(sample_tensor)
        probabilities = F.softmax(output, axis=1)
        predicted_class = paddle.argmax(output, axis=1).numpy()[0]
    
    true_class = y_test[random_idx]
    confidence = probabilities[0, predicted_class].numpy() * 100
    
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
    
    # 训练PaddlePaddle模型
    model, history = train_paddlepaddle_lenet(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device='cpu'  # 可以改为'gpu'如果有GPU
    )
    
    # 加载数据用于可视化
    X_train, y_train, X_test, y_test = load_mnist_data_paddlepaddle(data_dir)
    
    # 使用与EnNeuro相同格式的可视化
    plot_all_visualization(model, history, X_test, y_test, framework="PaddlePaddle")
    
    # 测试模型
    test_paddlepaddle_model(model, history, data_dir)
    
    # 保存模型
    # paddle.save(model.state_dict(), "lenet_paddlepaddle.pdparams")
    # paddle.save(optimizer.state_dict(), "lenet_paddlepaddle.pdopt")
    # print("模型已保存")
