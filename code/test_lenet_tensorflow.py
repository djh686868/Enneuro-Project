# file name: test_lenet_tensorflow.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import gzip
from sklearn.metrics import confusion_matrix
import seaborn as sns

class LeNetTensorFlow(models.Model):
    """TensorFlow版本的LeNet，与PyTorch版本保持一致"""
    
    def __init__(self, num_classes=10):
        super(LeNetTensorFlow, self).__init__()
        
        # 卷积层定义
        self.conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='valid', activation='sigmoid')
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2)
        
        self.conv2 = layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation='sigmoid')
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)
        
        # 全连接层定义
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='sigmoid')
        self.fc2 = layers.Dense(84, activation='sigmoid')
        self.fc3 = layers.Dense(num_classes, activation=None)
    
    def call(self, x):
        # 卷积块1: Conv -> Sigmoid -> MaxPool
        x = self.conv1(x)
        x = self.pool1(x)
        
        # 卷积块2: Conv -> Sigmoid -> MaxPool
        x = self.conv2(x)
        x = self.pool2(x)
        
        # 全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

def load_mnist_data_tensorflow(data_dir):
    """为TensorFlow加载MNIST数据"""
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
    
    # 重塑为图像格式 (batch, height, width, channel)
    if X_train.ndim == 2 and X_train.shape[1] == 784:
        X_train = X_train.reshape(-1, 28, 28, 1)  # (60000, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)    # (10000, 28, 28, 1)
    
    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print(f"TensorFlow数据加载完成:")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def train_tensorflow_lenet(data_dir, epochs=10, batch_size=64, lr=0.001, device='gpu'):
    """使用TensorFlow训练LeNet"""
    
    # 设置设备
    if device == 'gpu' and tf.config.list_physical_devices('GPU'):
        device = '/GPU:0'
        print("使用GPU进行训练")
    else:
        device = '/CPU:0'
        print("使用CPU进行训练")
    
    with tf.device(device):
        # 加载数据
        X_train, y_train, X_test, y_test = load_mnist_data_tensorflow(data_dir)
        
        # 创建数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 创建模型
        model = LeNetTensorFlow(num_classes=10)
        
        # 统计参数数量
        # 先让模型处理一个样本，自动构建
        sample_input = tf.random.normal(shape=(1, 28, 28, 1))
        _ = model(sample_input)
        total_params = model.count_params()
        print(f"TensorFlow LeNet参数数量: {total_params:,}")
        
        # 定义损失函数和优化器
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # 指标
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'time_per_epoch': []
        }
        
        # 训练循环
        print(f"\n开始TensorFlow训练 ({epochs}个epoch)...")
        print("="*60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练阶段
            train_loss = 0.0
            train_accuracy.reset_state()
            
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)
                    loss = loss_fn(y_batch, logits)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                train_loss += loss.numpy()
                train_accuracy.update_state(y_batch, logits)
                
                # 打印进度
                if (step + 1) % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} | Batch: {step+1}/{len(train_dataset)} | '\
                          f'Loss: {loss.numpy():.4f}')
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_dataset)
            
            # 测试阶段
            test_loss = 0.0
            test_accuracy.reset_state()
            
            for x_batch, y_batch in test_dataset:
                logits = model(x_batch, training=False)
                loss = loss_fn(y_batch, logits)
                
                test_loss += loss.numpy()
                test_accuracy.update_state(y_batch, logits)
            
            # 计算平均测试损失
            avg_test_loss = test_loss / len(test_dataset)
            
            # 计算时间
            epoch_time = time.time() - start_time
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy.result().numpy() * 100)
            history['test_loss'].append(avg_test_loss)
            history['test_acc'].append(test_accuracy.result().numpy() * 100)
            history['time_per_epoch'].append(epoch_time)
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{epochs} 结果:")
            print(f"训练损失: {avg_train_loss:.4f} | 训练准确率: {history['train_acc'][-1]:.2f}%")
            print(f"测试损失: {avg_test_loss:.4f} | 测试准确率: {history['test_acc'][-1]:.2f}%")
            print(f"时间: {epoch_time:.2f}秒")
            print("-"*60)
        
        # 最终结果
        print("\nTensorFlow训练完成!")
        print(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")
        print(f"平均每epoch时间: {np.mean(history['time_per_epoch']):.2f}秒")
        
        return model, history

def plot_all_visualization(model, history, X_test, y_test, framework="TensorFlow", save_path=None):
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
    logits = model(X_test, training=False)
    predictions = tf.argmax(logits, axis=1).numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
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

def test_tensorflow_model(model, history, data_dir):
    """测试TensorFlow模型"""
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data_tensorflow(data_dir)
    
    # 计算测试准确率
    logits = model(X_test, training=False)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_accuracy.update_state(y_test, logits)
    accuracy = test_accuracy.result().numpy() * 100
    
    print(f"TensorFlow模型测试准确率: {accuracy:.2f}%")
    
    # 单个样本预测示例
    print("\n随机样本预测示例:")
    random_idx = np.random.randint(0, len(X_test))
    sample = X_test[random_idx:random_idx+1]
    
    logits = model(sample, training=False)
    probabilities = tf.nn.softmax(logits, axis=1)
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    
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
    
    # 训练TensorFlow模型
    model, history = train_tensorflow_lenet(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device='cpu'  # 可以改为'gpu'如果有GPU
    )
    
    # 加载数据用于可视化
    X_train, y_train, X_test, y_test = load_mnist_data_tensorflow(data_dir)
    
    # 使用与EnNeuro相同格式的可视化
    plot_all_visualization(model, history, X_test, y_test, framework="TensorFlow")
    
    # 测试模型
    test_tensorflow_model(model, history, data_dir)
    
    # 保存模型
    # model.save("lenet_tensorflow", save_format='tf')
    # print("模型已保存")
