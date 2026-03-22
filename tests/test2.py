# file name: train_mnist.py
import numpy as np
from sklearn.datasets import load_digits
from generic_dataset import GenericDataset
from eneuro.nn.module import MLP
from eneuro.base import functions as F
from eneuro.base import Tensor
from eneuro.nn.optim import SGD
from eneuro.nn.loss import crossEntropyError
from eneuro.train import Trainer, Evaluator
from eneuro.data import DataLoader

def train_digits_classifier():
    """训练手写数字分类器"""
    print("开始训练手写数字分类器...")
    
    # 1. 加载数据
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 2. 创建数据集
    train_dataset = GenericDataset(
        data_source=(X, y),
        train=True,
        test_size=0.2,
        normalize=True
    )
    
    test_dataset = GenericDataset(
        data_source=(X, y),
        train=False,
        test_size=0.2,
        normalize=True
    )
    
    print(f"数据集加载完成: 训练集({len(train_dataset)}), 测试集({len(test_dataset)})")
    print(f"特征维度: {train_dataset.get_feature_dim()}, 类别数: {train_dataset.get_num_classes()}")
    
    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建模型（调整输入输出维度）
    input_dim = train_dataset.get_feature_dim()  # 64 (8x8图像展平)
    hidden_dim = 32
    output_dim = train_dataset.get_num_classes()  # 10
    
    model = MLP([input_dim, hidden_dim, output_dim], activation=F.relu)
    print(f"模型创建完成: 输入={input_dim}, 隐藏层={hidden_dim}, 输出={output_dim}")
    
    # 5. 设置损失函数和优化器
    loss_fn = crossEntropyError
    optimizer = SGD(model.params(), lr=0.01)
    
    # 6. 训练模型
    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(train_loader, test_loader, epochs=50, batch_size=32, verbose=True)
    
    # 7. 评估模型
    evaluator = Evaluator(model, loss_fn)
    test_loss, test_acc = evaluator.evaluate(test_loader, batch_size=32, verbose=True)
    
    print(f"\n最终测试结果: 准确率={test_acc:.4f}, 损失={test_loss:.4f}")

if __name__ == "__main__":
    train_digits_classifier()