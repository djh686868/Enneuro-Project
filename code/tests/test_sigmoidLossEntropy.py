import numpy as np
import sys
from pathlib import Path

# 添加code目录到Python搜索路径，这样就能找到eneuro模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.nn.module import MLP
from eneuro.base import functions as F
from eneuro.base import Tensor
from eneuro.nn.optim import SGD
from eneuro.nn.loss import crossEntropyError, sigmoidCrossEntropy, softmaxCrossEntropy, meanSquaredError
from eneuro.train import Trainer, Evaluator
from eneuro.data import DataLoader,Dataset

import pandas as pd

# 设置随机种子以便复现
np.random.seed(42)

class MultiLabelDatasetGenerator:
    """生成多标签分类数据集的类"""
    
    def __init__(self, n_samples=50, n_classes=5, n_labels_per_sample_range=(1, 3)):
        """
        参数:
        n_samples: 样本数量
        n_classes: 标签类别数量
        n_labels_per_sample_range: 每个样本的标签数量范围（最小，最大）
        """
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_labels_range = n_labels_per_sample_range
        
        # 定义电影类型标签
        self.class_names = ['动作', '喜剧', '爱情', '科幻', '恐怖']
        
        # 生成模拟数据
        self.generate_data()
        
    def generate_data(self):
        """生成模拟的多标签数据"""
        
        # 方法1：使用scikit-learn的make_multilabel_classification
        # X, y = make_multilabel_classification(
        #     n_samples=self.n_samples,
        #     n_features=10,  # 特征数量
        #     n_classes=self.n_classes,
        #     n_labels=2,  # 平均标签数量
        #     random_state=42
        # )
        
        # 方法2：手动生成更可控的数据
        self.X = np.random.randn(self.n_samples, 10)  # 10个特征
        self.y = np.zeros((self.n_samples, self.n_classes), dtype=int)
        
        # 为每个样本随机分配1-3个标签
        for i in range(self.n_samples):
            n_labels = np.random.randint(*self.n_labels_range)
            labels = np.random.choice(self.n_classes, n_labels, replace=False)
            self.y[i, labels] = 1
            
        # 确保每个类别至少有一些样本
        for j in range(self.n_classes):
            if np.sum(self.y[:, j]) == 0:
                # 随机选择一个样本添加该标签
                sample_idx = np.random.randint(0, self.n_samples)
                self.y[sample_idx, j] = 1
    
    def get_data(self):
        """返回特征和标签"""
        return self.X, self.y
    
    def get_dataframe(self):
        """返回DataFrame格式的数据"""
        # 创建特征列名
        feature_cols = [f'feature_{i}' for i in range(self.X.shape[1])]
        label_cols = self.class_names
        
        # 创建DataFrame
        df_features = pd.DataFrame(self.X, columns=feature_cols)
        df_labels = pd.DataFrame(self.y, columns=label_cols)
        
        # 添加标签的文本表示
        label_texts = []
        for i in range(self.n_samples):
            labels = [self.class_names[j] for j in range(self.n_classes) if self.y[i, j] == 1]
            label_texts.append(', '.join(labels))
        
        df_features['labels_text'] = label_texts
        df = pd.concat([df_features, df_labels], axis=1)
        
        return df
    
    def get_class_distribution(self):
        """返回每个类别的分布"""
        class_counts = np.sum(self.y, axis=0)
        distribution = pd.DataFrame({
            '类别': self.class_names,
            '样本数': class_counts,
            '占比(%)': (class_counts / self.n_samples * 100).round(2)
        })
        return distribution

class TestDataset(Dataset):
    def __init__(self, data_tuple, train=True, transform=None, target_transform=None):
        # 先设置属性，再调用父类初始化
        self.data_tuple = data_tuple
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = None
        self.label = None
        
        # 调用父类初始化，这可能会触发prepare()
        super().__init__()
    
    def prepare(self):
        self.data, self.label = self.data_tuple
        self.data = self.data.astype(np.float32)
        self.label = self.label.astype(np.float32)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        
        # 应用转换（如果有）
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def test():
    print("开始训练...")
    
    # 1. 加载数据
    # 生成数据
    print("=" * 50)
    print("多标签分类数据集生成")
    print("=" * 50)

    dataset = MultiLabelDatasetGenerator(
        n_samples=50, 
        n_classes=5, 
        n_labels_per_sample_range=(1, 3)
    )

    X, y = dataset.get_data()
    
    # 2. 创建数据集
    # 加载完整数据集
    full_dataset = TestDataset((X, y))
    data_size = len(full_dataset)
    
    # 分割训练集和测试集
    train_size = int(0.6 * data_size)
    val_size = int(0.2 * data_size)
    test_size = data_size - train_size - val_size
    
    # 使用numpy进行简单的数据分割
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    # 创建训练集和测试集
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建子数据集
    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices):
            super().__init__()
            self.dataset = dataset
            self.indices = indices
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
        
        def __len__(self):
            return len(self.indices)
    
        def get_feature_dim(self):
            return self.dataset.data.shape[1]
        
        def get_num_classes(self):
            return self.dataset.label.shape[1]
    
    train_dataset = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(full_dataset, val_indices)
    test_dataset = SubsetDataset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"数据集加载完成: 训练集({len(train_dataset)}), 验证集({len(val_dataset)}), 测试集({len(test_dataset)})")
    print(f"特征维度: {train_dataset.get_feature_dim()}, 类别数: {train_dataset.get_num_classes()}")
    
    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建模型（调整输入输出维度）
    input_dim = train_dataset.get_feature_dim()
    hidden_dim = 32
    output_dim = train_dataset.get_num_classes()
    
    model = MLP([input_dim, hidden_dim, output_dim], activation=F.relu)
    print(f"模型创建完成: 输入={input_dim}, 隐藏层={hidden_dim}, 输出={output_dim}")
    
    # 5. 设置损失函数和优化器
    loss_fn = sigmoidCrossEntropy
    optimizer = SGD(model.params(), lr=0.01)
    
    # 6. 训练模型
    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(train_loader, test_loader, epochs=50, batch_size=32, verbose=True)
    
    # 7. 评估模型
    evaluator = Evaluator(model, loss_fn)
    test_loss, test_acc = evaluator.evaluate(test_loader, batch_size=32, verbose=True)
    
    print(f"\n最终测试结果: 准确率={test_acc:.4f}, 损失={test_loss:.4f}")

if __name__ == "__main__":
    test()