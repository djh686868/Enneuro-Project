import numpy as np
from eneuro.nn.module import MLP
from eneuro.base import functions as F
from eneuro.base import Tensor
from eneuro.nn.optim import SGD,Adam
from eneuro.nn.loss import crossEntropyError, sigmoidCrossEntropy, softmaxCrossEntropy, meanSquaredError
from eneuro.train import Trainer, Evaluator
from eneuro.data import DataLoader,Dataset

import pandas as pd
from sklearn.datasets import make_regression

# 设置随机种子以便复现
np.random.seed(42)

class RegressionDatasetGenerator:
    """生成回归问题数据集的类"""
    
    def __init__(self, n_samples=100, n_features=5, noise_level=0.1, random_state=42):
        """
        参数:
        n_samples: 样本数量
        n_features: 特征数量
        noise_level: 噪声水平（0-1之间）
        random_state: 随机种子
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        self.random_state = random_state
        
        # 定义特征名称
        self.feature_names = [f'特征_{i+1}' for i in range(n_features)]
        
        # 生成数据
        self.generate_data()
        
    def generate_data(self):
        """生成回归数据集"""
        
        # 方法1：使用scikit-learn的make_regression生成线性关系数据
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise_level * 10,  # 调整噪声大小
            random_state=self.random_state
        )
        
        # 方法2：可以生成非线性关系（可选）
        # 这里我们保留线性关系，但可以添加一个可选的非线性变换
        
        # 为特征和目标添加更有意义的名称
        self.X = X
        self.y = y
        
        # 创建数据框以便于查看
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['目标值'] = self.y
        
    def add_nonlinearity(self, degree=2):
        """为数据添加非线性关系"""
        # 对目标值应用非线性变换
        self.y = self.y ** degree + np.random.normal(0, self.noise_level * 5, size=self.y.shape)
        self.df['目标值'] = self.y
        return self
    
    def add_outliers(self, n_outliers=5, outlier_scale=5):
        """添加离群点"""
        indices = np.random.choice(len(self.y), n_outliers, replace=False)
        self.y[indices] += np.random.normal(0, outlier_scale, n_outliers) * np.std(self.y)
        self.df['目标值'] = self.y
        return self
    
    def get_data(self):
        """返回特征和目标值"""
        return self.X, self.y
    
    def get_dataframe(self):
        """返回包含特征和目标值的DataFrame"""
        return self.df
    
    def describe(self):
        """显示数据集的统计描述"""
        print("=" * 50)
        print("回归数据集统计信息")
        print("=" * 50)
        print(f"样本数量: {self.n_samples}")
        print(f"特征数量: {self.n_features}")
        print(f"噪声水平: {self.noise_level}")
        print("\n目标值统计:")
        print(f"  均值: {self.y.mean():.4f}")
        print(f"  标准差: {self.y.std():.4f}")
        print(f"  最小值: {self.y.min():.4f}")
        print(f"  最大值: {self.y.max():.4f}")
        
        # 特征和目标值的相关性
        correlations = []
        for i in range(self.n_features):
            corr = np.corrcoef(self.X[:, i], self.y)[0, 1]
            correlations.append(corr)
        
        print("\n特征与目标值的相关性:")
        for i, corr in enumerate(correlations):
            print(f"  特征_{i+1}: {corr:.4f}")
        
        return self

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
    print("非线性回归数据集")

    dataset = RegressionDatasetGenerator(
        n_samples=1000,
        n_features=4,
        noise_level=0.01
    )

    # 添加一些离群点
    #dataset.add_outliers(n_outliers=3)
    
    # 显示统计信息
    dataset.describe()

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
            if len(self.dataset.label.shape) == 1:
                return 1
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
    loss_fn = meanSquaredError
    optimizer = Adam(model.params(), lr=0.01)
    
    # 6. 训练模型
    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(train_loader, test_loader, epochs=100, batch_size=32, verbose=True)
    
    # 7. 评估模型
    evaluator = Evaluator(model, loss_fn)
    test_loss, test_acc = evaluator.evaluate(test_loader, batch_size=32, verbose=True)
    
    print(f"\n最终测试结果: 准确率={test_acc:.4f}, 损失={test_loss:.4f}")

if __name__ == "__main__":
    test()