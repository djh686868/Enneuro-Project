import pandas as pd
import numpy as np
import time
import sys

# 导入EnNeuro框架的核心组件
from eneuro.data.dataset import Dataset
from eneuro.data.dataloader import DataLoader
from eneuro.nn.module import MLP
from eneuro.base import functions as F
from eneuro.base import Tensor
from eneuro.nn.optim import SGD
from eneuro.nn.loss import crossEntropyError,sigmoidCrossEntropy,softmaxCrossEntropy,meanSquaredError
from eneuro.train import Trainer, Evaluator
from eneuro.utils import Visualizer

# 自定义鸢尾花数据集类
class IrisDataset(Dataset):
    def __init__(self, file_path, train=True, transform=None, target_transform=None):
        # 先设置属性，再调用父类初始化
        self.file_path = file_path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = None
        self.label = None
        self.class_map = None
        
        # 调用父类初始化，这可能会触发prepare()
        super().__init__()
    
    def prepare(self):
        # 读取CSV文件
        df = pd.read_csv(self.file_path)
        
        # 提取特征和标签
        self.data = df.iloc[:, :-1].values.astype(np.float32)
        
        # 处理标签（字符串转整数）
        y_raw = df.iloc[:, -1].values
        self.class_map = {name: idx for idx, name in enumerate(np.unique(y_raw))}
        self.label = np.array([self.class_map[v] for v in y_raw], dtype=np.int64)
        
        # 特征标准化
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        self.data = (self.data - mean) / (std + 1e-8)
    
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

# 自定义DataLoader实现，因为框架提供的DataLoader是未实现的
'''
class SimpleDataLoader(DataLoader):
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
'''
# 训练和评估函数
def train_iris_classifier():
    print("开始测试EnNeuro框架 - 鸢尾花分类任务")
    
    # 1. 准备数据
    # 使用绝对路径确保文件能被正确找到
    file_path = "testdata\\Iris.csv"
    
    # 加载完整数据集
    full_dataset = IrisDataset(file_path)
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
    
    train_dataset = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(full_dataset, val_indices)
    test_dataset = SubsetDataset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"数据集加载完成: 训练集({len(train_dataset)}), 验证集({len(val_dataset)}), 测试集({len(test_dataset)})")
    
    # 2. 创建模型
    input_dim = 4  # 鸢尾花有4个特征
    hidden_dim = 12
    output_dim = 3  # 3个类别
    
    # 创建MLP模型
    model = MLP([input_dim, hidden_dim, output_dim], activation=F.relu)
    print("模型创建完成")
    
    # 3. 设置损失函数和优化器
    loss_fn = crossEntropyError
    optimizer = SGD(model.params(), lr=0.1)

    visualizer = Visualizer(num_classes=output_dim)
    
    # 4. 创建训练器
    trainer = Trainer(model, loss_fn, optimizer, visualizer)
    
    # 5. 训练模型
    trainer.fit(train_loader, val_loader, epochs=100, batch_size=32, verbose=True)

    # 6. 评估模型
    evaluator = Evaluator(model, loss_fn, visualizer)
    evaluator.evaluate(test_loader, batch_size=32, verbose=True)
    
    # 7. 绘制所有图表
    visualizer.plot_all()
    
# 运行测试
if __name__ == "__main__":
    train_iris_classifier()

