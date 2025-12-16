# file name: generic_dataset.py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from eneuro.data.dataset import Dataset

class GenericDataset(Dataset):
    """通用数据集类，支持多种数据格式"""
    
    def __init__(self, 
                 data_source,  # 可以是文件路径、numpy数组、pandas DataFrame等
                 target_column=None,  # 目标列名（对于CSV/DataFrame）
                 feature_columns=None,  # 特征列名列表（None表示除目标列外的所有列）
                 train=True,
                 test_size=0.2,
                 random_state=42,
                 transform=None,
                 target_transform=None,
                 normalize=True):
        
        self.data_source = data_source
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        self.data = None
        self.label = None
        self.scaler = None
        self.label_encoder = None
        
        # 调用父类初始化
        super().__init__()
    
    def prepare(self):
        """加载和准备数据"""
        # 根据数据类型加载数据
        if isinstance(self.data_source, str):
            # 文件路径
            self._load_from_file()
        elif isinstance(self.data_source, np.ndarray):
            # numpy数组
            self._load_from_array()
        elif isinstance(self.data_source, pd.DataFrame):
            # pandas DataFrame
            self._load_from_dataframe()
        elif isinstance(self.data_source, tuple):
            # (X, y) 元组
            self._load_from_tuple()
        else:
            raise ValueError(f"不支持的数据源类型: {type(self.data_source)}")
        
        # 数据预处理
        self._preprocess()
    
    def _load_from_file(self):
        """从文件加载数据"""
        file_path = self.data_source
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            self._load_from_dataframe(df)
        elif file_ext == '.npy':
            data = np.load(file_path, allow_pickle=True)
            self._load_from_array(data)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
            self._load_from_dataframe(df)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    def _load_from_dataframe(self, df=None):
        """从DataFrame加载数据"""
        if df is None:
            df = self.data_source
        
        if self.target_column is None:
            raise ValueError("对于DataFrame数据，必须指定target_column")
        
        # 分离特征和目标
        if self.feature_columns:
            X = df[self.feature_columns].values
        else:
            # 使用除目标列外的所有列作为特征
            X = df.drop(columns=[self.target_column]).values
        
        y = df[self.target_column].values
        
        self._load_from_tuple((X, y))
    
    def _load_from_array(self, data=None):
        """从numpy数组加载数据"""
        if data is None:
            data = self.data_source
        
        if data.ndim == 1:
            # 一维数组，只有特征或只有目标
            raise ValueError("对于numpy数组，请提供(X, y)元组")
        elif data.ndim == 2:
            # 二维数组，假设最后一列是目标
            X = data[:, :-1]
            y = data[:, -1]
            self._load_from_tuple((X, y))
        else:
            raise ValueError(f"不支持的数组维度: {data.ndim}")
    
    def _load_from_tuple(self, data_tuple=None):
        """从(X, y)元组加载数据"""
        if data_tuple is None:
            data_tuple = self.data_source
        
        X, y = data_tuple
        
        # 确保是float32类型
        X = X.astype(np.float32)
        
        # 如果是字符串标签，编码为整数
        if y.dtype == object or y.dtype.type == np.str_:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # 分割训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        if self.train:
            self.data, self.label = X_train, y_train
            self.test_data, self.test_label = X_test, y_test
        else:
            self.data, self.label = X_test, y_test
            self.test_data, self.test_label = X_train, y_train
    
    def _preprocess(self):
        """数据预处理"""
        if self.normalize and self.data is not None:
            self.scaler = StandardScaler()
            
            if self.train:
                self.data = self.scaler.fit_transform(self.data)
                if hasattr(self, 'test_data'):
                    self.test_data = self.scaler.transform(self.test_data)
            else:
                # 如果是测试模式，应该已经拟合过scaler
                # 在实际应用中，这里需要加载之前拟合的scaler
                pass
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        
        # 应用转换
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
    def get_num_classes(self):
        """获取类别数量"""
        return len(np.unique(self.label))
    
    def get_feature_dim(self):
        """获取特征维度"""
        return self.data.shape[1] if self.data is not None else None