from . import Dataset
import random
from typing import Tuple, List
from ..core import Tensor

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> '_LoaderIter':
        return _LoaderIter(self)   # 每次都新建一个迭代器
    
    def __len__(self) -> int:
        """返回DataLoader的批次数"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class _LoaderIter:
    def __init__(self, loader: DataLoader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        
        # 生成索引
        self.indices = list(range(len(self.dataset)))
        if loader.shuffle:
            random.shuffle(self.indices)
        
        self.cursor = 0
        self.num_samples = len(self.indices)

    def __iter__(self) -> '_LoaderIter':
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        """返回下一个批量数据"""
        
        # 检查是否已经遍历完所有样本
        if self.cursor >= self.num_samples:
            raise StopIteration
        
        # 检查是否剩余样本不足一个batch且设置了drop_last
        remaining = self.num_samples - self.cursor
        if remaining < self.batch_size and self.drop_last:
            raise StopIteration
        
        # 获取当前batch的实际大小
        current_batch_size = min(self.batch_size, remaining)
        
        # 获取当前批次的索引
        batch_indices = self.indices[self.cursor:self.cursor + current_batch_size]
        self.cursor += current_batch_size
        
        # 批量加载数据
        batch_data, batch_target = self._load_batch(batch_indices)
        
        return batch_data, batch_target
    
    def _load_batch(self, indices: List[int]) -> Tuple[Tensor, Tensor]:
        """加载指定索引的批量数据"""
        batch_data = []
        batch_target = []
        
        for idx in indices:
            # 从数据集中获取单个样本
            data, target = self.dataset[idx]
            batch_data.append(data)
            batch_target.append(target)
        
        # 将列表中的样本堆叠成批量Tensor
        batch_data_tensor = Tensor.stack(batch_data)
        batch_target_tensor = Tensor.stack(batch_target)
        
        return batch_data_tensor, batch_target_tensor
