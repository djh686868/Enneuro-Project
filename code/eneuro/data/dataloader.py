from . import Dataset

class DataLoader:
    def __init__(self) -> None:
        self.dataset: Dataset
        self.batch_size: int
        self.drop_last: bool

    def __iter__(self):
        return _LoaderIter(self)   # 每次都新建一个迭代器
    
class _LoaderIter:
    def __init__(self, loader: DataLoader):
        '''
        self.dataset    = loader.dataset
        self.batch_size = loader.batch_size
        self.indices    = list(range(len(self.dataset)))
        if loader.shuffle:
            random.shuffle(self.indices)
        self.cursor = 0
        '''
        pass

    def __iter__(self):          # 迭代器协议
        return self

    def __next__(self):          # 返回下一个 batch
        '''
        if self.cursor >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        # 把样本堆成两个列表再一次性转 Tensor，效率最高
        batch_data, batch_target = [], []
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_data.append(x)
            batch_target.append(y)
        return Tensor.stack(batch_data), Tensor.stack(batch_target)
        '''
        pass