from . import Dataset
import random
from typing import Tuple, List
from ..base import Tensor
import multiprocessing as mp
from multiprocessing import Queue, Process
import numpy as np


def _worker_loop(dataset, batch_indices_list, batch_size, drop_last, output_queue):
    """
    子进程中的数据加载循环函数
    
    """
    cursor = 0
    total_indices=len(batch_indices_list)


    while cursor < total_indices:
        remaining = total_indices - cursor
        current_batch_size = min(batch_size, remaining)
        
        if current_batch_size < batch_size and drop_last:
            break
        
        batch_indices = batch_indices_list[cursor:cursor + current_batch_size]
        cursor += current_batch_size
        
        batch_data = []
        batch_target = []
        
        for idx in batch_indices:
            data, target = dataset[idx]
            batch_data.append(data)
            batch_target.append(target)
        
        batch_data_tensor = Tensor.stack(batch_data)
        batch_target_tensor = Tensor.stack(batch_target)
        
        output_queue.put((batch_data_tensor, batch_target_tensor))
    
    output_queue.put(None)



class _SyncLoaderIter:
    """
    同步加载迭代器，当num_workers=0时使用
    
    在主进程中顺序加载数据，不涉及多进程开销
    """
    
    def __init__(self, loader: 'DataLoader'):
        self.loader = loader
        self.indices = list(range(len(loader.dataset)))
        if loader.shuffle:
            random.shuffle(self.indices)
        self.cursor = 0
        self.num_samples = len(self.indices)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """返回下一个batch数据"""
        if self.cursor >= self.num_samples:
            raise StopIteration
        
        remaining = self.num_samples - self.cursor
        if remaining < self.loader.batch_size and self.loader.drop_last:
            raise StopIteration
        
        current_batch_size = min(self.loader.batch_size, remaining)
        batch_indices = self.indices[self.cursor:self.cursor + current_batch_size]
        self.cursor += current_batch_size
        
        batch_data, batch_target = self._load_batch(batch_indices)
        return batch_data, batch_target
    
    def _load_batch(self, indices):
        """根据索引列表加载一个batch的数据"""
        batch_data = []
        batch_target = []
        for idx in indices:
            data, target = self.loader.dataset[idx]
            batch_data.append(data)
            batch_target.append(target)
        return Tensor.stack(batch_data), Tensor.stack(batch_target)


class _AsyncLoaderIter:
    """
    异步加载迭代器,当num_workers>0时使用
    
    使用多个子进程并行加载数据,通过Queue进行进程间通信
    """
    
    def __init__(self, loader: 'DataLoader'):
        self.loader = loader
        self.num_workers = loader.num_workers
        self.prefetch_factor = loader.prefetch_factor

        #生成当前epoch的索引顺序
        indices=list(range(len(loader.dataset)))
        if loader.shuffle:
            random.shuffle(indices)
        
        #计算总的批次数
        total_batches = len(self)

        #将批次分配给各个worker(按连续批次划分)
        batches_per_worker = (total_batches + self.num_workers - 1) // self.num_workers
        self.workers=[]
        self.output_queue=Queue(maxsize=self.prefetch_factor*self.num_workers)
        
        for w in range(self.num_workers):
            start_batch = w * batches_per_worker
            end_batch = min(start_batch + batches_per_worker, total_batches)
            if start_batch>=end_batch:
                continue

            worker_indices=[]
            for batch_idx in range(start_batch,end_batch):
                batch_start=batch_idx*loader.batch_size
                batch_end=min(batch_start+loader.batch_size,len(indices))
                worker_indices.extend(indices[batch_start:batch_end])

            p = Process(target=_worker_loop, args=(
                loader.dataset,
                worker_indices,
                loader.batch_size,
                loader.drop_last,
                self.output_queue
            ))
            p.start()
            self.workers.append(p)
        
        self._batches_yielded = 0
        self._total_batches = total_batches
        self._workers_finished = 0
        self._shutdown = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        获取下一个可用的batch数据
        
        优先从已有数据的队列中取，若都为空则阻塞等待
        """
        if self._batches_yielded >= self._total_batches:
            self._cleanup()
            raise StopIteration
        
        while True:
            result = self.output_queue.get()
            if result is None:
                self._workers_finished += 1
                if self._workers_finished == len(self.workers):
                    self._cleanup()
                    raise StopIteration
                continue
            self._batches_yielded += 1
            return result
    
    def _cleanup(self):
        """
        清理所有子进程，释放资源
        
        依次发送终止信号、等待退出，若超时则强制杀死进程
        """
        if self._shutdown:
            return
        self._shutdown = True
        for p in self.workers:
            if p in self.workers:
               p.terminate()
               p.join(timeout=1)
               if p.is_alive():
                p.kill()
    
    def __del__(self):
        """析构时确保进程被正确清理"""
        self._cleanup()

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False
                 ,num_workers:int=0,prefetch_factor:int=2) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def __iter__(self):
        if self.num_workers == 0:
            return _SyncLoaderIter(self)
        else:
            return _AsyncLoaderIter(self)
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
