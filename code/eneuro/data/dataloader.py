import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from operator import index

from .dataset import Dataset
from ..base import Tensor, as_Tensor
from ..base.core import get_array_module


def _load_batch(dataset, indices, *, prefetch=False):
    samples, targets = zip(*(dataset[i] for i in indices))
    if any(target is None for target in targets):
        if not all(target is None for target in targets):
            raise ValueError("Cannot batch labeled and unlabeled samples together")
        return _stack(samples, prefetch), None
    return _stack(samples, prefetch), _stack(targets, prefetch)


def _stack(values, prefetch):
    if isinstance(values[0], Tensor):
        batch = Tensor.stack(values)
    else:
        xp = get_array_module(values[0])
        batch = as_Tensor(xp.stack(values))
    if prefetch and batch.device != 'cpu':
        # A completed Python future can still have pending work on its CUDA stream.
        with batch.data.device:
            get_array_module(batch).cuda.get_current_stream().synchronize()
    return batch


class _AsyncLoaderIter:
    """Keep a bounded window of batches in flight and yield them in sampler order."""

    _executor = None

    def __init__(self, dataset, batches, total_batches, num_workers, prefetch_factor):
        self._dataset = dataset
        self._batches = batches
        self._total_batches = total_batches
        self._pending = deque()
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="EnNeuro-DataLoader"
        )
        try:
            for _ in range(min(total_batches, num_workers * prefetch_factor)):
                self._submit_next()
        except BaseException:
            self.close()
            raise

    def _submit_next(self):
        indices = next(self._batches, None)
        if indices is not None:
            future = self._executor.submit(_load_batch, self._dataset, indices, prefetch=True)
            self._pending.append(future)

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_batches

    def __next__(self):
        if not self._pending:
            self.close()
            raise StopIteration
        try:
            batch = self._pending.popleft().result()
            self._submit_next()
            if not self._pending:
                self.close()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        """Cancel queued batches. Workers finish any reads already in progress."""
        if self._executor is not None:
            for future in self._pending:
                future.cancel()
            self._pending.clear()
            self._executor.shutdown(wait=False)
            self._executor = None
            self._batches = iter(())

    def __del__(self):
        self.close()


class DataLoader:
    """按采样顺序返回批次，num_workers=0 时在调用线程中读取。

    num_workers>0 时，后台线程共享同一个数据集，可同时读取文件、解码图片。
    自定义 __getitem__ 和变换函数需支持并发读取。每个迭代器最多预取
    num_workers * prefetch_factor 个批次，提前停止时可调用迭代器的 close()。
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, num_workers: int = 0, prefetch_factor: int = 2):
        self.dataset = dataset
        self.batch_size = index(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = index(num_workers)
        self.prefetch_factor = index(prefetch_factor)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.num_workers and self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive when using workers")

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        size = self.batch_size
        stop = len(indices)
        if self.drop_last:
            stop -= stop % size
        starts = range(0, stop, size)
        batches = (indices[start:start + size] for start in starts)
        if self.num_workers:
            return _AsyncLoaderIter(self.dataset, batches, len(starts),
                                    self.num_workers, self.prefetch_factor)
        return (_load_batch(self.dataset, batch) for batch in batches)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
