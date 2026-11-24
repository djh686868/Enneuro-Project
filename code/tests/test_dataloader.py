import random
import sys
from pathlib import Path
from threading import Event, current_thread

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eneuro.base import Tensor
from eneuro.data import DataLoader, Dataset


@pytest.mark.parametrize("length,batch_size,drop_last", [
    (0, 3, False), (2, 4, True), (8, 3, True), (8, 3, False),
])
@pytest.mark.parametrize("shuffle", [False, True])
def test_async_matches_sync_across_epochs(length, batch_size, drop_last, shuffle):
    dataset = [(np.array([i, -i], dtype=np.float32), i) for i in range(length)]

    def collect(workers):
        loader = DataLoader(dataset, batch_size, shuffle, drop_last, workers)
        epochs = []
        for _ in range(2):
            batches = list(loader)
            assert len(batches) == len(loader)
            epoch = []
            for samples, labels in batches:
                assert isinstance(samples, Tensor) and isinstance(labels, Tensor)
                assert samples.data.dtype == np.float32
                assert np.issubdtype(labels.data.dtype, np.integer)
                np.testing.assert_array_equal(samples.data[:, 0], labels.data)
                np.testing.assert_array_equal(samples.data[:, 1], -labels.data)
                epoch.append(labels.data.tolist())
            epochs.append(epoch)
        return epochs

    state = random.getstate()
    try:
        sampler = random.Random(71)
        expected = []
        for _ in range(2):
            order = list(range(length))
            if shuffle:
                sampler.shuffle(order)
            if drop_last:
                order = order[:length // batch_size * batch_size]
            expected.append([order[i:i + batch_size] for i in range(0, len(order), batch_size)])
        for workers in (0, 2):
            random.seed(71)
            assert collect(workers) == expected
    finally:
        random.setstate(state)


def test_yield_order_when_first_read_finishes_last():
    second_read = Event()

    class DelayedDataset:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            if i == 0:
                assert second_read.wait(5), "second worker did not prefetch"
            else:
                second_read.set()
            return Tensor(i), Tensor(i)

    batches = list(DataLoader(DelayedDataset(), num_workers=2))
    assert [labels.data.item() for _, labels in batches] == [0, 1]


def test_prefetch_window_and_early_close():
    window_filled, window_exceeded, release = Event(), Event(), Event()
    reads = []
    workers = set()

    class CountingDataset:
        def __len__(self):
            return 50

        def __getitem__(self, i):
            workers.add(current_thread())
            reads.append(i)
            if i == 1:
                window_filled.set()
            elif i == 2:
                window_exceeded.set()
                assert release.wait(5)
            return Tensor(i), Tensor(i)

    iterator = iter(DataLoader(CountingDataset(), num_workers=1, prefetch_factor=2))
    try:
        assert window_filled.wait(5)
        assert not window_exceeded.wait(0.1), "prefetched more than two batches"
        assert next(iterator)[1].data.item() == 0
        assert window_exceeded.wait(5)
        assert next(iterator)[1].data.item() == 1
        # The worker is reading index 2. Index 3 is queued and must be cancelled.
        iterator.close()
    finally:
        release.set()
        iterator.close()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert reads == [0, 1, 2]
    with pytest.raises(StopIteration):
        next(iterator)


def test_worker_exception_reaches_consumer_and_next_epoch_can_run():
    class FailOnce:
        fail = True

        def __len__(self):
            return 3

        def __getitem__(self, i):
            if i == 1 and self.fail:
                self.fail = False
                raise OSError("broken sample")
            return Tensor(i), Tensor(i)

    loader = DataLoader(FailOnce(), num_workers=2)
    iterator = iter(loader)
    assert next(iterator)[1].data.item() == 0
    with pytest.raises(OSError, match="broken sample"):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    assert [y.data.item() for _, y in loader] == [0, 1, 2]


@pytest.mark.parametrize("workers", [0, 2])
def test_unlabeled_and_partially_labeled_batches(workers):
    dataset = Dataset()
    dataset.data = np.arange(6).reshape(2, 3)
    loader = DataLoader(dataset, batch_size=2, num_workers=workers)
    (samples, labels), = loader
    np.testing.assert_array_equal(samples.data, dataset.data)
    assert labels is None

    dataset.label = [None, 1]
    with pytest.raises(ValueError, match="labeled and unlabeled"):
        list(loader)


@pytest.mark.cuda
def test_gpu_batches_ready_on_consumer_stream():
    cp = pytest.importorskip("cupy")
    if not cp.is_available():
        pytest.skip("CUDA device unavailable")

    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)

    class GpuDataset:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            producer.use()
            x = cp.full(65536, i + 1, dtype=cp.int32)
            return Tensor(x, device="cuda"), Tensor(cp.array(i), device="cuda")

    loader = DataLoader(GpuDataset(), batch_size=2, num_workers=1)
    with consumer:
        (samples, labels), = loader
        assert samples.device == labels.device == "cuda"
        np.testing.assert_array_equal(samples.data.sum(axis=1).get(), [65536, 131072])
        np.testing.assert_array_equal(labels.data.get(), [0, 1])
