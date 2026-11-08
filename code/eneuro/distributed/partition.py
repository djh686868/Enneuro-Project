"""Deterministic global batches, divided among workers without overlap."""

import numpy as np


def worker_batches(size, global_batch_size, world_size, rank, seed, epoch, *, shuffle=True, drop_last=True):
    if size < 0 or global_batch_size < world_size or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('invalid dataset, batch, or worker configuration')
    indices = np.random.default_rng(seed + epoch).permutation(size) if shuffle else np.arange(size)
    batches = []
    for start in range(0, size, global_batch_size):
        batch = indices[start:start + global_batch_size]
        if len(batch) < global_batch_size and drop_last:
            break
        # A final batch smaller than world_size cannot supply every BSP worker.
        if len(batch) < world_size:
            raise ValueError('final batch is smaller than world_size; use drop_last or a larger dataset')
        batches.append(batch[rank::world_size].tolist())
    return batches
