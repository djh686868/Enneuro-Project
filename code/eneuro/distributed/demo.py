"""Small deterministic CPU workload for distributed smoke tests and benchmarks.

Run from the code directory: python -m eneuro.distributed.demo --backend ring --workers 2
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from ..base import Tensor
from ..nn.module import Sequential, Linear
from ..nn.optim import SGD, Adam
from ..nn.loss import meanSquaredError
from ..train import DistributedConfig, launch_distributed


class ToyDataset:
    def __init__(self):
        self.x = np.arange(32, dtype=np.float32).reshape(16, 2) / 32
        self.y = (self.x[:, :1] * 1.5 - self.x[:, 1:] * 0.7).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return Tensor(self.x[index]), Tensor(self.y[index])


def make_model():
    np.random.seed(42)
    return Sequential(Linear(1, in_size=2))


def make_dataset():
    return ToyDataset()


def make_optimizer(parameters):
    return SGD(parameters, lr=0.05)


def make_adam_optimizer(parameters):
    return Adam(parameters, lr=0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('ps', 'ring'), default='ps')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--port', type=int, default=0, help='PS listener port (0 selects a free port)')
    parser.add_argument('--metrics-dir', type=Path)
    args = parser.parse_args()
    config = DistributedConfig(backend=args.backend, world_size=args.workers, port=args.port,
                               metrics_dir=args.metrics_dir or Path('metrics') / f'{args.backend}-demo',
                               global_batch_size=4, shuffle=False, epochs=2)
    start = time.perf_counter()
    result = launch_distributed(make_model, make_dataset, meanSquaredError, make_optimizer, config)
    elapsed = time.perf_counter() - start
    print(json.dumps({'backend': args.backend, 'workers': args.workers, 'steps': result['model_version'],
                      'job_samples_per_second': len(ToyDataset()) * config.epochs / elapsed,
                      'job_elapsed_s': elapsed}, indent=2))


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
