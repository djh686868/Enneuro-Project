"""Worker gradient computation over a deterministic data shard."""

import time
import numpy as np

from ..base import Tensor
from ..train.trainer import Trainer
from .client import PSClient
from .monitor import Monitor
from .partition import worker_batches
from .state import load_state


def run_worker(rank, model_factory, dataset_factory, loss_fn, config, errors):
    client = None
    try:
        np.random.seed(config.seed)
        model = model_factory()
        dataset = dataset_factory()
        trainer = Trainer(model, loss_fn, optimizer=None)
        client = PSClient(config.host, config.port, config.timeout_s)
        monitor = Monitor(config.metrics_dir / f'worker-{rank}.jsonl', 'worker', rank)
        manifest = {name: [list(param.data.shape), param.data.dtype.str]
                    for name, param in model.named_parameters()}
        client.request({'type': 'REGISTER', 'worker_id': rank,
                        'world_size': config.world_size, 'manifest': manifest})
        client.request({'type': 'HEARTBEAT', 'worker_id': rank, 'step_id': 0})
        tick = time.perf_counter_ns()
        reply, arrays, sent, received = client.request({'type': 'PULL', 'worker_id': rank})
        monitor.log('pull', tick, bytes_sent=sent, bytes_received=received)
        load_state(model, arrays)
        version = reply['model_version']
        global_step = 0
        for epoch in range(config.epochs):
            batches = worker_batches(len(dataset), config.global_batch_size, config.world_size,
                                     rank, config.seed, epoch, shuffle=config.shuffle,
                                     drop_last=config.drop_last)
            for indices in batches:
                step_start = time.perf_counter_ns()
                samples = [dataset[index] for index in indices]
                xb = Tensor.stack([Tensor(np.asarray(sample[0])) for sample in samples])
                yb = Tensor.stack([Tensor(np.asarray(sample[1])) for sample in samples])
                tick = time.perf_counter_ns()
                loss, count, gradients = trainer.compute_batch_gradients(xb, yb)
                monitor.log('compute', tick, step=global_step, loss=loss, sample_count=count)
                tick = time.perf_counter_ns()
                reply, arrays, sent, received = client.request(
                    {'type': 'PUSH', 'worker_id': rank, 'step_id': global_step,
                     'model_version': version, 'sample_count': count}, gradients)
                monitor.log('push', tick, step=global_step, bytes_sent=sent,
                            bytes_received=received, barrier_wait_ms=reply['barrier_wait_ms'])
                load_state(model, arrays)
                version = reply['model_version']
                monitor.log('step', step_start, step=global_step, model_version=version)
                global_step += 1
    except Exception as exc:
        errors.put((f'worker-{rank}', repr(exc)))
    finally:
        if client is not None:
            client.close()
