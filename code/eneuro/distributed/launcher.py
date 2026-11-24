"""Windows-safe launch of local synchronous data-parallel training."""

from dataclasses import dataclass, field, replace
from pathlib import Path
import multiprocessing as mp
from queue import Empty
import time

from .client import PSClient
from .server import run_server
from .worker import run_worker


@dataclass
class DistributedConfig:
    host: str = '127.0.0.1'
    port: int = 29500
    world_size: int = 2
    epochs: int = 1
    global_batch_size: int = 32
    seed: int = 42
    timeout_s: float = 30.0
    shuffle: bool = True
    drop_last: bool = True
    metrics_dir: Path = field(default_factory=lambda: Path('metrics/ps'))
    checkpoint_path: Path | None = None
    backend: str = 'ps'


def launch(model_factory, dataset_factory, loss_fn, optimizer_factory, config=None):
    """Factories and loss function must be importable top-level callables."""
    config = config or DistributedConfig()
    if (config.world_size < 1 or config.global_batch_size < config.world_size
            or config.epochs < 1 or config.timeout_s <= 0):
        raise ValueError('invalid distributed configuration')
    config = replace(config, metrics_dir=Path(config.metrics_dir))
    context = mp.get_context('spawn')
    if config.backend == 'ps':
        result = _launch_ps(context, model_factory, dataset_factory, loss_fn, optimizer_factory, config)
    elif config.backend == 'ring':
        from .ring import launch_ring
        result = launch_ring(context, model_factory, dataset_factory, loss_fn, optimizer_factory, config)
    else:
        raise ValueError(f"unknown distributed backend {config.backend!r}, expected 'ps' or 'ring'")
    if config.checkpoint_path is not None:
        from ..utils.serializer import Serializer
        from .state import load_state
        model = model_factory()
        load_state(model, result['parameters'])
        optimizer = optimizer_factory(model.get_params_list())
        optimizer.from_dict(result['optimizer_state'])
        path = Path(config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Serializer.save_checkpoint(model, optimizer, config.epochs, str(path))
    return result


def _launch_ps(context, model_factory, dataset_factory, loss_fn, optimizer_factory, config):
    ready, errors = context.Event(), context.Queue()
    bound_port = context.RawValue('i', 0)
    last_update = context.Value('d', time.monotonic())
    ps = context.Process(target=run_server,
                         args=(model_factory, optimizer_factory, config, ready, errors, bound_port, last_update),
                         name='ParameterServer')
    workers = []
    ps.start()
    try:
        if not ready.wait(config.timeout_s) or not bound_port.value:
            raise RuntimeError('parameter server did not start')
        config = replace(config, port=bound_port.value)
        for rank in range(config.world_size):
            process = context.Process(target=run_worker,
                                      args=(rank, model_factory, dataset_factory, loss_fn, config, errors),
                                      name=f'Worker-{rank}')
            process.start()
            workers.append(process)
        started = time.monotonic()
        for process in workers:
            while process.is_alive():
                remaining = config.timeout_s - (time.monotonic() - max(started, last_update.value))
                if remaining <= 0:
                    raise TimeoutError('parameter server made no training progress within timeout')
                process.join(remaining)
        failures = []
        while True:
            try:
                failures.append(errors.get_nowait())
            except Empty:
                break
        if failures or any(process.exitcode != 0 for process in workers):
            raise RuntimeError(f'distributed training failed: {failures}')
        with PSClient(config.host, config.port, config.timeout_s) as client:
            result, parameters, _, _ = client.request({'type': 'RESULT'})
        return {'model_version': result['model_version'], 'parameters': parameters,
                'optimizer_state': result['optimizer_state']}
    finally:
        if ps.is_alive() and bound_port.value:
            try:
                with PSClient(config.host, config.port, 2) as client:
                    client.request({'type': 'SHUTDOWN'})
            except (OSError, RuntimeError):
                pass
        for process in workers + [ps]:
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(2)
