"""Windows-safe spawn orchestration for a local BSP parameter server."""

from dataclasses import dataclass, field
from pathlib import Path
import multiprocessing as mp
from queue import Empty

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


def launch(model_factory, dataset_factory, loss_fn, optimizer_factory, config=None):
    """Factories and loss function must be importable top-level callables."""
    config = config or DistributedConfig()
    if config.world_size < 1 or config.global_batch_size < config.world_size or config.epochs < 1:
        raise ValueError('invalid distributed configuration')
    config.metrics_dir = Path(config.metrics_dir)
    context = mp.get_context('spawn')
    ready, errors = context.Event(), context.Queue()
    ps = context.Process(target=run_server,
                         args=(model_factory, optimizer_factory, config, ready, errors), name='ParameterServer')
    workers = []
    ps.start()
    try:
        if not ready.wait(config.timeout_s) or not ps.is_alive():
            raise RuntimeError('parameter server did not start')
        for rank in range(config.world_size):
            process = context.Process(target=run_worker,
                                      args=(rank, model_factory, dataset_factory, loss_fn, config, errors),
                                      name=f'Worker-{rank}')
            process.start()
            workers.append(process)
        for process in workers:
            process.join(config.timeout_s * max(2, config.epochs))
            if process.is_alive():
                raise TimeoutError(f'{process.name} exceeded training timeout')
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
        if config.checkpoint_path is not None:
            from ..utils.serializer import Serializer
            from .state import load_state
            model = model_factory()
            load_state(model, parameters)
            optimizer = optimizer_factory(model.get_params_list())
            optimizer.from_dict(result['optimizer_state'])
            path = Path(config.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Serializer.save_checkpoint(model, optimizer, config.epochs, str(path))
        return {'model_version': result['model_version'], 'parameters': parameters,
                'optimizer_state': result['optimizer_state']}
    finally:
        if ps.is_alive():
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
