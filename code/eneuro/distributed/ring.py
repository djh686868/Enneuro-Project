"""Single-host ring with a reduce-scatter and an allgather on peer TCP streams."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from multiprocessing.connection import wait
import socket
import time

import numpy as np

from ..train.trainer import Trainer
from .monitor import Monitor
from .protocol import recv_message, send_message
from .state import install_gradients, load_state, named_state
from .worker import gradient_steps


def _receive_into(sock, chunk):
    remaining = memoryview(chunk).cast('B')
    while remaining:
        size = sock.recv_into(remaining)
        if not size:
            raise ConnectionError('predecessor closed the ring')
        remaining = remaining[size:]


def _allreduce(values, rank, world_size, incoming, outgoing, sender):
    """Sum a contiguous vector in-place. Every rank owns the full sum on return."""
    chunks = np.array_split(values, world_size)
    sent = received = 0
    # At reduce-scatter hop h rank r forwards r-h and accumulates r-h-1.
    # After P-1 hops its completed chunk is r+1, the first allgather send.
    for reduce in (True, False):
        first = rank if reduce else rank + 1
        for hop in range(world_size - 1):
            source = chunks[(first - hop) % world_size]
            destination = chunks[(first - hop - 1) % world_size]
            target = np.empty_like(destination) if reduce else destination
            # Simultaneous send/receive is necessary when chunks exceed socket buffers.
            pending = sender.submit(outgoing.sendall, source)
            _receive_into(incoming, target)
            pending.result()
            if reduce:
                destination += target
            sent += source.nbytes
            received += target.nbytes
    return sent, received


class GradientBuffer:
    """Stable parameter slices followed by the sample count in the same collective."""

    def __init__(self, model):
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.values = np.empty(1 + sum(p.data.size for _, p in params), dtype=np.float64)
        self.views = {}
        start = 0
        for name, p in params:
            end = start + p.data.size
            self.views[name] = self.values[start:end].reshape(p.data.shape)
            start = end

    def pack(self, gradients, count):
        if gradients.keys() != self.views.keys() or count <= 0:
            raise ValueError('gradient names or sample count mismatch')
        self.values[-1] = count
        for name, view in self.views.items():
            gradient = gradients[name]
            if gradient.shape != view.shape or not np.all(np.isfinite(gradient)):
                raise ValueError(f'invalid gradient for {name}')
            np.multiply(gradient, count, out=view, dtype=self.values.dtype)

    def mean(self):
        self.values[:-1] /= self.values[-1]
        return self.views


def _broadcast(rank, world_size, model, dataset, incoming, outgoing, monitor):
    """Pass rank zero's initial parameters once around the ring before optimization."""
    tick = time.perf_counter_ns()
    manifest = {'trainable': [n for n, p in model.named_parameters() if p.requires_grad],
                'dataset_size': len(dataset)}
    sent = received = 0
    if rank:
        header, arrays, received = recv_message(incoming)
        if any(header[key] != value for key, value in manifest.items()):
            raise ValueError('ring workers have different parameters or dataset lengths')
        load_state(model, arrays)
    if rank < world_size - 1:
        sent = send_message(outgoing, manifest, named_state(model))
    monitor.log('broadcast', tick, bytes_sent=sent, bytes_received=received)


def _run_rank(rank, model_factory, dataset_factory, loss_fn, optimizer_factory, config,
              incoming, outgoing, report):
    with report, ExitStack() as resources:
        try:
            for sock in (incoming, outgoing):
                if sock is not None:
                    resources.enter_context(sock)
                    sock.settimeout(config.timeout_s)
            np.random.seed(config.seed)
            model = model_factory()
            if any(p.data is None for _, p in model.named_parameters()):
                raise ValueError('model parameters must be initialized before ring startup')
            dataset = dataset_factory()
            monitor = Monitor(config.metrics_dir / f'worker-{rank}.jsonl', 'worker', rank)
            _broadcast(rank, config.world_size, model, dataset, incoming, outgoing, monitor)
            trainer = Trainer(model, loss_fn, optimizer=None)
            optimizer = optimizer_factory(model.get_params_list())
            buffer = GradientBuffer(model)
            # PS owns a copy of frozen parameters, including BatchNorm buffers.
            # Preserve that policy even if a worker's forward pass mutates a buffer.
            frozen = [(p, p.data.copy()) for _, p in model.named_parameters() if not p.requires_grad]
            sender = resources.enter_context(ThreadPoolExecutor(max_workers=1))
            version = 0
            # Report progress sparsely so a healthy job can run for many hours.
            report.send(('progress', version))
            interval = min(5.0, config.timeout_s / 4)
            next_report = time.monotonic() + interval
            for step, start, count, gradients in gradient_steps(rank, trainer, dataset, config, monitor):
                tick = time.perf_counter_ns()
                buffer.pack(gradients, count)
                sent, received = _allreduce(buffer.values, rank, config.world_size, incoming,
                                            outgoing, sender)
                averaged = buffer.mean()
                monitor.log('allreduce', tick, step=step, bytes_sent=sent, bytes_received=received)
                install_gradients(model, averaged)
                optimizer.step()
                model.cleargrads()
                for parameter, initial in frozen:
                    parameter.data[...] = initial
                version += 1
                monitor.log('step', start, step=step, model_version=version)
                if time.monotonic() >= next_report:
                    report.send(('progress', version))
                    next_report = time.monotonic() + interval
            result = None
            if rank == 0:
                result = {'model_version': version, 'parameters': named_state(model),
                          'optimizer_state': optimizer.to_dict()}
            report.send(('done', result))
        except Exception as exc:
            report.send(('error', repr(exc)))


def _make_links(config, resources):
    """Establish one TCP connection per edge before spawn, with no port races."""
    incoming = [None] * config.world_size
    outgoing = [None] * config.world_size
    if config.world_size > 1:
        with socket.socket() as listener:
            # All workers are local. Use an ephemeral loopback port for wiring so
            # independent jobs never contend for a fixed parameter-server address.
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            listener.settimeout(config.timeout_s)
            for rank in range(config.world_size):
                source = resources.enter_context(socket.create_connection(listener.getsockname(), config.timeout_s))
                destination = resources.enter_context(listener.accept()[0])
                source.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                outgoing[rank] = source
                incoming[(rank + 1) % config.world_size] = destination
    return incoming, outgoing


def launch_ring(context, model_factory, dataset_factory, loss_fn, optimizer_factory, config):
    workers = []
    reports = {}
    try:
        with ExitStack() as resources:
            # Closing parent copies after spawn lets a failed peer signal EOF immediately.
            with ExitStack() as links:
                incoming, outgoing = _make_links(config, links)
                for rank in range(config.world_size):
                    reader, writer = context.Pipe(duplex=False)
                    resources.enter_context(reader)
                    reports[reader] = rank
                    with writer:
                        process = context.Process(
                            target=_run_rank, name=f'RingWorker-{rank}',
                            args=(rank, model_factory, dataset_factory, loss_fn, optimizer_factory,
                                  config, incoming[rank], outgoing[rank], writer))
                        process.start()
                        workers.append(process)
            result = None
            deadlines = {pipe: time.monotonic() + config.timeout_s for pipe in reports}
            while reports:
                ready = wait(reports, timeout=max(0, min(deadlines.values()) - time.monotonic()))
                if not ready:
                    stalled = [rank for pipe, rank in reports.items() if deadlines[pipe] <= time.monotonic()]
                    raise TimeoutError(f'ring workers made no progress within timeout: {stalled}')
                for report in ready:
                    rank = reports[report]
                    try:
                        status, value = report.recv()
                    except EOFError as exc:
                        raise RuntimeError(f'ring worker {rank} exited without a result') from exc
                    if status == 'progress':
                        deadlines[report] = time.monotonic() + config.timeout_s
                        continue
                    del reports[report], deadlines[report]
                    if status == 'error':
                        raise RuntimeError(f'ring worker {rank} failed: {value}')
                    if rank == 0:
                        result = value
            for process in workers:
                process.join(config.timeout_s)
                if process.exitcode != 0:
                    raise RuntimeError(f'{process.name} did not exit successfully')
            return result
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
