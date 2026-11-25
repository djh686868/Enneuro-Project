"""Synchronous parameter server: exactly one optimizer update per global step."""

import json
import socketserver
import threading
import time

from .aggregator import aggregate
from .monitor import Monitor
from .protocol import recv_message, send_message
from .state import named_state, install_gradients


class ServerState:
    def __init__(self, model, optimizer, world_size, timeout, monitor, last_update=None):
        self.model, self.optimizer = model, optimizer
        self.world_size, self.timeout, self.monitor = world_size, timeout, monitor
        self.last_update = last_update
        self.condition = threading.Condition()
        self.registered = set()
        self.pending = {}
        self.version = 0
        self.failure = None
        self.shutting_down = False
        self.shapes = {name: param.data.shape for name, param in model.named_parameters() if param.requires_grad}
        self.manifest = {name: [list(param.data.shape), param.data.dtype.str]
                         for name, param in model.named_parameters()}

    def handle(self, message, arrays):
        kind = message.get('type')
        if kind == 'REGISTER':
            rank = message.get('worker_id')
            with self.condition:
                if (not isinstance(rank, int) or not 0 <= rank < self.world_size or
                        rank in self.registered or message.get('world_size') != self.world_size or
                        message.get('manifest') != self.manifest):
                    raise ValueError('invalid or duplicate worker registration')
                self.registered.add(rank)
                self.condition.notify_all()
            return {'type': 'REGISTER_ACK', 'model_version': self.version}, {}
        if kind == 'PULL':
            with self.condition:
                if message.get('worker_id') not in self.registered:
                    raise ValueError('unregistered worker')
                return {'type': 'UPDATE', 'model_version': self.version}, named_state(self.model)
        if kind == 'HEARTBEAT':
            if message.get('worker_id') not in self.registered:
                raise ValueError('unregistered worker')
            return {'type': 'HEARTBEAT_ACK', 'model_version': self.version}, {}
        if kind == 'PUSH':
            return self._push(message, arrays)
        if kind == 'RESULT':
            with self.condition:
                reply = {'type': 'RESULT', 'model_version': self.version}
                values = named_state(self.model)
                state = self.optimizer.to_dict()
                # Old clients expect inline JSON. New clients can carry large
                # Adam/Momentum states in the body, outside the 1 MiB header.
                if message.get('optimizer_payload'):
                    import numpy as np
                    key = '__optimizer_state__'
                    while key in values:
                        key += '_'
                    encoded = json.dumps(state, separators=(',', ':'), allow_nan=False).encode('utf-8')
                    values[key] = np.frombuffer(encoded, dtype=np.uint8)
                    reply['optimizer_payload_key'] = key
                else:
                    reply['optimizer_state'] = state
                return reply, values
        if kind == 'SHUTDOWN':
            self.shutting_down = True
            return {'type': 'SHUTDOWN_ACK'}, {}
        raise ValueError(f'unknown message type: {kind}')

    def _push(self, message, gradients):
        rank, step = message.get('worker_id'), message.get('step_id')
        count, version = message.get('sample_count'), message.get('model_version')
        start = time.perf_counter_ns()
        with self.condition:
            if rank not in self.registered:
                raise ValueError('unregistered worker')
            registration_deadline = time.monotonic() + self.timeout
            while len(self.registered) != self.world_size:
                remaining = registration_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('worker registration timeout')
                self.condition.wait(remaining)
            if self.failure:
                raise RuntimeError(self.failure)
            if step != self.version or version != self.version:
                raise ValueError('stale or future gradient version')
            if rank in self.pending:
                raise ValueError('duplicate gradient push')
            # Validate before adding to the barrier, including missing gradients.
            aggregate([(count, gradients)], self.shapes)
            self.pending[rank] = (count, gradients)
            if len(self.pending) == self.world_size:
                try:
                    tick = time.perf_counter_ns()
                    averaged = aggregate([self.pending[r] for r in sorted(self.pending)], self.shapes)
                    install_gradients(self.model, averaged)
                    self.optimizer.step()
                    self.model.cleargrads()
                    self.pending.clear()
                    self.version += 1
                    if self.last_update is not None:
                        self.last_update.value = time.monotonic()
                    self.monitor.log('aggregate_update', tick, step=step)
                    self.condition.notify_all()
                except Exception as exc:
                    self.failure = str(exc)
                    self.condition.notify_all()
                    raise
            else:
                deadline = time.monotonic() + self.timeout
                while self.version == version and not self.failure:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.failure = f'barrier timeout at step {step}'
                        self.condition.notify_all()
                        break
                    self.condition.wait(remaining)
                if self.failure:
                    raise RuntimeError(self.failure)
            waited = (time.perf_counter_ns() - start) / 1e6
            return {'type': 'UPDATE', 'model_version': self.version,
                    'barrier_wait_ms': waited}, named_state(self.model)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                message, arrays, _ = recv_message(self.request)
            except ConnectionError:
                return
            except (OSError, ValueError, KeyError, TypeError) as exc:
                try:
                    send_message(self.request, {'type': 'ERROR', 'message': str(exc)})
                except OSError:
                    pass
                return
            try:
                reply, values = self.server.state.handle(message, arrays)
                send_message(self.request, reply, values)
                if message.get('type') == 'SHUTDOWN':
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                try:
                    send_message(self.request, {'type': 'ERROR', 'message': str(exc)})
                except OSError:
                    pass
                return


class _TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128


def run_server(model_factory, optimizer_factory, config, ready, errors, bound_port=None, last_update=None):
    try:
        import numpy as np
        np.random.seed(config.seed)
        model = model_factory()
        if any(param.data is None for _, param in model.named_parameters()):
            raise ValueError('model parameters must be initialized before PS startup')
        optimizer = optimizer_factory(model.get_params_list())
        monitor = Monitor(config.metrics_dir / 'ps.jsonl', 'ps')
        state = ServerState(model, optimizer, config.world_size, config.timeout_s, monitor, last_update)
        with _TCPServer((config.host, config.port), _Handler) as server:
            server.state = state
            if bound_port is not None:
                bound_port.value = server.server_address[1]
            ready.set()
            server.serve_forever(poll_interval=0.1)
    except Exception as exc:
        errors.put(('ps', repr(exc)))
        ready.set()
