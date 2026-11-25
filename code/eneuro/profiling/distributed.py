"""Read per-role distributed training logs for reports and benchmarks."""

import json
from pathlib import Path


_BENCHMARK_EVENTS = ('compute', 'push', 'pull', 'allreduce', 'step')
_COMMUNICATION_EVENTS = ('push', 'pull', 'allreduce', 'broadcast')


def _worker_paths(directory, world_size):
    directory = Path(directory)
    return [directory / f'worker-{rank}.jsonl' for rank in range(world_size)]


def _records(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def _role_summary(path):
    timings = {}
    samples = 0
    first_step_ns, last_step_ns = None, None
    for record in _records(path):
        event = record['event']
        duration = record['duration_ms']
        entry = timings.setdefault(event, {
            'count': 0, 'total_ms': 0.0, 'max_ms': duration,
            'bytes_sent': 0, 'bytes_received': 0,
        })
        entry['count'] += 1
        entry['total_ms'] += duration
        entry['max_ms'] = max(entry['max_ms'], duration)
        entry['bytes_sent'] += record['bytes_sent']
        entry['bytes_received'] += record['bytes_received']
        if event == 'compute':
            samples += record['sample_count']
        elif event == 'step':
            start_ns, end_ns = record['start_perf_ns'], record['end_perf_ns']
            first_step_ns = start_ns if first_step_ns is None else min(first_step_ns, start_ns)
            last_step_ns = end_ns if last_step_ns is None else max(last_step_ns, end_ns)
        if 'barrier_wait_ms' in record:
            entry['barrier_wait_ms'] = entry.get('barrier_wait_ms', 0) + record['barrier_wait_ms']
    for entry in timings.values():
        entry['mean_ms'] = entry['total_ms'] / entry['count']
    return {
        'samples': samples, 'timings': timings,
        'first_step_ns': first_step_ns, 'last_step_ns': last_step_ns,
    }


def summarize_timings(directory, *, backend, world_size):
    """Summarize one completed launch, retaining unavailable roles as errors."""
    paths = _worker_paths(directory, world_size)
    if backend == 'ps':
        paths.append(Path(directory) / 'ps.jsonl')
    roles, errors = {}, {}
    for path in paths:
        try:
            roles[path.stem] = _role_summary(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[path.stem] = str(exc)
    return {'backend': backend, 'roles': roles, 'errors': errors}


def benchmark_metrics(directory, world_size):
    """Aggregate worker durations, step-window throughput and communication."""
    roles = [_role_summary(path) for path in _worker_paths(directory, world_size)]
    completed = [role for role in roles if role['first_step_ns'] is not None]
    if not completed:
        raise ValueError('no completed training steps found')
    first_step_ns = min(role['first_step_ns'] for role in completed)
    last_step_ns = max(role['last_step_ns'] for role in completed)
    if last_step_ns <= first_step_ns:
        raise ValueError('no completed training steps found')
    training_s = (last_step_ns - first_step_ns) / 1e9
    durations = dict.fromkeys(_BENCHMARK_EVENTS, 0.0)
    samples = sent = received = 0
    for role in roles:
        samples += role['samples']
        for event, entry in role['timings'].items():
            if event in durations:
                durations[event] += entry['total_ms']
            if event in _COMMUNICATION_EVENTS:
                sent += entry['bytes_sent']
                received += entry['bytes_received']
    return {
        'training_elapsed_s': training_s,
        'samples_processed': samples,
        'train_samples_per_second': samples / training_s,
        **{f'{event}_ms_sum': value for event, value in durations.items()},
        'communication_ratio': (durations['push'] + durations['allreduce']) / durations['step'],
        'bytes_sent': sent,
        'bytes_received': received,
    }


def worker_loss_trace(directory, world_size):
    """Return one-based steps with sample-weighted loss and global sample counts."""
    by_step = {}
    for path in _worker_paths(directory, world_size):
        for record in _records(path):
            if record['event'] == 'compute':
                pieces = by_step.setdefault(record['step'], [])
                pieces.append((record['loss'], record['sample_count']))
    trace = []
    for step, pieces in sorted(by_step.items()):
        if len(pieces) != world_size:
            raise ValueError(f'incomplete loss trace at step {step}')
        count = sum(size for _, size in pieces)
        trace.append((step + 1, sum(loss * size for loss, size in pieces) / count, count))
    return trace
