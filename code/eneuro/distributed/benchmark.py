"""Small reproducible 1/2/4-worker PS benchmark; write CSV for plotting."""

import argparse
import csv
import json
import socket
import statistics
import time
from functools import partial
from pathlib import Path

from ..nn.loss import meanSquaredError, crossEntropyError
from .demo import make_model, make_dataset, make_optimizer
from .mnist import make_lenet, make_mnist_dataset, make_optimizer as make_mnist_optimizer
from .autodrive import (make_autodrive_model, make_steering_dataset,
                        make_optimizer as make_autodrive_optimizer)
from .launcher import DistributedConfig, launch


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _metrics(directory, world_size):
    compute_ms = push_ms = pull_ms = step_ms = bytes_sent = bytes_received = samples = 0
    first_step_ns, last_step_ns = None, None
    for rank in range(world_size):
        with (directory / f'worker-{rank}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                entry = json.loads(line)
                if entry['event'] == 'compute':
                    compute_ms += entry['duration_ms']
                    samples += entry['sample_count']
                if entry['event'] in ('push', 'pull'):
                    if entry['event'] == 'push':
                        push_ms += entry['duration_ms']
                    else:
                        pull_ms += entry['duration_ms']
                    bytes_sent += entry['bytes_sent']
                    bytes_received += entry['bytes_received']
                elif entry['event'] == 'step':
                    step_ms += entry['duration_ms']
                    start_ns, end_ns = entry['start_perf_ns'], entry['end_perf_ns']
                    first_step_ns = start_ns if first_step_ns is None else min(first_step_ns, start_ns)
                    last_step_ns = end_ns if last_step_ns is None else max(last_step_ns, end_ns)
    training_s = (last_step_ns - first_step_ns) / 1e9 if first_step_ns is not None else 0
    return compute_ms, push_ms, pull_ms, step_ms, bytes_sent, bytes_received, samples, training_s


def _loss_trace(directory, world_size):
    by_step = {}
    for rank in range(world_size):
        with (directory / f'worker-{rank}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                entry = json.loads(line)
                if entry['event'] == 'compute':
                    by_step.setdefault(entry['step'], []).append((entry['loss'], entry['sample_count']))
    trace = []
    for step, pieces in sorted(by_step.items()):
        if len(pieces) != world_size:
            raise ValueError(f'incomplete loss trace at step {step}')
        count = sum(size for _, size in pieces)
        trace.append((step + 1, sum(loss * size for loss, size in pieces) / count, count))
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--dataset', choices=['toy', 'mnist', 'autodrive'], default='toy')
    parser.add_argument('--samples', type=int, default=1024, help='MNIST or autodrive training subset size')
    parser.add_argument('--batch-size', type=int, default=None, help='global batch size, constant across worker counts')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--timeout-s', type=float, default=120)
    parser.add_argument('--mnist-path', type=Path, default=None)
    parser.add_argument('--data-dir', type=Path, default=None, help='autodrive data directory containing driving_log.csv')
    parser.add_argument('--output', type=Path, default=Path('metrics/ps-benchmark'))
    args = parser.parse_args()
    if args.dataset == 'mnist':
        model_factory, loss_fn, optimizer_factory = make_lenet, crossEntropyError, make_mnist_optimizer
        model_name = 'LeNetMNIST'
        dataset_factory = partial(make_mnist_dataset, limit=args.samples, **(
            {'path': args.mnist_path} if args.mnist_path else {}))
        batch_size = args.batch_size or 64
        epochs = args.epochs or 1
    elif args.dataset == 'autodrive':
        model_factory, loss_fn, optimizer_factory = (make_autodrive_model, meanSquaredError,
                                                      make_autodrive_optimizer)
        model_name = 'PSAutoDriveNet'
        dataset_factory = partial(make_steering_dataset, limit=args.samples, **(
            {'data_dir': args.data_dir} if args.data_dir else {}))
        batch_size = args.batch_size or 32
        epochs = args.epochs or 1
    else:
        model_factory, dataset_factory = make_model, make_dataset
        loss_fn, optimizer_factory = meanSquaredError, make_optimizer
        model_name = 'Linear(2,1)'
        batch_size = args.batch_size or 4
        epochs = args.epochs or 2
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    loss_rows = []
    for world_size in args.workers:
        for run in range(args.repeats):
            directory = args.output / f'workers-{world_size}-run-{run}'
            config = DistributedConfig(port=_free_port(), world_size=world_size, epochs=epochs,
                                       global_batch_size=batch_size, shuffle=False, timeout_s=args.timeout_s,
                                       metrics_dir=directory)
            start = time.perf_counter()
            result = launch(model_factory, dataset_factory, loss_fn, optimizer_factory, config)
            elapsed = time.perf_counter() - start
            compute, push, pull, step, sent, received, samples, training_s = _metrics(directory, world_size)
            rows.append({'dataset': args.dataset, 'model': model_name,
                         'dataset_size': args.samples if args.dataset != 'toy' else len(make_dataset()),
                         'global_batch_size': batch_size, 'epochs': epochs,
                         'workers': world_size, 'run': run, 'steps': result['model_version'],
                         'job_elapsed_s': elapsed, 'training_elapsed_s': training_s,
                         'samples_processed': samples,
                         'train_samples_per_second': samples / training_s if training_s else 0,
                         'job_samples_per_second': samples / elapsed,
                         'compute_ms_sum': compute, 'push_ms_sum': push, 'pull_ms_sum': pull,
                         'step_ms_sum': step, 'communication_ratio': push / step if step else 0,
                         'bytes_sent': sent, 'bytes_received': received})
            if args.dataset == 'autodrive':
                loss_rows.extend({'workers': world_size, 'run': run, 'step': step,
                                  'global_batch_mse': loss, 'sample_count': count}
                                 for step, loss, count in _loss_trace(directory, world_size))
            print(json.dumps(rows[-1], ensure_ascii=False))
    with (args.output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if loss_rows:
        with (args.output / 'loss_curves.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=loss_rows[0])
            writer.writeheader()
            writer.writerows(loss_rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; results.csv can be plotted separately')
        return
    counts = sorted(set(row['workers'] for row in rows))
    throughput = [sum(row['train_samples_per_second'] for row in rows if row['workers'] == n) /
                  sum(row['workers'] == n for row in rows) for n in counts]
    ratios = [sum(row['communication_ratio'] for row in rows if row['workers'] == n) /
              sum(row['workers'] == n for row in rows) for n in counts]
    for filename, values, ylabel in (
        ('throughput.png', throughput, 'Samples / second'),
        ('communication_ratio.png', ratios, 'Communication + barrier fraction'),
    ):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(counts, values, marker='o')
        axis.set_xticks(counts)
        axis.set_xlabel('Workers')
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output / filename, dpi=160)
        plt.close(figure)

    if args.dataset == 'autodrive':
        import numpy as np

        deviations = [statistics.stdev(row['train_samples_per_second'] for row in rows if row['workers'] == n)
                      if sum(row['workers'] == n for row in rows) > 1 else 0 for n in counts]
        x = np.arange(len(counts))
        figure, axis = plt.subplots(figsize=(8, 5))
        bars = axis.bar(x, throughput, width=0.55, yerr=deviations, capsize=5,
                        color='#3487b8', label='Training throughput')
        axis.set_xticks(x)
        axis.set_xticklabels(counts)
        axis.set_xlabel('Worker processes')
        axis.set_ylabel('Training throughput (images/s)')
        axis.set_ylim(0, max(throughput) * 1.35)
        axis.grid(axis='y', alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, throughput):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(throughput) * 0.035,
                      f'{value:.1f}', ha='center', fontsize=9)
        ratio_axis = axis.twinx()
        ratio_axis.plot(x, [ratio * 100 for ratio in ratios], color='#d36c35',
                        marker='o', linewidth=2.3, label='PUSH + barrier fraction')
        ratio_axis.set_ylabel('PUSH + barrier / Worker step (%)')
        ratio_axis.set_ylim(0, 100)
        axis.legend(loc='upper left')
        ratio_axis.legend(loc='upper right')
        figure.suptitle(f'Autodrive PS: {args.samples} center-camera images, global batch {batch_size}')
        figure.tight_layout()
        figure.savefig(args.output / 'throughput_and_communication.png', dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4.5))
        for n in counts:
            steps = sorted(set(row['step'] for row in loss_rows if row['workers'] == n))
            means = [statistics.mean(row['global_batch_mse'] for row in loss_rows
                                     if row['workers'] == n and row['step'] == step)
                     for step in steps]
            axis.plot(steps, means, marker='.', linewidth=1.5, label=f'{n} worker(s)')
        axis.set_xlabel('Optimizer step')
        axis.set_ylabel('Global-batch training MSE')
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output / 'loss_curves.png', dpi=180)
        plt.close(figure)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
