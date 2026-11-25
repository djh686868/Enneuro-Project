"""Benchmark synchronous training with fixed global batches across worker counts and backends."""

import argparse
import csv
import json
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np

from ..nn.loss import meanSquaredError, crossEntropyError
from ..profiling.distributed import benchmark_metrics, worker_loss_trace
from .demo import make_model, make_dataset, make_optimizer
from .mnist import make_lenet, make_mnist_dataset, make_optimizer as make_mnist_optimizer
from .autodrive import (make_autodrive_model, make_steering_dataset,
                        make_optimizer as make_autodrive_optimizer)
from .launcher import DistributedConfig, launch


def _compare_results(first, second):
    first_losses, first_params = first
    losses, params = second
    np.testing.assert_allclose(losses, first_losses, rtol=1e-5, atol=1e-6, equal_nan=False)
    if params.keys() != first_params.keys():
        raise ValueError('backends returned different parameter names')
    max_param_diff = 0.0
    for name, value in params.items():
        other = first_params[name]
        np.testing.assert_allclose(value, other, rtol=1e-5, atol=1e-6, equal_nan=False, err_msg=name)
        max_param_diff = max(max_param_diff, float(np.max(np.abs(value - other))))
    return {'max_loss_abs_diff': float(np.max(np.abs(losses - first_losses))),
            'max_param_abs_diff': max_param_diff}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backends', '--backend', choices=['ps', 'ring'], nargs='+', default=['ps'],
                        help='one or both training backends; paired runs verify the same updates')
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
    if args.repeats < 1 or len(set(args.backends)) != len(args.backends):
        parser.error('use a positive repeat count and unique backends')
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
    comparisons = []
    dataset_size = args.samples if args.dataset != 'toy' else len(make_dataset())
    expected_steps = epochs * (dataset_size // batch_size)
    if expected_steps < 1:
        parser.error('the benchmark requires at least one full global batch')
    for world_size in args.workers:
        for run in range(args.repeats):
            reference = None
            # Alternate which backend runs first on successive repetitions.
            for backend in (args.backends if run % 2 == 0 else reversed(args.backends)):
                directory = args.output if args.backends == ['ps'] else args.output / backend
                directory = directory / f'workers-{world_size}-run-{run}'
                config = DistributedConfig(backend=backend, port=0, world_size=world_size, epochs=epochs,
                                           global_batch_size=batch_size, shuffle=False, timeout_s=args.timeout_s,
                                           metrics_dir=directory)
                start = time.perf_counter()
                result = launch(model_factory, dataset_factory, loss_fn, optimizer_factory, config)
                elapsed = time.perf_counter() - start
                metrics = benchmark_metrics(directory, world_size)
                samples = metrics['samples_processed']
                trace = worker_loss_trace(directory, world_size)
                if (len(trace) != expected_steps or result['model_version'] != expected_steps
                        or samples != expected_steps * batch_size):
                    raise ValueError('benchmark processed an unexpected number of samples or optimizer steps')
                if len(args.backends) > 1:
                    state = np.asarray([loss for _, loss, _ in trace]), result['parameters']
                    if reference is None:
                        reference = state
                    else:
                        comparisons.append({'workers': world_size, 'run': run, **_compare_results(reference, state)})
                rows.append({'backend': backend, 'dataset': args.dataset, 'model': model_name,
                             'dataset_size': dataset_size, 'global_batch_size': batch_size, 'epochs': epochs,
                             'workers': world_size, 'run': run, 'steps': result['model_version'],
                             'job_elapsed_s': elapsed, 'job_samples_per_second': samples / elapsed, **metrics})
                if args.dataset == 'autodrive':
                    loss_rows.extend({'backend': backend, 'workers': world_size, 'run': run, 'step': step,
                                      'global_batch_mse': loss, 'sample_count': count}
                                     for step, loss, count in trace)
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    with (args.output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if comparisons:
        with (args.output / 'equivalence.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=comparisons[0])
            writer.writeheader()
            writer.writerows(comparisons)
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
    for filename, field, ylabel in (
        ('throughput.png', 'train_samples_per_second', 'Samples / second'),
        ('communication_ratio.png', 'communication_ratio', 'Synchronization fraction of worker step'),
    ):
        figure, axis = plt.subplots(figsize=(6, 4))
        for backend in args.backends:
            values = [statistics.mean(row[field] for row in rows
                                      if row['workers'] == n and row['backend'] == backend) for n in counts]
            axis.plot(counts, values, marker='o', label=backend.upper())
        axis.legend()
        axis.set_xticks(counts)
        axis.set_xlabel('Workers')
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output / filename, dpi=160)
        plt.close(figure)

    if args.dataset == 'autodrive':
        for backend in args.backends:
            output = args.output if len(args.backends) == 1 else args.output / backend
            _plot_autodrive(plt, backend, counts, [r for r in rows if r['backend'] == backend],
                            [r for r in loss_rows if r['backend'] == backend], output)


def _plot_autodrive(plt, backend, counts, rows, loss_rows, output):
    throughput = [statistics.mean(row['train_samples_per_second'] for row in rows if row['workers'] == n)
                  for n in counts]
    ratios = [statistics.mean(row['communication_ratio'] for row in rows if row['workers'] == n)
              for n in counts]
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
                    marker='o', linewidth=2.3, label='Synchronization fraction')
    ratio_axis.set_ylabel('Synchronization / Worker step (%)')
    ratio_axis.set_ylim(0, 100)
    axis.legend(loc='upper left')
    ratio_axis.legend(loc='upper right')
    figure.suptitle(f"Autodrive {backend.upper()}: {rows[0]['dataset_size']} images, global batch {rows[0]['global_batch_size']}")
    figure.tight_layout()
    figure.savefig(output / 'throughput_and_communication.png', dpi=180)
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
    figure.savefig(output / 'loss_curves.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
