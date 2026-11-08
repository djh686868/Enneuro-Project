"""Compare a complete synchronous PS loss trace with single-process MNIST training."""

import argparse
import csv
import json
import socket
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

from ..base import Tensor
from ..nn.loss import crossEntropyError
from ..train.trainer import Trainer
from .launcher import DistributedConfig, launch
from .mnist import make_lenet, make_mnist_dataset, make_optimizer
from .state import install_gradients, named_state


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / 'doc' / 'M2_MNIST_PS结果'


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _single_process_losses(samples, batch_size):
    model = make_lenet()
    dataset = make_mnist_dataset(limit=samples)
    trainer = Trainer(model, crossEntropyError, optimizer=None)
    optimizer = make_optimizer(model.get_params_list())
    losses = []
    for start in range(0, samples, batch_size):
        indices = range(start, start + batch_size)
        xb = Tensor.stack([Tensor(np.asarray(dataset[i][0])) for i in indices])
        yb = Tensor.stack([Tensor(np.asarray(dataset[i][1])) for i in indices])
        loss, _, gradients = trainer.compute_batch_gradients(xb, yb)
        losses.append(loss)
        install_gradients(model, gradients)
        optimizer.step()
        model.cleargrads()
    return np.asarray(losses), named_state(model)


def _ps_losses(metrics_dir, steps, workers):
    losses = [[] for _ in range(steps)]
    for rank in range(workers):
        with (metrics_dir / f'worker-{rank}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if record['event'] == 'compute':
                    losses[record['step']].append((record['loss'], record['sample_count']))
    if any(len(step_losses) != workers for step_losses in losses):
        raise AssertionError('missing or duplicated Worker loss in a PS step')
    return np.asarray([
        sum(loss * count for loss, count in step_losses) / sum(count for _, count in step_losses)
        for step_losses in losses
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--timeout-s', type=float, default=120)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.samples <= 0 or args.samples > 60000 or args.batch_size < 2 or args.samples % args.batch_size:
        parser.error('samples must be 1..60000 and divisible by batch-size >= 2')

    steps = args.samples // args.batch_size
    with tempfile.TemporaryDirectory(prefix='enneuro-mnist-equivalence-') as directory:
        metrics_dir = Path(directory)
        config = DistributedConfig(port=_free_port(), world_size=2, epochs=1,
                                   global_batch_size=args.batch_size, seed=42,
                                   shuffle=False, timeout_s=args.timeout_s,
                                   metrics_dir=metrics_dir)
        distributed = launch(make_lenet, partial(make_mnist_dataset, limit=args.samples),
                             crossEntropyError, make_optimizer, config)
        ps_losses = _ps_losses(metrics_dir, steps, workers=2)

    reference_losses, reference_state = _single_process_losses(args.samples, args.batch_size)
    if distributed['model_version'] != steps:
        raise AssertionError(f"expected {steps} PS updates, got {distributed['model_version']}")
    np.testing.assert_allclose(ps_losses, reference_losses, atol=1e-5, rtol=1e-5)
    parameter_differences = {}
    for name, expected in reference_state.items():
        actual = distributed['parameters'][name]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5,
                                   err_msg=f'parameter mismatch: {name}')
        parameter_differences[name] = float(np.max(np.abs(actual - expected)))

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'equivalence_loss.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'single_process_loss', 'ps_two_worker_loss', 'absolute_difference'])
        for step, (single, ps) in enumerate(zip(reference_losses, ps_losses), start=1):
            writer.writerow([step, single, ps, abs(single - ps)])

    import matplotlib.pyplot as plt

    figure, (loss_axis, difference_axis) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    x = np.arange(1, steps + 1)
    loss_axis.plot(x, reference_losses, label='Single process', linewidth=1.8)
    loss_axis.plot(x, ps_losses, label='PS, 2 workers', linestyle='--', linewidth=1.4)
    loss_axis.set_ylabel('Global batch loss')
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()
    difference_axis.plot(x, np.abs(reference_losses - ps_losses), color='tab:red')
    difference_axis.set_xlabel('Optimizer step')
    difference_axis.set_ylabel('|Difference|')
    difference_axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output / 'equivalence_loss.png', dpi=160)
    plt.close(figure)

    print(json.dumps({'samples': args.samples, 'global_batch_size': args.batch_size,
                      'steps': steps, 'max_loss_absolute_difference': float(np.max(np.abs(reference_losses - ps_losses))),
                      'max_parameter_absolute_difference': max(parameter_differences.values()),
                      'single_first_loss': float(reference_losses[0]),
                      'single_last_loss': float(reference_losses[-1]),
                      'csv': str(csv_path), 'figure': str(args.output / 'equivalence_loss.png')},
                     ensure_ascii=False))


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
