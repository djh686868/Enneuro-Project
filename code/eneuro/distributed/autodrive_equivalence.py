"""Compare every AutoDriveNet training step: single process versus 2-worker PS."""

import argparse
import csv
import json
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

from ..base import Tensor
from ..nn.loss import meanSquaredError
from ..train.trainer import Trainer
from .autodrive import make_autodrive_model, make_optimizer, make_steering_dataset
from .benchmark import _free_port, _loss_trace
from .launcher import DistributedConfig, launch
from .state import install_gradients, named_state


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / 'doc' / 'M2_AutoDrive_PS结果'


def _single_process(samples, batch_size, dataset_factory):
    dataset = dataset_factory()
    model = make_autodrive_model()
    trainer = Trainer(model, meanSquaredError, optimizer=None)
    optimizer = make_optimizer(model.get_params_list())
    losses = []
    for start in range(0, samples, batch_size):
        batch = [dataset[index] for index in range(start, start + batch_size)]
        images = Tensor.stack([Tensor(image) for image, _ in batch])
        steering = Tensor.stack([Tensor(label) for _, label in batch])
        loss, _, gradients = trainer.compute_batch_gradients(images, steering)
        losses.append(loss)
        install_gradients(model, gradients)
        optimizer.step()
        model.cleargrads()
    return np.asarray(losses), named_state(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--timeout-s', type=float, default=120)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size < 2 or args.samples % args.batch_size:
        parser.error('samples must be positive and divisible by batch-size >= 2')

    dataset_factory = partial(make_steering_dataset, limit=args.samples, **(
        {'data_dir': args.data_dir} if args.data_dir else {}))
    steps = args.samples // args.batch_size
    with tempfile.TemporaryDirectory(prefix='enneuro-autodrive-equivalence-') as directory:
        metrics_dir = Path(directory)
        config = DistributedConfig(port=_free_port(), world_size=2, epochs=1,
                                   global_batch_size=args.batch_size, seed=42,
                                   shuffle=False, timeout_s=args.timeout_s,
                                   metrics_dir=metrics_dir)
        distributed = launch(make_autodrive_model, dataset_factory,
                             meanSquaredError, make_optimizer, config)
        ps_trace = _loss_trace(metrics_dir, 2)
    if distributed['model_version'] != steps or len(ps_trace) != steps:
        raise AssertionError(f'expected {steps} complete PS updates')

    ps_losses = np.asarray([loss for _, loss, _ in ps_trace])
    reference_losses, reference_state = _single_process(args.samples, args.batch_size,
                                                          dataset_factory)
    np.testing.assert_allclose(ps_losses, reference_losses, atol=1e-5, rtol=1e-5)
    if set(distributed['parameters']) != set(reference_state):
        raise AssertionError('parameter names differ after training')
    parameter_differences = []
    for name, expected in reference_state.items():
        actual = distributed['parameters'][name]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5,
                                   err_msg=f'parameter mismatch: {name}')
        parameter_differences.append(float(np.max(np.abs(actual - expected))))

    differences = np.abs(reference_losses - ps_losses)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'equivalence_loss.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'single_process_mse', 'ps_two_worker_mse', 'absolute_difference'])
        for step, (single, ps) in enumerate(zip(reference_losses, ps_losses), start=1):
            writer.writerow([step, single, ps, abs(single - ps)])

    import matplotlib.pyplot as plt

    figure, (loss_axis, difference_axis) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    x = np.arange(1, steps + 1)
    loss_axis.plot(x, reference_losses, label='Single process', linewidth=1.8)
    loss_axis.plot(x, ps_losses, label='PS, 2 workers', linestyle='--', linewidth=1.4)
    loss_axis.set_ylabel('Global-batch training MSE')
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()
    difference_axis.plot(x, differences, color='tab:red')
    difference_axis.set_xlabel('Optimizer step')
    difference_axis.set_ylabel('|Difference|')
    difference_axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output / 'equivalence_loss.png', dpi=180)
    plt.close(figure)

    print(json.dumps({'samples': args.samples, 'global_batch_size': args.batch_size,
                      'steps': steps, 'max_loss_absolute_difference': float(differences.max()),
                      'mean_loss_absolute_difference': float(differences.mean()),
                      'max_parameter_absolute_difference': max(parameter_differences),
                      'single_first_loss': float(reference_losses[0]),
                      'single_last_loss': float(reference_losses[-1]),
                      'csv': str(csv_path),
                      'figure': str(args.output / 'equivalence_loss.png')}, ensure_ascii=False))


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
