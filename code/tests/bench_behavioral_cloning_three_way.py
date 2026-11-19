"""Compare NumPy, CuPy and CUDA C on the same driving-image training steps.

Images are decoded once before timing. Each measured step includes batch input
transfer (for GPU), forward, backward, Adam update and GPU synchronization.
Model weights, image order and targets are identical across backends.
"""

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eneuro.base import Config, Tensor  # noqa: E402
from eneuro.base.cuda import dispatch  # noqa: E402
from eneuro.nn.loss import meanSquaredError  # noqa: E402
from train_behavioral_cloning import (  # noqa: E402
    DrivingDataset, ResNet18Steering, make_entries, make_optimizer, read_log,
)


def snapshot_initial_weights(image_batch, seed):
    np.random.seed(seed)
    model = ResNet18Steering()
    # Initialize the lazy final linear layer on CPU before the snapshot.
    with Config.test_mode(), Config.no_grad():
        model(Tensor(image_batch[:1]))
    params = {}
    model._flatten_params(params)
    return {name: np.array(param.data, copy=True) for name, param in params.items()}


def fresh_model(weights, device):
    model = ResNet18Steering()
    params = {}
    model._flatten_params(params)
    for name, value in weights.items():
        params[name].data = value.copy()
    return model.to(device)


def to_host(value):
    import cupy as cp
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def predict_initial(model, images, device):
    import cupy as cp
    x = images if device == 'cpu' else cp.asarray(images)
    with Config.test_mode(), Config.no_grad():
        result = model(Tensor(x, device=device))
    if device == 'cuda':
        cp.cuda.Stream.null.synchronize()
    return to_host(result.data).copy()


def first_batch_gradients(model, images, targets, device):
    import cupy as cp
    if device == 'cuda':
        images, targets = cp.asarray(images), cp.asarray(targets)
    with Config.using_config('train', True):
        prediction = model(Tensor(images, device=device))
        loss = meanSquaredError(prediction, Tensor(targets, device=device))
        model.cleargrads()
        loss.backward()
    if device == 'cuda':
        cp.cuda.Stream.null.synchronize()
    params = {}
    model._flatten_params(params)
    gradients = {
        name: to_host(param.grad.data).copy()
        for name, param in params.items() if param.grad is not None
    }
    return float(loss.data), gradients


def compare_gradients(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError('backend gradients have different parameter names')
    diff_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    for name, baseline in reference.items():
        if baseline.shape != candidate[name].shape:
            raise ValueError(f'gradient shape differs for {name}')
        baseline = baseline.astype(np.float64)
        difference = candidate[name].astype(np.float64) - baseline
        diff_sq += float(np.sum(difference * difference))
        reference_sq += float(np.sum(baseline * baseline))
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
    return float(np.sqrt(diff_sq / reference_sq)), max_abs


def train_step(model, optimizer, images, targets, device):
    import cupy as cp
    if device == 'cuda':
        images = cp.asarray(images)
        targets = cp.asarray(targets)
    xb = Tensor(images, device=device)
    yb = Tensor(targets, device=device)
    with Config.using_config('train', True):
        output = model(xb)
        loss = meanSquaredError(output, yb)
        model.cleargrads()
        loss.backward()
        optimizer.step()
    if device == 'cuda':
        cp.cuda.Stream.null.synchronize()
    return float(loss.data)


def plot_results(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = {'numpy': 'NumPy CPU', 'cupy': 'CuPy GPU', 'cuda_c': 'CUDA C mode'}
    colors = {'numpy': '#7f8c8d', 'cupy': '#3498db', 'cuda_c': '#e67e22'}
    names = [key for key in ('numpy', 'cupy', 'cuda_c') if results[key]['status'] == 'ok']
    means = [results[key]['mean_batch_ms'] for key in names]
    stds = [results[key]['repeat_std_ms'] for key in names]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar([labels[key] for key in names], means, yerr=stds, capsize=5,
                  color=[colors[key] for key in names])
    ax.set_ylabel('Training batch time (ms)')
    ax.set_title('Behavioral cloning ResNet18: three backend benchmark')
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(value + spread for value, spread in zip(means, stds)) * 1.18)
    for bar, value, spread in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + spread + max(means) * 0.015,
                f'{value:.1f} ms', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=64,
                        help='sampled CSV training rows; each contributes three camera images')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backends', nargs='+', choices=('numpy', 'cupy', 'cuda_c'),
                        default=('numpy', 'cupy', 'cuda_c'))
    args = parser.parse_args()
    if tuple(args.backends) != ('numpy', 'cupy', 'cuda_c'):
        parser.error('three-way benchmark requires --backends numpy cupy cuda_c in this order')
    if min(args.rows, args.batch_size, args.steps, args.repeats, args.height, args.width) < 1:
        parser.error('rows, batch size, steps, repeats and dimensions must be positive')
    if args.warmup < 0:
        parser.error('warmup cannot be negative')

    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('No CUDA device is available')
    rows = read_log(args.data_dir)
    train_rows = rows[:int(len(rows) * 0.7)]
    if args.rows > len(train_rows):
        parser.error(f'only {len(train_rows)} training rows are available')
    rng = np.random.default_rng(args.seed)
    selected_indices = sorted(rng.choice(len(train_rows), size=args.rows, replace=False))
    selected = [train_rows[index] for index in selected_indices]
    entries = make_entries(selected, 0.2)
    order = rng.permutation(len(entries))
    data = DrivingDataset(entries, args.height, args.width, augment=False)
    images = np.stack([data[int(index)][0].data for index in order])
    targets = np.stack([data[int(index)][1].data for index in order])
    available_steps = len(images) // args.batch_size
    if args.steps > available_steps:
        parser.error(f'{len(images)} images support only {available_steps} full batches')
    weights = snapshot_initial_weights(images, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference_output = None
    reference_gradients = None
    results = {}
    detail_rows = []
    for backend in args.backends:
        device = 'cpu' if backend == 'numpy' else 'cuda'
        dispatch.set_backend('cupy' if backend == 'numpy' else
                             'rawmodule' if backend == 'cuda_c' else 'cupy')
        print(f'[BACKEND] {backend} device={device}', flush=True)
        try:
            model = fresh_model(weights, device)
            first_output = predict_initial(model, images[:args.batch_size], device)
            if reference_output is None:
                reference_output = first_output
                initial_max_abs_error = 0.0
            else:
                initial_max_abs_error = float(np.max(np.abs(first_output - reference_output)))
            first_loss, gradients = first_batch_gradients(
                model, images[:args.batch_size], targets[:args.batch_size], device,
            )
            if reference_gradients is None:
                reference_gradients = gradients
                relative_gradient_l2, max_gradient_abs_error = 0.0, 0.0
            else:
                relative_gradient_l2, max_gradient_abs_error = compare_gradients(
                    reference_gradients, gradients,
                )
            del gradients
            # Warm up kernels and allocator on a disposable model. Timed models
            # always start from the same original weights and fresh Adam state.
            warm_model = fresh_model(weights, device)
            warm_optimizer = make_optimizer(warm_model, args.lr)
            for _ in range(args.warmup):
                train_step(warm_model, warm_optimizer,
                           images[:args.batch_size], targets[:args.batch_size], device)
            del warm_model, warm_optimizer
            if device == 'cuda':
                cp.cuda.Stream.null.synchronize()
            if backend == 'cuda_c':
                dispatch.launch_counts(reset=True)

            repeat_means = []
            repeat_first_losses = []
            repeat_last_losses = []
            for repeat in range(args.repeats):
                model = fresh_model(weights, device)
                optimizer = make_optimizer(model, args.lr)
                samples = []
                losses = []
                for step in range(args.steps):
                    lo = step * args.batch_size
                    hi = lo + args.batch_size
                    started = time.perf_counter()
                    loss = train_step(model, optimizer, images[lo:hi], targets[lo:hi], device)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    samples.append(elapsed_ms)
                    losses.append(loss)
                    detail_rows.append({'backend': backend, 'repeat': repeat + 1,
                                        'step': step + 1, 'elapsed_ms': elapsed_ms,
                                        'train_mse': loss})
                mean_ms = float(np.mean(samples))
                repeat_means.append(mean_ms)
                repeat_first_losses.append(losses[0])
                repeat_last_losses.append(losses[-1])
                print(f'[REPEAT] {backend} {repeat + 1}/{args.repeats} '
                      f'mean_batch_ms={mean_ms:.3f} first_mse={losses[0]:.6f} '
                      f'last_mse={losses[-1]:.6f}', flush=True)
            results[backend] = {
                'status': 'ok', 'device': device,
                'mean_batch_ms': float(np.mean(repeat_means)),
                'repeat_std_ms': float(np.std(repeat_means)),
                'repeat_mean_batch_ms': repeat_means,
                'images_per_second': float(args.batch_size * 1000.0 / np.mean(repeat_means)),
                'initial_max_abs_error_vs_numpy': initial_max_abs_error,
                'first_batch_mse_for_gradient_check': first_loss,
                'relative_gradient_l2_vs_numpy': relative_gradient_l2,
                'max_gradient_abs_error_vs_numpy': max_gradient_abs_error,
                'first_batch_mse': float(np.mean(repeat_first_losses)),
                'last_batch_mse': float(np.mean(repeat_last_losses)),
            }
            if backend == 'cuda_c':
                results[backend]['native_kernel_launches'] = dispatch.launch_counts()
        except Exception as exc:
            results[backend] = {'status': 'error', 'error': repr(exc)}
            print(f'[ERROR] {backend}: {exc!r}', flush=True)

    cpu_ms = results.get('numpy', {}).get('mean_batch_ms')
    cupy_ms = results.get('cupy', {}).get('mean_batch_ms')
    raw_ms = results.get('cuda_c', {}).get('mean_batch_ms')
    if cpu_ms:
        for backend in ('cupy', 'cuda_c'):
            if results.get(backend, {}).get('status') == 'ok':
                results[backend]['speedup_vs_numpy'] = cpu_ms / results[backend]['mean_batch_ms']
    if cupy_ms and raw_ms:
        results['cuda_c']['speedup_vs_cupy'] = cupy_ms / raw_ms

    report = {
        'schema_version': 1,
        'dataset': str(args.data_dir.resolve() / 'driving_log.csv'),
        'task': 'ResNet18 steering regression: forward + backward + Adam',
        'timing': 'wall time per batch, including CPU-to-GPU input transfer and CUDA synchronization; image decoding excluded',
        'input_selection': 'seeded sample of chronological train split; all three cameras; no random flip',
        'rows': args.rows, 'images': len(images), 'batch_size': args.batch_size,
        'steps_per_repeat': args.steps, 'repeats': args.repeats, 'warmup_steps': args.warmup,
        'image_shape_chw': [3, args.height, args.width], 'seed': args.seed,
        'python': platform.python_version(), 'numpy': np.__version__,
        'cupy': cp.__version__, 'gpu': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'cuda_c_note': 'rawmodule mode may use CuPy fallback for large operators; native_kernel_launches shows actual custom kernels',
        'results': results,
    }
    report['numerical_check_passed'] = all(
        value['status'] == 'ok'
        and value['initial_max_abs_error_vs_numpy'] <= 1e-4
        and value['relative_gradient_l2_vs_numpy'] <= 1e-3
        for value in results.values()
    )
    report_path = args.out_dir / 'benchmark.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (args.out_dir / 'steps.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('backend', 'repeat', 'step', 'elapsed_ms', 'train_mse'))
        writer.writeheader()
        writer.writerows(detail_rows)
    if all(results.get(key, {}).get('status') == 'ok' for key in args.backends):
        plot_results(results, args.out_dir / 'benchmark.png')
    print('[REPORT] ' + str(report_path), flush=True)
    return 0 if report['numerical_check_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
