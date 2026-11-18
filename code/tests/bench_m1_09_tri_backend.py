# -*- coding: utf-8 -*-
"""M1-09 three-backend benchmark for EnNeuro Conv2D and max pooling.

The benchmark compares the same public operator entry points on NumPy, CuPy,
and the CUDA C RawModule route added for M1-11.  GPU timings use CUDA events;
NumPy timings use ``perf_counter``.  Correctness is checked against an
independent float64 NumPy implementation.

CUDA C rows record the effective backend and kernel launch counts.  A CuPy
fallback is never labelled as native CUDA C performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


CSV_FIELDS = [
    'case_id', 'op', 'shape', 'kernel', 'stride', 'pad', 'dilation',
    'backend', 'effective_backend', 'status', 'timing_source',
    'avg_ms', 'std_ms', 'p50_ms', 'min_ms', 'max_ms',
    'iters', 'warmup', 'output_shape', 'speedup_vs_numpy',
    'correct', 'max_abs_error', 'max_rel_error', 'native_cuda_c',
    'kernel_launches', 'fallback_reason', 'note',
]


def _default_shapes():
    # The largest default Conv2D output contains 4096 elements, which remains
    # inside the currently verified native CUDA C range instead of silently
    # taking the large-map CuPy route.
    return [(1, 3, 8, 8), (1, 3, 16, 16), (2, 4, 16, 16)]


def _default_kernels():
    return [(4, 3, 3), (8, 3, 3)]


def _parse_tuple_list(text):
    return [
        tuple(int(value.strip()) for value in item.split(','))
        for item in text.split(';') if item.strip()
    ]


def _pair(values):
    return tuple(int(value) for value in values)


def _validate_args(args):
    if args.iters <= 0 or args.warmup < 0:
        raise ValueError('--iters must be positive and --warmup non-negative')
    for shape in args.shapes or _default_shapes():
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise ValueError(f'invalid NCHW shape: {shape}')
    for kernel in args.kernels or _default_kernels():
        if len(kernel) != 3 or any(value <= 0 for value in kernel):
            raise ValueError(f'invalid OIHW kernel descriptor: {kernel}')


def _conv_reference(x, weight, bias, stride, pad, dilation):
    """Independent float64 NCHW cross-correlation reference."""
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    bias = None if bias is None else np.asarray(bias, dtype=np.float64)
    sh, sw = stride
    ph, pw = pad
    dh, dw = dilation
    n, _, height, width = x.shape
    out_channels, _, kh, kw = weight.shape
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError('Conv2D output shape is empty')
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    output = np.zeros((n, out_channels, oh, ow), dtype=np.float64)
    for kernel_h in range(kh):
        h_slice = slice(kernel_h * dh, kernel_h * dh + sh * oh, sh)
        for kernel_w in range(kw):
            w_slice = slice(kernel_w * dw, kernel_w * dw + sw * ow, sw)
            patch = padded[:, :, h_slice, w_slice]
            output += np.einsum(
                'nchw,oc->nohw', patch, weight[:, :, kernel_h, kernel_w],
                optimize=True,
            )
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)
    return output


def _pool_reference(x, kernel, stride, pad):
    """Independent float64 reference matching EnNeuro's zero padding."""
    x = np.asarray(x, dtype=np.float64)
    kh, kw = kernel
    sh, sw = stride
    ph, pw = pad
    n, channels, height, width = x.shape
    oh = (height + 2 * ph - kh) // sh + 1
    ow = (width + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError('Pooling output shape is empty')
    padded = np.pad(
        x, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
        mode='constant', constant_values=0,
    )
    output = np.empty((n, channels, oh, ow), dtype=np.float64)
    for out_h in range(oh):
        for out_w in range(ow):
            window = padded[
                :, :, out_h * sh:out_h * sh + kh,
                out_w * sw:out_w * sw + kw,
            ]
            output[:, :, out_h, out_w] = window.max(axis=(2, 3))
    return output


def _to_numpy(value):
    if hasattr(value, 'data') and not isinstance(value, np.ndarray):
        value = value.data
    try:
        import cupy as cp
        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except ImportError:
        pass
    return np.asarray(value)


def _correctness(actual, reference, atol, rtol):
    actual_np = _to_numpy(actual).astype(np.float64, copy=False)
    absolute = np.abs(actual_np - reference)
    scale = np.maximum(np.abs(reference), np.finfo(np.float32).eps)
    return (
        bool(np.allclose(actual_np, reference, atol=atol, rtol=rtol)),
        float(absolute.max(initial=0.0)),
        float((absolute / scale).max(initial=0.0)),
    )


def _prepare_arrays(backend, arrays):
    from eneuro.base import as_Tensor

    if backend == 'numpy':
        return tuple(as_Tensor(np.asarray(value)) for value in arrays)
    import cupy as cp
    return tuple(as_Tensor(cp.asarray(value)) for value in arrays)


def _invoke_conv(values, backend, stride, pad, dilation, track=False):
    from eneuro.base import Config
    from eneuro.base.functions import conv2d

    x, weight, bias = values
    with Config.test_mode():
        if track:
            return conv2d(
                x, weight, bias, stride=stride, pad=pad,
                dilation=dilation, backend=backend,
            )
        with Config.no_grad():
            return conv2d(
                x, weight, bias, stride=stride, pad=pad,
                dilation=dilation, backend=backend,
            )


def _invoke_pool(values, backend, kernel, stride, pad, track=False):
    from eneuro.base import Config
    from eneuro.base.functions import pooling

    (x,) = values
    with Config.test_mode():
        if track:
            return pooling(
                x, kernel_size=kernel, stride=stride, pad=pad,
                backend=backend,
            )
        with Config.no_grad():
            return pooling(
                x, kernel_size=kernel, stride=stride, pad=pad,
                backend=backend,
            )


def _execution_evidence(output, backend):
    if backend != 'cuda_c':
        return backend, backend == 'cuda_c', {}, ''
    creator = getattr(output, 'creator', None)
    effective = getattr(creator, '_backend_effective', 'unknown')
    launches = getattr(creator, '_backend_launch_counts', {}) or {}
    fallback = getattr(creator, '_backend_fallback_reason', None) or ''
    native = effective == 'cuda_c' and bool(launches)
    return effective, native, launches, fallback


def _time_case(function, backend, warmup, iters):
    for _ in range(warmup):
        function()

    if backend == 'numpy':
        samples = []
        for _ in range(iters):
            started = time.perf_counter()
            function()
            samples.append((time.perf_counter() - started) * 1000.0)
        return np.asarray(samples, dtype=np.float64), 'perf_counter'

    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    samples = []
    for _ in range(iters):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))
    return np.asarray(samples, dtype=np.float64), 'cuda_event'


def _empty_row(case, backend, args):
    return {
        **case,
        'backend': backend,
        'effective_backend': '',
        'status': 'pending',
        'timing_source': '',
        'avg_ms': None,
        'std_ms': None,
        'p50_ms': None,
        'min_ms': None,
        'max_ms': None,
        'iters': args.iters,
        'warmup': args.warmup,
        'output_shape': '',
        'speedup_vs_numpy': None,
        'correct': None,
        'max_abs_error': None,
        'max_rel_error': None,
        'native_cuda_c': None,
        'kernel_launches': '',
        'fallback_reason': '',
        'note': '',
    }


def _run_backend(case, backend, arrays, reference, function_factory,
                 args, capabilities):
    row = _empty_row(case, backend, args)
    capability = capabilities[backend]
    if not capability['available']:
        row.update(status='unavailable', note=capability['reason'])
        print(f"  {backend:<7} unavailable: {capability['reason']}")
        return row

    try:
        prepared = _prepare_arrays(backend, arrays)
        function = lambda: function_factory(prepared, backend, False)
        actual = function_factory(prepared, backend, True)
        correct, max_abs, max_rel = _correctness(
            actual, reference, args.atol, args.rtol,
        )
        effective, native, launches, fallback = _execution_evidence(
            actual, backend,
        )
        samples, timing_source = _time_case(
            function, backend, args.warmup, args.iters,
        )
        if not correct:
            status = 'incorrect'
        elif backend == 'cuda_c' and effective == 'cuda_c_mixed':
            status = 'mixed'
        elif backend == 'cuda_c' and not native:
            status = 'fallback'
        else:
            status = 'ok'
        row.update({
            'effective_backend': effective,
            'status': status,
            'timing_source': timing_source,
            'avg_ms': float(samples.mean()),
            'std_ms': float(samples.std()),
            'p50_ms': float(np.median(samples)),
            'min_ms': float(samples.min()),
            'max_ms': float(samples.max()),
            'output_shape': str(tuple(actual.shape)),
            'correct': correct,
            'max_abs_error': max_abs,
            'max_rel_error': max_rel,
            'native_cuda_c': native if backend == 'cuda_c' else None,
            'kernel_launches': json.dumps(launches, sort_keys=True),
            'fallback_reason': fallback,
            'note': '' if correct else 'output differs from float64 reference',
        })
        print(
            f"  {backend:<7} {status:<9} effective={effective:<15} "
            f"avg={row['avg_ms']:.4f} ms max_abs={max_abs:.3e}"
        )
    except Exception as exc:
        row.update(
            status='error',
            note=f'{type(exc).__name__}: {exc}',
        )
        print(f'  {backend:<7} error: {exc!r}')
    return row


def _conv_rows(args, capabilities):
    rows = []
    shapes = args.shapes or _default_shapes()
    kernels = args.kernels or _default_kernels()
    stride, pad, dilation = map(
        _pair, (args.stride, args.pad, args.dilation)
    )
    total = len(shapes) * len(kernels)
    index = 0
    for shape in shapes:
        n, channels, height, width = shape
        for descriptor in kernels:
            index += 1
            out_channels, kh, kw = descriptor
            rng = np.random.default_rng(args.seed + index)
            x = rng.normal(size=shape).astype(np.float32)
            weight = rng.normal(
                size=(out_channels, channels, kh, kw)
            ).astype(np.float32)
            bias = rng.normal(size=(out_channels,)).astype(np.float32)
            case_id = (
                f'conv_n{n}_c{channels}_h{height}_w{width}_'
                f'oc{out_channels}_k{kh}x{kw}'
            )
            case = {
                'case_id': case_id,
                'op': 'conv2d',
                'shape': str(shape),
                'kernel': str(descriptor),
                'stride': str(stride),
                'pad': str(pad),
                'dilation': str(dilation),
            }
            print(f'[{index}/{total}] {case_id}')
            reference = _conv_reference(
                x, weight, bias, stride, pad, dilation,
            )
            factory = lambda values, backend, track: _invoke_conv(
                values, backend, stride, pad, dilation, track=track,
            )
            for backend in args.backends:
                rows.append(_run_backend(
                    case, backend, (x, weight, bias), reference,
                    factory, args, capabilities,
                ))
    return rows


def _pool_rows(args, capabilities):
    rows = []
    shapes = args.shapes or _default_shapes()
    kernel, stride, pad = map(
        _pair, (args.pool_kernel, args.pool_stride, args.pool_pad)
    )
    for index, shape in enumerate(shapes, start=1):
        n, channels, height, width = shape
        rng = np.random.default_rng(args.seed + 1000 + index)
        x = rng.normal(size=shape).astype(np.float32)
        case_id = (
            f'pool_n{n}_c{channels}_h{height}_w{width}_'
            f'k{kernel[0]}x{kernel[1]}'
        )
        case = {
            'case_id': case_id,
            'op': 'pooling',
            'shape': str(shape),
            'kernel': str(kernel),
            'stride': str(stride),
            'pad': str(pad),
            'dilation': '',
        }
        print(f'[{index}/{len(shapes)}] {case_id}')
        reference = _pool_reference(x, kernel, stride, pad)
        factory = lambda values, backend, track: _invoke_pool(
            values, backend, kernel, stride, pad, track=track,
        )
        for backend in args.backends:
            rows.append(_run_backend(
                case, backend, (x,), reference,
                factory, args, capabilities,
            ))
    return rows


def _add_speedups(rows):
    baselines = {
        row['case_id']: row['avg_ms'] for row in rows
        if row['backend'] == 'numpy' and row['status'] == 'ok'
    }
    for row in rows:
        baseline = baselines.get(row['case_id'])
        if row['status'] == 'ok' and baseline and row['avg_ms']:
            row['speedup_vs_numpy'] = baseline / row['avg_ms']


def _write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'[csv] {path}')


def _environment_metadata(args, capabilities):
    metadata = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'capabilities': capabilities,
        'command_args': vars(args),
        'cuda_c_route': 'CuPy RawModule compiling sources/kernels.cu with NVRTC',
        'extension_todo': (
            'nvcc DLL sources exist, but the Python ctypes/extension loader '
            'is not implemented'
        ),
    }
    if capabilities['cupy']['available']:
        import cupy as cp
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties['name']
        if isinstance(name, bytes):
            name = name.decode(errors='replace')
        metadata.update({
            'cupy': cp.__version__,
            'gpu': name,
            'compute_capability': cp.cuda.Device().compute_capability,
            'cuda_runtime': int(cp.cuda.runtime.runtimeGetVersion()),
            'cuda_driver': int(cp.cuda.runtime.driverGetVersion()),
        })
    return metadata


def _write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )
    print(f'[metadata] {path}')


def _plot(rows, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plot] matplotlib is unavailable; skipped')
        return False

    operations = list(dict.fromkeys(row['op'] for row in rows))
    fig, axes = plt.subplots(
        1, len(operations),
        figsize=(max(7, 7 * len(operations)), 5), squeeze=False,
    )
    for axis, operation in zip(axes[0], operations):
        operation_rows = [row for row in rows if row['op'] == operation]
        case_ids = list(dict.fromkeys(row['case_id'] for row in operation_rows))
        for backend, color in (('cupy', '#3498db'), ('cuda_c', '#e67e22')):
            values = []
            for case_id in case_ids:
                match = next((
                    row for row in operation_rows
                    if row['case_id'] == case_id and row['backend'] == backend
                ), None)
                values.append(
                    np.nan if not match or match['speedup_vs_numpy'] is None
                    else match['speedup_vs_numpy']
                )
            if np.isfinite(values).any():
                axis.plot(
                    range(len(case_ids)), values, marker='o',
                    label=backend, color=color,
                )
        axis.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        axis.set_xticks(range(len(case_ids)))
        axis.set_xticklabels(case_ids, rotation=35, ha='right', fontsize=7)
        axis.set_ylabel('speedup vs NumPy')
        axis.set_title(f'{operation} backend speedup')
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[plot] {path}')
    return True


def _parser():
    parser = argparse.ArgumentParser(description='M1-09 tri-backend benchmark')
    parser.add_argument('--ops', nargs='+', choices=['conv', 'pool'],
                        default=['conv', 'pool'])
    parser.add_argument('--backends', nargs='+',
                        choices=['numpy', 'cupy', 'cuda_c'],
                        default=['numpy', 'cupy', 'cuda_c'])
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--stride', type=int, nargs=2, default=[1, 1])
    parser.add_argument('--pad', type=int, nargs=2, default=[1, 1])
    parser.add_argument('--dilation', type=int, nargs=2, default=[1, 1])
    parser.add_argument('--shapes', type=_parse_tuple_list, default=None)
    parser.add_argument('--kernels', type=_parse_tuple_list, default=None)
    parser.add_argument('--pool-kernel', type=int, nargs=2, default=[2, 2])
    parser.add_argument('--pool-stride', type=int, nargs=2, default=[2, 2])
    parser.add_argument('--pool-pad', type=int, nargs=2, default=[0, 0])
    parser.add_argument('--out-csv', default=None)
    parser.add_argument('--out-plot', default=None)
    parser.add_argument('--out-meta', default=None)
    parser.add_argument('--fail-on-error', action='store_true')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    _validate_args(args)
    from eneuro.base.functions import get_backend_capabilities

    capabilities = get_backend_capabilities()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    stem = THIS_DIR / f'bench_m1_09_{timestamp}'
    out_csv = Path(args.out_csv) if args.out_csv else stem.with_suffix('.csv')
    out_plot = Path(args.out_plot) if args.out_plot else stem.with_suffix('.png')
    out_meta = (
        Path(args.out_meta) if args.out_meta
        else Path(f'{stem}_environment.json')
    )

    print('==== backend capabilities ====')
    for backend in args.backends:
        info = capabilities[backend]
        print(
            f"{backend:<7} implemented={info['implemented']} "
            f"available={info['available']} {info['reason']}"
        )

    rows = []
    if 'conv' in args.ops:
        rows.extend(_conv_rows(args, capabilities))
    if 'pool' in args.ops:
        rows.extend(_pool_rows(args, capabilities))
    _add_speedups(rows)
    _write_csv(rows, out_csv)
    _write_json(_environment_metadata(args, capabilities), out_meta)
    _plot(rows, out_plot)

    statuses = {
        status: sum(row['status'] == status for row in rows)
        for status in (
            'ok', 'mixed', 'fallback', 'incorrect', 'unavailable', 'error'
        )
    }
    print(f'==== completed: {len(rows)} rows, {statuses} ====')
    failed = statuses['incorrect'] or statuses['error']
    if 'cuda_c' in args.backends:
        failed = failed or statuses['mixed'] or statuses['fallback']
    return 1 if args.fail_on_error and failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
