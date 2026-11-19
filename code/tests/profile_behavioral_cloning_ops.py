"""Profile CUDA time by EnNeuro autograd operator for behavioral cloning."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import bench_behavioral_cloning_three_way as bench  # noqa: E402
from eneuro.base.core import Function  # noqa: E402
from eneuro.base.cuda import dispatch  # noqa: E402
from eneuro.nn.optim import Adam  # noqa: E402


def descendants(cls):
    result = []
    for child in cls.__subclasses__():
        result.append(child)
        result.extend(descendants(child))
    return result


def install_timers(cp):
    records = defaultdict(list)
    stream = cp.cuda.get_current_stream()

    def wrap(owner, method, label):
        original = owner.__dict__.get(method)
        if original is None:
            return

        def timed(self, *args, **kwargs):
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record(stream)
            result = original(self, *args, **kwargs)
            stop.record(stream)
            records[label].append((start, stop))
            return result

        setattr(owner, method, timed)

    for cls in descendants(Function):
        for method in ("forward", "backward"):
            wrap(cls, method, f"{cls.__name__}.{method}")
    wrap(Adam, "step", "Adam.step")
    return records


def summarize(records, cp):
    rows = []
    for name, events in records.items():
        total_ms = sum(cp.cuda.get_elapsed_time(start, stop) for start, stop in events)
        rows.append((total_ms, len(events), name))
    return sorted(rows, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backend", choices=("cupy", "rawmodule"), default="cupy")
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    import cupy as cp

    rows = bench.read_log(args.data_dir)
    train_rows = rows[: int(len(rows) * 0.7)]
    if args.rows > len(train_rows):
        parser.error(f"only {len(train_rows)} training rows are available")
    rng = np.random.default_rng(args.seed)
    selected_indices = sorted(rng.choice(len(train_rows), size=args.rows, replace=False))
    entries = bench.make_entries([train_rows[i] for i in selected_indices], 0.2)
    order = rng.permutation(len(entries))
    dataset = bench.DrivingDataset(entries, 32, 64, augment=False)
    images = np.stack([dataset[int(i)][0].data for i in order])
    targets = np.stack([dataset[int(i)][1].data for i in order])
    available_steps = len(images) // args.batch_size
    if args.steps > available_steps:
        parser.error(f"sample supports only {available_steps} full batches")

    weights = bench.snapshot_initial_weights(images, args.seed)
    device = "cuda"
    dispatch.set_backend(args.backend)
    model = bench.fresh_model(weights, device)
    optimizer = bench.make_optimizer(model, 1e-4)
    for _ in range(args.warmup):
        bench.train_step(model, optimizer, images[: args.batch_size], targets[: args.batch_size], device)
    cp.cuda.get_current_stream().synchronize()

    records = install_timers(cp)
    dispatch.set_conv_backward_profile(True)
    for step in range(args.steps):
        lo = step * args.batch_size
        hi = lo + args.batch_size
        bench.train_step(model, optimizer, images[lo:hi], targets[lo:hi], device)
    cp.cuda.get_current_stream().synchronize()

    print(f"backend={args.backend} gpu={cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"profiled_steps={args.steps} batch_size={args.batch_size}")
    print("total_gpu_ms calls avg_ms_per_call operator")
    for total_ms, calls, name in summarize(records, cp):
        print(f"{total_ms:12.3f} {calls:5d} {total_ms / calls:15.4f} {name}")
    print("conv2d_backward phase totals (GPU ms)")
    for name, total_ms in sorted(dispatch.get_conv_backward_profile().items(), key=lambda item: item[1], reverse=True):
        print(f"{total_ms:12.3f} {name}")


if __name__ == "__main__":
    main()
