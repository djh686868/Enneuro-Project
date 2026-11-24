"""Measure CPU loading throughput for DonkeyCar and Car Behavioral Cloning images.

Each worker setting reads the same samples in order with a warm filesystem cache.
"""
import argparse
import statistics
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))

from car_data import load_steering_dataset
from eneuro.data import DataLoader


def measure(loader):
    samples = batches = 0
    first = 0.0
    start = time.perf_counter()
    for images, _ in loader:
        if batches == 0:
            first = time.perf_counter() - start
        batches += 1
        samples += len(images)
    return first, time.perf_counter() - start, (samples, batches)


def verify_batches(loader):
    samples = batches = checksum = 0
    for images, labels in loader:
        batches += 1
        samples += len(images)
        checksum = zlib.crc32(images.data.tobytes(), checksum)
        checksum = zlib.crc32(labels.data.tobytes(), checksum)
    return samples, batches, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--samples", type=int, default=1024,
                        help="maximum samples, 0 uses the complete dataset")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()
    if min(args.workers) < 0 or args.samples < 0:
        parser.error("workers and samples must be non-negative")
    if min(args.rounds, args.batch_size, args.prefetch_factor) < 1:
        parser.error("rounds, batch size and prefetch factor must be positive")

    try:
        dataset = load_steering_dataset(args.data_dir, max_samples=args.samples)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if len(dataset) == 0:
        parser.error("no samples in the dataset")
    loader_options = dict(batch_size=args.batch_size, prefetch_factor=args.prefetch_factor)
    loaders = {n: DataLoader(dataset, num_workers=n, **loader_options) for n in args.workers}
    reference = loaders[0] if 0 in loaders else DataLoader(dataset, **loader_options)
    expected_count = (len(dataset), len(reference))
    expected = verify_batches(reference)
    if expected[:2] != expected_count:
        raise RuntimeError(f"incomplete reference pass: {expected[:2]}, expected {expected_count}")
    # Verify the full ordered images and labels and warm each configuration before timing.
    for n, loader in loaders.items():
        if n > 0 and verify_batches(loader) != expected:
            raise RuntimeError(f"workers={n} did not match synchronous batches")

    runs = {n: [] for n in loaders}
    workers = list(loaders)
    for i in range(args.rounds):
        offset = i % len(workers)
        for n in workers[offset:] + workers[:offset]:
            first, elapsed, observed = measure(loaders[n])
            if observed != expected_count:
                raise RuntimeError(f"workers={n} yielded {observed}, expected {expected_count}")
            runs[n].append((first, elapsed))

    print(f"{expected[0]} samples, {expected[1]} batches, batch size {args.batch_size}, "
          f"prefetch factor {args.prefetch_factor}, checksum {expected[2]:08x}")
    print(f"Warm filesystem cache, median of {args.rounds} timed rounds")
    print("workers   first batch (s)   epoch (s)   samples/s   vs sync")
    medians = {n: tuple(map(statistics.median, zip(*times))) for n, times in runs.items()}
    for n, (first, epoch) in medians.items():
        speedup = f"{medians[0][1] / epoch:.2f}x" if 0 in medians else "-"
        print(f"{n:7d} {first:17.5f} {epoch:11.5f} {len(dataset) / epoch:11.1f} {speedup:>9s}")


if __name__ == "__main__":
    main()
