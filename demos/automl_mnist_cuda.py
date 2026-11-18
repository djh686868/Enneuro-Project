"""用 CUDA 训练 EnNeuro LeNet，按墙钟预算搜索 MNIST 超参数。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from math import isfinite
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import SearchSpace
from eneuro.agent.mnist_cuda_executor import CudaMNISTExecutor
from eneuro.agent.search import SEARCH_STRATEGIES
from eneuro.agent.timed_search import run_timed_search
from train_mnist_three_way import load_mnist


def run(args) -> int:
    budget_s = args.budget_minutes * 60
    if not isfinite(budget_s) or budget_s <= 0:
        raise ValueError("--budget-minutes 必须是正数")
    if min(args.train_samples, args.val_samples, args.test_samples, args.epochs) < 1:
        raise ValueError("样本数与 epoch 数必须是正整数")
    if args.trials is not None and args.trials < 1:
        raise ValueError("--trials 必须是正整数")

    x_train, y_train, x_test, y_test = load_mnist(args.data)
    development_samples = args.train_samples + args.val_samples
    if development_samples > len(x_train) or args.test_samples > len(x_test):
        raise ValueError(f"请求的样本数超过文件内容（训练分区 {len(x_train)}，测试分区 {len(x_test)}）")
    split_rng = np.random.default_rng(args.split_seed)
    dev_indices = split_rng.permutation(len(x_train))[:development_samples]
    train_indices = dev_indices[:args.train_samples]
    val_indices = dev_indices[args.train_samples:]
    test_indices = split_rng.permutation(len(x_test))[:args.test_samples]
    executor = CudaMNISTExecutor(
        x_train[train_indices], y_train[train_indices],
        x_train[val_indices], y_train[val_indices],
        x_test[test_indices], y_test[test_indices],
        epochs=args.epochs, seed=args.seed, device_id=args.device_id, backend=args.backend,
    )
    space = SearchSpace(lr=(1e-4, 3e-2), batch_size=(64, 128, 256), optimizer=("adam", "sgd"))
    print(f"设备 {executor.device_name} | CUDA {executor.backend} | EnNeuro LeNet", flush=True)
    print(f"训练/验证/测试 {args.train_samples:,}/{args.val_samples:,}/{args.test_samples:,} | "
          f"每个配置 {args.epochs} epoch", flush=True)
    return run_timed_search(
        executor, space, budget_s=budget_s, trials=args.trials, seed=args.seed, output=args.output,
        strategy=getattr(args, "strategy", "random"),
        protocol={
            "data": str(args.data.resolve()), "model": "EnNeuro LeNet", "device": executor.device_name,
            "device_id": args.device_id, "backend": executor.backend, "epochs": args.epochs,
            "train_samples": args.train_samples, "val_samples": args.val_samples, "test_samples": args.test_samples,
            "split_seed": args.split_seed, "objective": "validation_error_rate",
            "outer_split": "official_mnist_test",
        },
        plot_ylabel="Best validation error (1 − accuracy)",
        format_validation=lambda error: f"准确率 {1.0 - error:.2%}",
        format_epoch_validation=lambda accuracy: f"准确率 {accuracy:.2%}",
        format_test=lambda metrics: f"准确率 {metrics['test_accuracy']:.2%}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="真实 MNIST pickle 文件")
    parser.add_argument("--budget-minutes", type=float, default=20, help="搜索墙钟预算，默认 20 分钟")
    parser.add_argument("--trials", type=int, help="可选的试验数上限")
    parser.add_argument("--strategy", choices=SEARCH_STRATEGIES, default="random", help="搜索策略，默认随机搜索")
    parser.add_argument("--epochs", type=int, default=2, help="每个候选配置训练的 epoch 数")
    parser.add_argument("--train-samples", type=int, default=55000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--backend", choices=("cupy", "rawmodule"), default="cupy")
    default_output = ROOT / "artifacts" / f"automl_mnist_cuda_{datetime.now():%Y%m%d_%H%M%S}"
    parser.add_argument("--output", type=Path, default=default_output, help="本次运行的新输出目录")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
