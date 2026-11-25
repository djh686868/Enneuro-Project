"""运行一个可复现的 MNIST 小规模自动调参演示。

脚本读取真实的 `mnist.pkl`，也支持使用内置的合成 MNIST 形状数据。
每个 trial 用训练来源的数据做内层验证，选定配置后再用外层测试集评分。
选定的搜索策略和固定超参数基线使用相同的数据划分和训练配方。

示例：
`python demos/automl_mnist_agent.py --trials 4 --output artifacts/automl`
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import MNISTExecutor, SearchSpace, make_tiny_mnist
from eneuro.agent._profiling import Profiling, add_profiling_arguments, profiling_from_args
from eneuro.agent.evaluation import run_evaluation
from eneuro.agent.search import SEARCH_STRATEGIES


def _load_mnist(path: str | Path):
    """兼容项目中常见的三种 MNIST pickle 文件结构。

    既支持字典格式（train_img/train_label/test_img/test_label），也支持
    `(train, test)` 或 `(train, validation, test)` 元组格式。演示统一从
    原始训练部分划出内层验证集，保留原始测试部分用于外层评估。
    """
    path = Path(path)
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    # 通过文件头识别 gzip。显式给出的坏路径或坏文件会保留原始错误。
    opener = gzip.open if compressed else open
    with opener(path, "rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    if isinstance(raw, dict):
        return raw["train_img"], raw["train_label"], raw["test_img"], raw["test_label"]
    if isinstance(raw, tuple) and len(raw) == 2:
        (x_train, y_train), (x_test, y_test) = raw
        return x_train, y_train, x_test, y_test
    if isinstance(raw, tuple) and len(raw) == 3:
        (x_train, y_train), _, (x_test, y_test) = raw
        return x_train, y_train, x_test, y_test
    raise ValueError(f"Unsupported MNIST format: {type(raw)!r}")


def run_mnist_demo(trials: int = 4, seed: int = 7, output: str | Path = "artifacts/automl_mnist",
                   data: str | Path | None = None, train_samples: int = 256,
                   test_samples: int = 256, epochs: int = 1, *, val_samples: int = 64,
                   inner_folds: int = 1, runs: int = 1, split_seed: int = 0,
                   strategy: str = "random", profiling: Profiling = None):
    """执行 MNIST 调参和外层评估，返回首个种子的搜索结果。

    ``train_samples + val_samples`` 决定开发集的样本预算。留出模式下，
    两者分别控制训练和验证样本数。K 折模式会重新划分整个开发集。
    """
    if min(train_samples, val_samples, test_samples, epochs, inner_folds) < 1:
        raise ValueError("sample counts, epochs and inner_folds must be positive")
    development_samples = train_samples + val_samples
    if data is not None:
        x_train, y_train, x_test, y_test = _load_mnist(data)
        data_source = str(Path(data).resolve())
    else:
        # 用一次生成的不同样本构造开发集和外层测试集。训练种子不参与生成。
        x, y = make_tiny_mnist(seed=split_seed, samples=development_samples + test_samples)
        x_train, y_train = x[:development_samples], y[:development_samples]
        x_test, y_test = x[development_samples:], y[development_samples:]
        data_source = "generated_fixture"

    if len(x_train) < 2 or len(x_test) == 0:
        raise ValueError("MNIST needs at least two training samples and one outer test sample")
    # 整个实验共享固定样本划分，只有搜索和模型训练会随运行种子变化。
    split_rng = np.random.default_rng(split_seed)
    dev_idx = split_rng.permutation(len(x_train))[:development_samples]
    test_idx = split_rng.permutation(len(x_test))[:test_samples]
    split = min(train_samples, len(dev_idx) - 1)
    if inner_folds > len(dev_idx):
        raise ValueError("inner_folds cannot exceed the development sample count")
    x_dev, y_dev = np.asarray(x_train)[dev_idx], np.asarray(y_train)[dev_idx]
    x_outer, y_outer = np.asarray(x_test)[test_idx], np.asarray(y_test)[test_idx]

    def executor_factory(run_seed: int) -> MNISTExecutor:
        return MNISTExecutor(
            x_dev[:split], y_dev[:split], x_dev[split:], y_dev[split:],
            epochs=epochs, seed=run_seed, max_samples=None,
            x_test=x_outer, y_test=y_outer, inner_folds=inner_folds, split_seed=split_seed,
        )

    if inner_folds > 1:
        val_sizes = [len(fold) for fold in np.array_split(dev_idx, inner_folds)]
        train_sizes = [len(dev_idx) - size for size in val_sizes]
    else:
        train_sizes, val_sizes = [split], [len(dev_idx) - split]
    report = run_evaluation(
        executor_factory,
        SearchSpace(lr=(1e-3, 3e-1), batch_size=(16, 32), optimizer=("sgd", "adam")),
        baseline_config={"lr": 0.001, "batch_size": 32, "optimizer": "adam"},
        n_trials=trials,
        seed=seed,
        runs=runs,
        output=output,
        strategy=strategy,
        profiling=profiling,
        protocol={
            "task": "mnist",
            "data": data_source,
            "model": "numpy_linear_softmax",
            "objective": "validation_error_rate",
            "test_metric": "test_error_rate",
            "seed": seed,
            "split_seed": split_seed,
            "inner_split": "k_fold" if inner_folds > 1 else "holdout",
            "inner_folds": inner_folds,
            "inner_train_sizes": train_sizes,
            "inner_validation_sizes": val_sizes,
            "development_samples": len(dev_idx),
            "test_samples": len(test_idx),
            "development_indices": dev_idx.tolist(),
            "outer_test_indices": test_idx.tolist(),
            "outer_split": "fixture_holdout" if data is None else "official_mnist_test_subset",
            "epochs": epochs,
            "device": "cpu",
        },
    )
    # 搜索报告保留首个种子的事件，外层分数只出现在评估报告里。
    directory = Path(output)
    print(json.dumps({"data": data_source,
                      "strategy": strategy,
                      "best_config": report.best_config,
                      "best_result": report.best_result.to_dict() if report.best_result else None,
                      "report": str(directory / "report.json"),
                      "events": str(directory / "events.jsonl"),
                      "evaluation": str(directory / "evaluation.json")}, indent=2))
    return report


if __name__ == "__main__":
    # CLI 参数保持与 run_mnist_demo 一一对应，使命令行和 Python API 行为一致。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--strategy", choices=SEARCH_STRATEGIES, default="random", help="搜索策略，默认随机搜索")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data", default=None, help="MNIST pickle 或 gzip 压缩 pickle 路径")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64,
                        help="留出验证样本数，交叉验证时与训练样本数一起确定开发集大小")
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--inner-folds", type=int, default=1, help="1 表示留出验证，2 及以上表示 K 折")
    parser.add_argument("--runs", type=int, default=1, help="使用连续训练种子重复的次数")
    parser.add_argument("--split-seed", type=int, default=0, help="固定数据划分和验证折的种子")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default="artifacts/automl_mnist")
    add_profiling_arguments(parser)
    args = parser.parse_args()
    run_mnist_demo(args.trials, args.seed, args.output, args.data,
                   args.train_samples, args.test_samples, args.epochs,
                   val_samples=args.val_samples, inner_folds=args.inner_folds,
                   runs=args.runs, split_seed=args.split_seed, strategy=args.strategy,
                   profiling=profiling_from_args(args))
