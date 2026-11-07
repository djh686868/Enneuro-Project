"""运行一个可复现的 MNIST 小规模自动调参演示。

脚本既可以读取真实的 `mnist.pkl`，也可以在没有数据文件时退化为内置
的合成 MNIST 形状数据。每个 trial 会训练一个轻量 softmax 分类器，搜索
器负责选择学习率、batch size 和 optimizer，并把完整事件流写入 JSON/JSONL。

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import EventLogger, MNISTExecutor, SearchSpace, SequentialRandomSearch


def _load_mnist(path: str | Path):
    """兼容项目中常见的三种 MNIST pickle 文件结构。

    既支持字典格式（train_img/train_label/test_img/test_label），也支持
    `(train, test)` 或 `(train, validation, test)` 元组格式；这样演示脚本
    不需要修改原始数据文件。
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception:
        # 某些公开 MNIST 文件是 gzip 压缩的 pickle，普通读取失败后再尝试
        # latin1 解码以兼容 Python 2 时代保存的对象。
        with gzip.open(path, "rb") as handle:
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
                   test_samples: int = 256, epochs: int = 1):
    """执行 MNIST 调参并保存报告。

    `data` 存在时使用真实 MNIST；否则使用 MNISTExecutor 内置的确定性
    fixture。`train_samples/test_samples` 和 `epochs` 是为了让教学演示
    在几秒内完成，而不是追求完整训练精度。
    """
    logger = EventLogger()
    if data and Path(data).exists():
        # 验证集直接使用测试文件的前若干样本，保证每个 trial 的评估集合
        # 固定，调参结果差异主要来自超参数而不是数据划分。
        x_train, y_train, x_test, y_test = _load_mnist(data)
        executor = MNISTExecutor(
            x_train=x_train[:train_samples],
            y_train=y_train[:train_samples],
            x_val=x_test[:test_samples],
            y_val=y_test[:test_samples],
            seed=seed,
            epochs=epochs,
            max_samples=train_samples,
        )
        data_source = str(data)
    else:
        # 没有真实数据时仍然保留完整 observe/decide/execute/feedback 流程，
        # 方便 CI 和课程示例在干净环境中运行。
        executor = MNISTExecutor(seed=seed, epochs=epochs, max_samples=train_samples)
        data_source = "generated_fixture"
    agent = SequentialRandomSearch(
        executor,
        SearchSpace(lr=(1e-3, 3e-1), batch_size=(16, 32), optimizer=("sgd", "adam")),
        n_trials=trials,
        seed=seed,
        logger=logger,
    )
    report = agent.run()
    # 报告包含 trial 表、最优配置和四类事件；JSON 适合整体查看，JSONL
    # 适合逐行流式消费或接入后续监控工具。
    json_path, jsonl_path = logger.write_report(output, report)
    print(json.dumps({"data": data_source,
                      "best_config": report.best_config,
                      "best_result": report.best_result.to_dict() if report.best_result else None,
                      "report": str(json_path), "events": str(jsonl_path)}, indent=2))
    return report


if __name__ == "__main__":
    # CLI 参数保持与 run_mnist_demo 一一对应，使命令行和 Python API 行为一致。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data", default=None, help="Path to an MNIST pickle/gzip pickle")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default="artifacts/automl_mnist")
    args = parser.parse_args()
    run_mnist_demo(args.trials, args.seed, args.output, args.data,
                   args.train_samples, args.test_samples, args.epochs)
