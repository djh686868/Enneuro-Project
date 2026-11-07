"""Run a small deterministic MNIST-shaped AutoML search.

Usage: ``python demos/automl_mnist_agent.py --trials 4 --output artifacts/automl``
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
    path = Path(path)
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception:
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
    logger = EventLogger()
    if data and Path(data).exists():
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
    json_path, jsonl_path = logger.write_report(output, report)
    print(json.dumps({"data": data_source,
                      "best_config": report.best_config,
                      "best_result": report.best_result.to_dict() if report.best_result else None,
                      "report": str(json_path), "events": str(jsonl_path)}, indent=2))
    return report


if __name__ == "__main__":
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
