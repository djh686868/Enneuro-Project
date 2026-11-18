"""在解压后的车辆行为克隆数据上运行基础 AutoML 调参。

数据目录包含 driving_log.csv 和 IMG/。默认使用 center、left、right
三路摄像头，对左右图像应用转向修正，并按批次读取所需图像。

CPU 示例
`python demos/automl_car_agent.py --data-dir data --device cpu --trials 3`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import SearchSpace
from eneuro.agent.car_executor import CarBehavioralCloningExecutor
from eneuro.agent.evaluation import run_evaluation
from eneuro.agent.search import SEARCH_STRATEGIES


def run_car_demo(data_dir: str | Path, *, trials: int = 3, epochs: int = 1,
                 seed: int = 42, device: str = "cuda", backend: str = "auto",
                 image_width: int = 64, image_height: int = 32,
                 cameras: tuple[str, ...] = ("center", "left", "right"),
                 max_samples: int | None = None, runs: int = 1,
                 output: str | Path = "artifacts/automl_car_full", strategy: str = "random"):
    """构造车辆执行器、运行固定预算搜索并保存报告。

    按驾驶记录划分数据，内层验证 MSE 用于选配置，外层 MSE/MAE 用于报告。
    """
    def executor_factory(run_seed: int) -> CarBehavioralCloningExecutor:
        return CarBehavioralCloningExecutor(
            data_dir, epochs=epochs, seed=run_seed, device=device, backend=backend,
            image_size=(image_width, image_height), cameras=cameras, max_samples=max_samples,
        )

    search_space = SearchSpace(
        lr=(1e-5, 3e-3),
        batch_size=(32, 64),
        optimizer=("adam", "sgd"),
    )
    result = run_evaluation(
        executor_factory,
        search_space,
        baseline_config={"lr": 1e-3, "batch_size": 64, "optimizer": "adam"},
        n_trials=trials,
        seed=seed,
        runs=runs,
        output=output,
        strategy=strategy,
        protocol={
            "task": "car_behavioral_cloning", "objective": "val_mse", "test_metric": "test_mse",
            "data": str(Path(data_dir).resolve()), "split_type": "chronological_by_csv_row",
            "splits": executor_factory(seed).split_metadata(),
            "cameras": list(cameras), "camera_correction": 0.2, "max_samples": max_samples,
            "epochs": max(1, epochs), "device": device, "cuda_backend": backend if device == "cuda" else None,
            "image_size": [image_width, image_height],
        },
    )
    payload = {
        "strategy": strategy,
        "best_config": result.best_config,
        "best_result": result.best_result.to_dict() if result.best_result else None,
        "trials": len(result.trials),
        "report": str(Path(output) / "report.json"),
        "events": str(Path(output) / "events.jsonl"),
        "evaluation": str(Path(output) / "evaluation.json"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    """解析命令行参数并启动车辆调参演示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="包含 driving_log.csv 和 IMG/ 的目录，默认使用项目根目录下的 data/")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--strategy", choices=SEARCH_STRATEGIES, default="random", help="搜索策略，默认随机搜索")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1, help="重复运行次数，使用连续的随机种子")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", choices=("auto", "cupy", "rawmodule"), default="auto")
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--cameras", nargs="+", choices=("center", "left", "right"),
                        default=("center", "left", "right"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="artifacts/automl_car_full")
    args = parser.parse_args()
    run_car_demo(
        args.data_dir,
        trials=args.trials,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        backend=args.backend,
        image_width=args.image_width,
        image_height=args.image_height,
        cameras=tuple(args.cameras),
        max_samples=args.max_samples or None,
        runs=args.runs,
        output=args.output,
        strategy=args.strategy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

