"""在车辆行为克隆 ZIP 数据上运行基础 AutoML 调参 agent。

脚本默认使用 center/left/right 三路摄像头，并为左右图像应用标准的
转向修正。数据不会整体解压到内存；执行器按 batch 从 ZIP 中读取图像。

CUDA 示例：
`python demos/automl_car_agent.py --zip D:\\Undergraduate\\WZH\\EnNeuro\\car_behavioral_cloning_data.zip --device cuda --backend rawmodule --trials 3`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import EventLogger, SearchSpace, SequentialRandomSearch
from eneuro.agent.car_executor import CarBehavioralCloningExecutor


def run_car_demo(zip_path: str | Path, *, trials: int = 3, epochs: int = 1,
                 seed: int = 42, device: str = "cuda", backend: str = "auto",
                 image_width: int = 64, image_height: int = 32,
                 cameras: tuple[str, ...] = ("center", "left", "right"),
                 max_samples: int | None = None,
                 output: str | Path = "artifacts/automl_car_full"):
    """构造车辆执行器、运行固定预算搜索并保存报告。

    调参目标是验证集 MSE；报告还会保留测试集 MSE/MAE，便于观察最优
    验证配置在未参与决策的数据上的表现。
    """
    executor = CarBehavioralCloningExecutor(
        zip_path,
        epochs=epochs,
        seed=seed,
        device=device,
        backend=backend,
        image_size=(image_width, image_height),
        cameras=cameras,
        max_samples=max_samples,
    )
    logger = EventLogger()
    search_space = SearchSpace(
        lr=(1e-5, 3e-3),
        batch_size=(32, 64),
        optimizer=("adam", "sgd"),
    )
    agent = SequentialRandomSearch(
        executor,
        search_space,
        n_trials=trials,
        seed=seed,
        logger=logger,
    )
    result = agent.run()
    # JSON 报告用于查看全局结果，JSONL 事件日志用于按 trial 复盘决策过程。
    report_path, events_path = logger.write_report(output, result)
    payload = {
        "best_config": result.best_config,
        "best_result": result.best_result.to_dict() if result.best_result else None,
        "trials": len(result.trials),
        "report": str(report_path),
        "events": str(events_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    """解析命令行参数并启动车辆调参演示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
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
        args.zip_path,
        trials=args.trials,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        backend=args.backend,
        image_width=args.image_width,
        image_height=args.image_height,
        cameras=tuple(args.cameras),
        max_samples=args.max_samples or None,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

