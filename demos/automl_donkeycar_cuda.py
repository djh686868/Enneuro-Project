"""在 CUDA 上限时搜索历史 DonkeyCar 转向角回归的超参数。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import SearchSpace
from eneuro.agent._profiling import add_profiling_arguments, profiling_from_args
from eneuro.agent.donkeycar_executor import DonkeyCarSteeringExecutor
from eneuro.agent.search import SEARCH_STRATEGIES
from eneuro.agent.timed_search import run_timed_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="包含 id_angle.jpg 的历史 DonkeyCar 目录")
    parser.add_argument("--budget-minutes", type=float, default=120,
                        help="搜索时间预算（分钟），默认 %(default)s")
    parser.add_argument("--trials", type=int, help="可选的试验数上限，默认仅按时间停止")
    parser.add_argument("--strategy", choices=SEARCH_STRATEGIES, default="random", help="搜索策略，默认随机搜索")
    parser.add_argument("--epochs", type=int, default=10,
                        help="每个候选配置的训练轮数，默认 %(default)s")
    parser.add_argument("--refit-epochs", type=int, default=30,
                        help="最佳配置最终重训的轮数，默认 %(default)s")
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--max-samples", type=int, help="快速验证时限制图像数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    default_output = ROOT / "artifacts" / f"automl_donkeycar_cuda_{datetime.now():%Y%m%d_%H%M%S}"
    parser.add_argument("--output", type=Path, default=default_output, help="本次运行的新输出目录")
    add_profiling_arguments(parser)
    args = parser.parse_args()

    budget_s = args.budget_minutes * 60
    if not isfinite(budget_s) or budget_s <= 0:
        parser.error("--budget-minutes 必须是正数")
    if min(args.epochs, args.refit_epochs, args.image_width, args.image_height) < 1:
        parser.error("训练轮数和图像宽高必须是正整数")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials 必须是正整数")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples 必须是正整数")

    executor = DonkeyCarSteeringExecutor(
        args.data_dir, device="cuda", backend="cupy", device_id=args.device_id, seed=args.seed,
        epochs=args.epochs, refit_epochs=args.refit_epochs, image_size=(args.image_width, args.image_height),
        max_samples=args.max_samples,
    )
    device_name = executor.device_name
    print(f"设备 {device_name} | CUDA CuPy | EnNeuro ResNet18 单目标转向角回归", flush=True)
    print(f"数据目录 {executor.data_dir}", flush=True)
    splits = executor.split_metadata()
    counts = "/".join(f"{splits[name]['images']:,}" for name in ("train", "validation", "test"))
    print(f"训练/验证/测试 {counts} 张 | {args.image_width}×{args.image_height}", flush=True)
    search_limit = f"最多 {args.trials} 组" if args.trials is not None else "按时间预算"
    print(f"搜索{search_limit}，每组 {args.epochs} epoch | 最终重训 {args.refit_epochs} epoch，单独计时",
          flush=True)

    space = SearchSpace(lr=(1e-4, 3e-3), batch_size=(16, 32), optimizer=("adam", "sgd"))
    return run_timed_search(
        executor, space, budget_s=budget_s, trials=args.trials, seed=args.seed, output=args.output,
        strategy=args.strategy,
        profiling=profiling_from_args(args),
        protocol={
            "task": "donkeycar_steering", "data": str(executor.data_dir),
            "model": "ResNet18Regressor", "device": device_name, "device_id": args.device_id,
            "backend": executor.backend, "image_size": list(executor.image_size),
            "epochs": args.epochs, "refit_epochs": args.refit_epochs,
            "max_samples": args.max_samples, "split_type": "grouped_contiguous_holdout_by_numeric_id",
            "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "splits": splits, "objective": "val_mse", "test_metric": "test_mse",
            "target_names": ["steering"], "target_source": "filename_id_angle",
            "target_normalization": None,
            "search_space": {"lr": list(space.lr), "batch_size": list(space.batch_size),
                             "optimizer": list(space.optimizer)},
        },
        plot_ylabel="Best validation steering MSE",
        format_validation=lambda mse: f"转向角 MSE {mse:.6f}",
        format_epoch_validation=lambda mse: f"转向角 MSE {mse:.6f}",
        format_test=lambda scores: f"转向角 MSE {scores['test_mse']:.6f} | MAE {scores['test_mae']:.6f}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
