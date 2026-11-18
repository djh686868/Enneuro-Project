"""在 CUDA 上限时搜索 ResNet18 四目标驾驶回归的超参数。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eneuro.agent import CarBehavioralCloningExecutor, SearchSpace
from eneuro.agent.search import SEARCH_STRATEGIES
from eneuro.agent.timed_search import run_timed_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="包含 driving_log.csv 和 IMG/ 的目录")
    # 默认按墙钟时间停止搜索，快速验证时也可以额外限制候选数。
    parser.add_argument("--budget-minutes", type=float, default=120, help="搜索时间预算（分钟），默认 %(default)s")
    parser.add_argument("--trials", type=int, help="可选的试验数上限，默认仅按时间停止")
    parser.add_argument("--strategy", choices=SEARCH_STRATEGIES, default="random", help="搜索策略，默认随机搜索")
    parser.add_argument("--epochs", type=int, default=10, help="每个候选配置的训练轮数，默认 %(default)s")
    parser.add_argument("--refit-epochs", type=int, default=30, help="最佳配置最终重训的轮数，默认 %(default)s")
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--cameras", nargs="+", choices=("center", "left", "right"),
                        default=("center", "left", "right"))
    parser.add_argument("--max-samples", type=int, help="快速验证时限制图像数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    default_output = ROOT / "artifacts" / f"automl_car_cuda_{datetime.now():%Y%m%d_%H%M%S}"
    parser.add_argument("--output", type=Path, default=default_output, help="本次运行的新输出目录")
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

    executor = CarBehavioralCloningExecutor(
        args.data_dir, device="cuda", backend="cupy", device_id=args.device_id, seed=args.seed,
        epochs=args.epochs, refit_epochs=args.refit_epochs, image_size=(args.image_width, args.image_height),
        max_samples=args.max_samples, cameras=tuple(args.cameras), model="resnet18",
    )
    device_name = executor.device_name
    print(f"设备 {device_name} | CUDA CuPy | EnNeuro ResNet18 四目标回归", flush=True)
    print(f"数据目录 {executor.data_dir}", flush=True)
    splits = executor.split_metadata()
    counts = "/".join(f"{splits[name]['images']:,}" for name in ("train", "validation", "test"))
    print(f"训练/验证/测试 {counts} 张 | {args.image_width}×{args.image_height}", flush=True)
    search_limit = f"最多 {args.trials} 组" if args.trials is not None else "按时间预算"
    print(f"搜索{search_limit}，每组 {args.epochs} epoch | 最终重训 {args.refit_epochs} epoch，单独计时",
          flush=True)
    print(f"回归目标 {', '.join(executor.target_names)}", flush=True)

    def format_test(scores: dict) -> str:
        per_target = scores["test_per_target"]
        fields = " | ".join(f"{name} MAE {per_target[name]['mae']:.4f}" for name in executor.target_names)
        return f"标准化 MSE {scores['test_mse']:.6f} | {fields}"

    return run_timed_search(
        executor, SearchSpace(lr=(1e-4, 3e-3), batch_size=(16, 32), optimizer=("adam", "sgd")),
        budget_s=budget_s, trials=args.trials, seed=args.seed, output=args.output,
        strategy=args.strategy,
        protocol={
            "task": "car_behavioral_cloning", "data": str(executor.data_dir),
            "model": "ResNet18Regressor", "device": device_name, "device_id": args.device_id,
            "backend": executor.backend, "image_size": list(executor.image_size),
            "epochs": args.epochs, "refit_epochs": args.refit_epochs,
            "max_samples": args.max_samples, "cameras": list(executor.cameras),
            "camera_correction": executor.camera_correction,
            "split_type": "grouped_contiguous_inner_holdout_and_chronological_outer_holdout",
            "splits": splits, "objective": "val_mse", "test_metric": "test_mse",
            "target_names": list(executor.target_names), "target_normalization": executor.target_normalization(),
            "aggregate_metric": "mean_normalized_error_over_four_targets",
        },
        plot_ylabel="Best validation normalized MSE (4 targets)",
        format_validation=lambda mse: f"标准化 MSE {mse:.6f}",
        format_epoch_validation=lambda mse: f"标准化 MSE {mse:.6f}",
        format_test=format_test,
    )


if __name__ == "__main__":
    raise SystemExit(main())
