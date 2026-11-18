"""复用上次 Car 终训配置重训 100 轮，验证后供模拟器加载。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from drive_car_simulator import DEFAULT_MODEL, REFIT_DIR, REFIT_MODEL, REFIT_VERIFIED, ROOT

EPOCHS = 100
SOURCE_REPORT = DEFAULT_MODEL.with_name("report.json")
REFIT_REPORT = REFIT_DIR / "report.json"


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def train(source: dict, data_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "code"))
    from eneuro.agent import CarBehavioralCloningExecutor

    protocol = source["protocol"]
    config = source["best_config"]
    executor = CarBehavioralCloningExecutor(
        data_dir, refit_epochs=EPOCHS, seed=source["seed"], device="cuda",
        backend=protocol["backend"], device_id=protocol["device_id"], model="resnet18",
        image_size=tuple(protocol["image_size"]), cameras=tuple(protocol["cameras"]),
        camera_correction=protocol["camera_correction"], max_samples=protocol["max_samples"],
    )
    if (executor.split_metadata() != protocol["splits"]
            or executor.target_normalization() != protocol["target_normalization"]):
        raise ValueError("当前数据划分或目标归一化参数与上次终训不同")

    print(f"{executor.device_name} | seed={source['seed']} | {config}", flush=True)
    print(f"100 轮重新训练 | 输出目录 {REFIT_DIR}", flush=True)
    started = time.perf_counter()

    def log_epoch(epoch: int, loss: float, _validation: float | None) -> None:
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch:03d}/{EPOCHS} | train_mse={loss:.6f} | elapsed_s={elapsed:.1f}",
              flush=True)

    metrics = executor.evaluate_final(config, on_epoch=log_epoch, checkpoint=REFIT_MODEL)
    return {
        "source_report": str(SOURCE_REPORT),
        "seed": source["seed"],
        "best_config": config,
        "protocol": {**protocol, "refit_epochs": EPOCHS, "device": executor.device_name},
        "outer_test": {"status": "completed", **metrics},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autodrive-python", type=Path,
                        default=Path.home() / ".conda" / "envs" / "AutoDrive" / "python.exe",
                        help="AutoDrive 环境的 Python")
    args = parser.parse_args()
    if not args.autodrive_python.is_file():
        parser.error(f"找不到 AutoDrive Python：{args.autodrive_python}")

    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    data_dir = Path(source["protocol"]["data"])
    if (source["protocol"]["task"] != "car_behavioral_cloning"
            or source["outer_test"]["status"] != "completed"):
        parser.error("来源必须是已完成终训的 Car Agent 报告")
    # 挑一张真实的中央摄像头帧做接口验证。
    sample = next((data_dir / "IMG").glob("center_*.jpg"), None)
    if sample is None:
        parser.error(f"找不到中央摄像头图像：{data_dir / 'IMG'}")
    validator = Path(__file__).with_name("check_car_simulator.py")
    if not validator.is_file():
        parser.error(f"找不到模拟器验证脚本：{validator}")

    if REFIT_REPORT.is_file():
        report = json.loads(REFIT_REPORT.read_text(encoding="utf-8"))
    else:
        REFIT_DIR.mkdir(exist_ok=False)
        report = train(source, data_dir)
        write_json(REFIT_REPORT, report)

    if report["outer_test"]["epochs"] != EPOCHS:
        raise ValueError("训练报告没有记录完整的 100 轮")
    steering_mae = report["outer_test"]["test_per_target"]["steering"]["mae"]
    print(f"留出集转向 MAE {steering_mae:.6f} | 训练报告 {REFIT_REPORT}", flush=True)
    validation = subprocess.check_output(
        [str(args.autodrive_python), "-s", "-X", "utf8", str(validator),
         str(REFIT_MODEL), str(sample), str(EPOCHS)], encoding="utf-8",
    )
    print(validation.strip(), flush=True)
    write_json(REFIT_VERIFIED, {"checkpoint": REFIT_MODEL.name, **json.loads(validation)})
    print(f"模拟器已通过验证。下次启动将默认加载 {REFIT_MODEL}", flush=True)


if __name__ == "__main__":
    main()
