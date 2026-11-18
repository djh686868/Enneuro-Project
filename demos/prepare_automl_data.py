"""准备 Agent 演示的数据，或在训练前检查现有数据和划分。"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def _check_mnist(x_train, y_train, x_test, y_test) -> dict[str, int]:
    counts = {}
    for name, images, labels, minimum in (
        ("development", x_train, y_train, 2),
        ("test", x_test, y_test, 1),
    ):
        images = np.asarray(images)
        labels = np.asarray(labels)
        if images.shape[1:] not in ((784,), (28, 28), (1, 28, 28)):
            raise ValueError(f"{name}：图像必须为 N×784、N×28×28 或 N×1×28×28")
        if len(images) < minimum or labels.shape != (len(images),):
            raise ValueError(f"{name}：样本至少需要 {minimum} 张图像，且每张图像对应一个标签")
        if images.dtype.kind not in "uif" or not (0 <= images.min() <= images.max() <= 255):
            raise ValueError(f"{name}：像素值必须是 0 到 255 之间的有限数")
        if labels.dtype.kind not in "ui" or not (0 <= labels.min() <= labels.max() <= 9):
            raise ValueError(f"{name}：标签必须是 0 到 9 之间的整数")
        counts[name] = len(images)
    return counts


def _read_idx(directory: Path, name: str, *, image: bool) -> np.ndarray:
    path = directory / name
    if path.is_file():
        payload = path.read_bytes()
    else:
        path = directory / f"{name}.gz"
        with gzip.open(path, "rb") as stream:
            payload = stream.read()

    header_size = 16 if image else 8
    if len(payload) < header_size:
        raise ValueError(f"IDX 文件头不完整：{path}")
    magic, count = struct.unpack_from(">II", payload)
    expected_magic = 2051 if image else 2049
    shape = (count, 28, 28) if image else (count,)
    if magic != expected_magic or (image and struct.unpack_from(">II", payload, 8) != (28, 28)):
        raise ValueError(f"IDX 类型或图像尺寸不匹配：{path}")
    if len(payload) != header_size + count * (784 if image else 1):
        raise ValueError(f"IDX 内容长度与头部记录不一致：{path}")
    return np.frombuffer(payload, dtype=np.uint8, offset=header_size).reshape(shape)


def _convert_mnist(directory: Path, output: Path) -> dict[str, int]:
    train = (_read_idx(directory, "train-images-idx3-ubyte", image=True),
             _read_idx(directory, "train-labels-idx1-ubyte", image=False))
    test = (_read_idx(directory, "t10k-images-idx3-ubyte", image=True),
            _read_idx(directory, "t10k-labels-idx1-ubyte", image=False))
    counts = _check_mnist(*train, *test)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("xb")
    try:
        with stream:
            # 训练和测试保留原始分区。内层训练和验证的划分由启动脚本完成。
            pickle.dump((train, test), stream, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="dataset", required=True)
    mnist = commands.add_parser("mnist", help="检查已有的 MNIST pickle")
    mnist.add_argument("--data", type=Path, required=True, help="可信来源的 MNIST pickle")
    idx = commands.add_parser("mnist-idx", help="将原始 MNIST IDX 文件转换为 pickle")
    idx.add_argument("--source-dir", type=Path, required=True, help="包含四个 IDX 文件的目录，可为 .gz")
    idx.add_argument("--output", type=Path, required=True, help="生成的 pickle 路径，须为新文件")
    car = commands.add_parser("car", help="检查车辆 CSV 数据的训练、验证和测试划分")
    car.add_argument("--data-dir", type=Path, required=True)
    car.add_argument("--telemetry", action="store_true", help="按四目标 CUDA 入口检查标签与划分")
    donkey = commands.add_parser("donkeycar", help="检查历史 DonkeyCar 图像的分组和划分")
    donkey.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.dataset == "mnist":
            from automl_mnist_agent import _load_mnist

            path = args.data
            summary = _check_mnist(*_load_mnist(path))
        elif args.dataset == "mnist-idx":
            path = args.output
            summary = _convert_mnist(args.source_dir, path)
        elif args.dataset == "car":
            from eneuro.agent.car_executor import CarBehavioralCloningExecutor

            path = args.data_dir
            model = "resnet18" if args.telemetry else "tiny"
            executor = CarBehavioralCloningExecutor(path, model=model, cameras=("center", "left", "right"))
            summary = executor.split_metadata()
        else:
            from eneuro.agent.donkeycar_executor import DonkeyCarSteeringExecutor

            path = args.data_dir
            summary = DonkeyCarSteeringExecutor(path, device="cpu").split_metadata()
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        parser.exit(1, f"数据准备失败：{exc}\n")
    print(json.dumps({"dataset": args.dataset, "path": str(path.resolve()), "summary": summary},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
