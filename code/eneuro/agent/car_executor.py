"""读取本地 CSV 和图像文件的车辆行为克隆调参执行器。

先从 driving_log.csv 读取路径和标签，DataLoader 请求样本时才读取并
解码对应的 JPEG。执行器支持 center/left/right 三路摄像头，并可对
左右摄像头的转向标签施加修正量。
"""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import cached_property
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import TrialResult

DRIVING_TARGETS = ("steering", "throttle", "brake", "speed")


# 统一把 CPU 上的 NumPy 数组和 CUDA 上的 CuPy 数组转换成 NumPy，方便
# 计算 MAE、写 JSON 报告以及比较不同设备上的同一个 trial。


def _as_numpy(value: Any) -> np.ndarray:
    """把 NumPy/CuPy 标量或数组统一转换成 NumPy 数组。

    验证阶段可能在 CPU 或 CUDA 上执行。CuPy 数组不能直接交给 NumPy 的
    统计函数，所以这里集中处理设备转换，避免在训练主流程中散落判断。
    """
    try:
        import cupy as cp  # type: ignore

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except ImportError:
        pass
    return np.asarray(value)


class _SearchDeadline(TimeoutError):
    pass


class _NonFiniteTrial(ValueError):
    pass


def preprocess_camera_image(encoded_image: bytes, image_size: tuple[int, int]) -> np.ndarray:
    """将摄像头图像解码为模型使用的 RGB、CHW、float32 输入。"""
    if not encoded_image:
        raise ValueError("摄像头图像为空")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("读取车辆图像需要安装 opencv-python") from exc

    image = cv2.imdecode(np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV 无法解码摄像头图像")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise _SearchDeadline("试验完成验证前已达到搜索时限")


class SteeringDataset:
    """按需从磁盘读取图像的车辆转向数据集。

    `rows` 保存 CSV 解析出的图像路径和标签，`indices` 只保存当前数据
    划分的行号。这样训练、验证、测试集可以共享同一份元数据，而不会复制
    图像本身。
    """

    def __init__(self, rows: Sequence[dict[str, Any]],
                 indices: Sequence[int], image_size: tuple[int, int] = (64, 32), *,
                 targets: np.ndarray | None = None):
        self.rows = rows
        self.indices = list(indices)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.targets = targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        from eneuro.base import Tensor

        source = self.indices[index]
        row = self.rows[source]
        # 先由 Python 读取文件字节，再由 OpenCV 解码，兼容 Windows 中文路径。
        path = row["image_path"]
        try:
            image = preprocess_camera_image(path.read_bytes(), self.image_size)
        except ValueError as exc:
            raise ValueError(f"OpenCV 无法解码图像：{path}") from exc
        target = (self.targets[source] if self.targets is not None
                  else np.asarray([row["steering"]], dtype=np.float32))
        return Tensor(image), Tensor(target)


def read_steering_rows(data_dir: str | Path, *, camera: str | Sequence[str] = "center",
                       max_samples: int | None = None, camera_correction: float = 0.2,
                       include_telemetry: bool = False) -> list[dict[str, Any]]:
    """读取并校验 driving_log.csv，但不解码图像。

    参数中的 `camera` 可以是单个摄像头名称，也可以是多个名称。left/right
    图像通常需要相对 center 图像施加转向修正，这里分别加上和减去
    `camera_correction`，并把标签裁剪到 [-1, 1]。
    """

    root = Path(data_dir).resolve()
    cameras = (camera,) if isinstance(camera, str) else tuple(camera)
    if not cameras or any(c not in {"center", "left", "right"} for c in cameras):
        raise ValueError("camera must be center, left, right, or a sequence of those names")
    rows: list[dict[str, Any]] = []
    with (root / "driving_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = DRIVING_TARGETS if include_telemetry else ("steering",)
        missing = set(required).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"driving_log.csv 缺少标签列：{', '.join(sorted(missing))}")
        for source_group, raw_row in enumerate(reader):
            try:
                values = tuple(float(raw_row[label]) for label in required)
            except (TypeError, ValueError):
                # 个别坏行不应使整个搜索任务失败，跳过即可。
                continue
            if not all(isfinite(value) for value in values):
                continue
            # 保留 CSV 行号，供训练、验证和测试按原始驾驶记录分组。
            for selected_camera in cameras:
                image_ref = (raw_row.get(selected_camera) or "").strip()
                if not image_ref:
                    continue
                relative = Path(image_ref.replace("\\", "/").removeprefix("data/"))
                if relative.anchor or ".." in relative.parts:
                    continue
                image_path = root / relative
                if not image_path.is_file():
                    continue
                correction = 0.0
                if selected_camera == "left":
                    correction = float(camera_correction)
                elif selected_camera == "right":
                    correction = -float(camera_correction)
                steering = float(np.clip(values[0] + correction, -1.0, 1.0))
                row = {
                    "image_path": image_path,
                    "steering": steering,
                    "camera": selected_camera,
                    "group": source_group,
                }
                if include_telemetry:
                    # 左右摄像头的方向偏移只改变转向标签，其余标签来自同一驾驶记录。
                    row["targets"] = (steering, *values[1:])
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    break
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"no valid steering rows found in {root / 'driving_log.csv'}")
    return rows


def _validation_start(development_groups: list[int], size: int, brake_groups: set[int]) -> int:
    """在开发数据中选连续验证块，尽量在训练和验证两边保留刹车记录。"""
    brake_positions = [i for i, group in enumerate(development_groups) if group in brake_groups]
    if len(brake_positions) >= 2:
        prefix = np.cumsum([0, *(group in brake_groups for group in development_groups)])
        latest_start = len(development_groups) - size
        for position in reversed(brake_positions):
            start = min(max(0, position - size // 2), latest_start)
            # 避免通过切开同一段刹车来人为制造两边都有正例。
            near_edges = (start - 1, start, start + size - 1, start + size)
            if any(0 <= i < len(development_groups) and development_groups[i] in brake_groups
                   for i in near_edges):
                continue
            if 0 < prefix[start + size] - prefix[start] < len(brake_positions):
                return start
    return len(development_groups) - size


class TinySteeringRegressor:
    """用于车辆调参演示的小型 EnNeuro 回归模型。

    首轮端到端验收的重点是验证调参 agent 的闭环，而不是比较复杂 CNN 的
    精度。这里使用一个延迟初始化的 Linear 头直接回归转向角，既能真正
    调用 EnNeuro Trainer，又能把每个 trial 的运行时间控制在可接受范围内。
    将来可以在相同 Executor 接口中替换为更大的 CNN。
    """

    def __new__(cls):
        # Linear 是延迟初始化的，第一次前向时会根据展平后的图像维度创建权重。
        from eneuro.base import functions as F
        from eneuro.nn.module import Linear, Module

        class _Model(Module):
            def __init__(self):
                super().__init__()
                self.fc = Linear(1)

            def forward(self, x):
                x = F.flatten(x)
                return self.fc(x)

        return _Model()


class CarBehavioralCloningExecutor:
    """在本地车辆数据上执行一次 EnNeuro 回归 trial。

    AutoML 搜索器只需要调用 `execute(config)`。执行器内部负责固定随机
    种子、读取 CSV、建立时间顺序划分、训练模型并返回验证 MSE。
    配置选定后，evaluate_final 用训练与验证数据重训，再评分外层留出集。
    重训可以指定独立的轮数，未指定时沿用候选配置的训练轮数。
    """

    def __init__(self, data_dir: str | Path, *, epochs: int = 1, refit_epochs: int | None = None,
                 seed: int = 0, device: str = "cpu", backend: str = "auto",
                 device_id: int = 0,
                 image_size: tuple[int, int] = (64, 32), max_samples: int | None = None,
                 cameras: Sequence[str] = ("center",), camera_correction: float = 0.2,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, model: str = "tiny"):
        if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio and val_ratio must leave a test split")
        if model not in {"tiny", "resnet18"}:
            raise ValueError("model must be tiny or resnet18")
        self.model = model
        self.target_names = DRIVING_TARGETS if model == "resnet18" else ("steering",)
        self.data_dir = Path(data_dir).resolve()
        self.epochs = int(epochs)
        self.refit_epochs = self.epochs if refit_epochs is None else int(refit_epochs)
        if refit_epochs is not None and self.refit_epochs < 1:
            raise ValueError("refit_epochs must be positive")
        self.seed = int(seed)
        self.device = str(device).lower()
        self.device_id = device_id
        self.backend = str(backend).lower()
        self.image_size = image_size
        self.max_samples = max_samples
        self.cameras = tuple(cameras)
        self.camera_correction = float(camera_correction)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)

    @cached_property
    def device_name(self) -> str:
        """确认 CUDA 设备可用并取得名称。"""
        if self.device not in {"cuda", "gpu"}:
            return "CPU"
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
        if not 0 <= self.device_id < count:
            raise RuntimeError(f"CUDA 设备 {self.device_id} 不可用（可见设备数 {count}）")
        name = cp.cuda.runtime.getDeviceProperties(self.device_id)["name"]
        return name.decode(errors="replace") if isinstance(name, bytes) else str(name)

    @contextmanager
    def _device_scope(self):
        if self.device not in {"cuda", "gpu"}:
            yield
            return
        import cupy as cp
        from eneuro.base.cuda import dispatch

        previous = dispatch.get_backend()
        try:
            if self.backend != "auto":
                dispatch.set_backend(self.backend)
            with cp.cuda.Device(self.device_id):
                yield
        finally:
            dispatch.set_backend(previous)

    @cached_property
    def _splits(self):
        # 按 CSV 行划分，哪怕样本上限截断了最后一行的摄像头也不跨集合。
        rows = read_steering_rows(self.data_dir, camera=self.cameras,
                                  max_samples=self.max_samples,
                                  camera_correction=self.camera_correction,
                                  include_telemetry=self.model == "resnet18")
        groups = sorted({row["group"] for row in rows})
        n_groups = len(groups)
        if n_groups < 3:
            raise ValueError("need at least three driving records for train/validation/test")
        train_end = min(max(1, int(n_groups * self.train_ratio)), n_groups - 2)
        val_end = min(max(train_end + 1, int(n_groups * (self.train_ratio + self.val_ratio))),
                      n_groups - 1)
        test_groups = set(groups[val_end:])
        validation_start = train_end
        if self.model == "resnet18":
            brake_groups = {row["group"] for row in rows
                            if row["group"] not in test_groups and row["targets"][2] > 0}
            validation_start = _validation_start(groups[:val_end], val_end - train_end, brake_groups)
        validation_groups = set(groups[validation_start:validation_start + val_end - train_end])
        train, val, test = [], [], []
        for i, row in enumerate(rows):
            group = row["group"]
            split = test if group in test_groups else val if group in validation_groups else train
            split.append(i)
        return rows, train, val, test

    def split_metadata(self) -> dict[str, Any]:
        rows, train, val, test = self._splits
        splits = {}
        for name, indices in (("train", train), ("validation", val), ("test", test)):
            groups = sorted({rows[i]["group"] for i in indices})
            info = {"images": len(indices), "driving_rows": len(groups),
                    "first_csv_row_index": groups[0], "last_csv_row_index": groups[-1]}
            if self.model == "resnet18":
                ranges = [[groups[0], groups[0]]]
                for group in groups[1:]:
                    if group == ranges[-1][1] + 1:
                        ranges[-1][1] = group
                    else:
                        ranges.append([group, group])
                info["csv_row_ranges_inclusive"] = ranges
                info["braking_driving_rows"] = len({rows[i]["group"] for i in indices if rows[i]["targets"][2] > 0})
            splits[name] = info
        return splits

    @cached_property
    def _normalized_targets(self):
        if len(self.target_names) == 1:
            return None, None
        rows, train, _, _ = self._splits
        targets = np.asarray([row["targets"] for row in rows], dtype=np.float32)
        # 统计量只从内层训练样本拟合。固定同一坐标系用于所有候选和最终重训。
        mean = targets[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = targets[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale > 1e-6, scale, np.float32(1))
        stats = {name: {"mean": float(mu), "scale": float(sigma)}
                 for name, mu, sigma in zip(self.target_names, mean, scale)}
        return (targets - mean) / scale, stats

    def target_normalization(self) -> dict | None:
        return self._normalized_targets[1]

    def execute(self, config: Mapping[str, Any], *, deadline: float | None = None,
                on_epoch: Callable[[int, float, float | None], None] | None = None) -> TrialResult:
        """执行车辆回归试验，超时或发散的结果不参与排名。"""
        try:
            _check_deadline(deadline)
            _, train, val, _ = self._splits
            with self._device_scope():
                metrics = self._fit_and_score(
                    config, train, val, "val", epochs=self.epochs, deadline=deadline, on_epoch=on_epoch,
                )
            return TrialResult(metrics["val_mse"], metrics)
        except _SearchDeadline as exc:
            return TrialResult(float("inf"), status="timed_out", error=str(exc))
        except _NonFiniteTrial as exc:
            return TrialResult(float("inf"), status="failed", error=str(exc))
        except Exception as exc:
            if deadline is not None:
                # 限时脚本遇到设备或数据错误时结束运行，避免不断重复失败。
                raise
            return TrialResult(float("inf"), status="failed",
                               error=f"{type(exc).__name__}: {exc}")

    def evaluate_final(self, config: Mapping[str, Any], *,
                       on_epoch: Callable[[int, float, float | None], None] | None = None,
                       checkpoint: str | Path | None = None) -> dict[str, Any]:
        """用完整开发集重训，并对选定配置评分一次。"""
        _, train, val, test = self._splits
        with self._device_scope():
            return self._fit_and_score(
                config, train + val, test, "test", epochs=self.refit_epochs, on_epoch=on_epoch, checkpoint=checkpoint,
            )

    def _fit_and_score(self, config: Mapping[str, Any], train_indices: list[int],
                       score_indices: list[int], split: str, *, epochs: int, deadline: float | None = None,
                       on_epoch=None, checkpoint: str | Path | None = None) -> dict[str, Any]:
        from eneuro.base import Config, as_Tensor
        from eneuro.data import DataLoader
        from eneuro.nn.loss import meanSquaredError
        from eneuro.nn.optim import Adam, MomentumSGD, SGD
        from eneuro.train.trainer import Trainer

        # 同一评估对的所有配置共享训练种子，控制初始化和样本顺序。
        np.random.seed(self.seed)
        random.seed(self.seed)
        _check_deadline(deadline)
        rows = self._splits[0]
        start = time.perf_counter()
        targets, normalization = self._normalized_targets
        train_ds = SteeringDataset(rows, train_indices, self.image_size, targets=targets)
        score_ds = SteeringDataset(rows, score_indices, self.image_size, targets=targets)
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        score_loader = DataLoader(score_ds, batch_size=32 if self.model == "resnet18" else 64, shuffle=False)

        if self.model == "resnet18":
            from eneuro.nn.resnet18_regression import ResNet18Regressor

            model = ResNet18Regressor(out_features=len(self.target_names))
        else:
            model = TinySteeringRegressor()
        sample_x, _ = train_ds[0]
        # 先用 NumPy 的固定种子初始化延迟层，训练开始前再移动参数到 CUDA。
        # 初始化阶段禁用 BatchNorm 统计量更新。
        with Config.no_grad(), Config.test_mode():
            model(as_Tensor(np.expand_dims(sample_x.data, axis=0)))
        model.to(self.device)
        params = {}
        model._flatten_params(params)
        if self.device in {"cuda", "gpu"}:
            import cupy as cp

            if not all(isinstance(param.data, cp.ndarray) for param in params.values()):
                raise RuntimeError("车辆模型参数未移动到 CUDA")
        optimizer_name = str(config.get("optimizer", "adam")).lower()
        optimizer_cls = {"adam": Adam, "sgd": SGD, "momentum_sgd": MomentumSGD}.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"unknown optimizer: {optimizer_name}")
        optimizer = optimizer_cls([params[name] for name in sorted(params)], lr=float(config["lr"]))

        def check_training_batch(metrics: dict) -> None:
            # Trainer 在回调前将损失转为 float，当前批次的 GPU 工作已同步。
            _check_deadline(deadline)
            loss = metrics["batch_loss"]
            if loss is None or not isfinite(loss):
                raise _NonFiniteTrial("车辆训练损失出现非有限值")

        trainer = Trainer(model, meanSquaredError, optimizer,
                          on_batch_end=check_training_batch if deadline is not None else None)
        epochs = max(1, epochs)
        train_mse_by_epoch = []
        for epoch in range(epochs):
            _check_deadline(deadline)
            trainer._epoch = epoch
            train_mse, _ = trainer._one_step(train_loader, batch_size=batch_size,
                                             training=True, verbose=False, device=self.device)
            # 框架的 MSE 按样本数归约，这里再按标签数平均供日志比较。
            train_mse = float(train_mse) / len(self.target_names)
            if not isfinite(train_mse):
                raise _NonFiniteTrial("车辆训练损失出现非有限值")
            # 保留逐轮训练损失，便于回看重训后段是否仍在下降。
            train_mse_by_epoch.append(train_mse)
            if on_epoch is not None and (split == "test" or epoch + 1 < epochs):
                on_epoch(epoch + 1, train_mse, None)
        mse, mae, per_target = self._evaluate(
            model, score_loader, meanSquaredError, self.device, Config,
            deadline=deadline, normalization=normalization,
        )
        if not isfinite(mse) or not isfinite(mae):
            raise _NonFiniteTrial("车辆评估指标出现非有限值")
        if on_epoch is not None and split == "val":
            on_epoch(epochs, train_mse, mse)
        _check_deadline(deadline)

        metrics = {
            "train_mse": train_mse,
            "train_mse_by_epoch": train_mse_by_epoch,
            "epochs": epochs,
            f"{split}_mse": mse,
            f"{split}_mae": mae,
            "samples": len(rows),
            "train_samples": len(train_ds),
            f"{split}_samples": len(score_ds),
            "elapsed_s": time.perf_counter() - start,
            "device": self.device,
            "image_size": list(self.image_size),
        }
        if per_target is not None:
            metrics[f"{split}_per_target"] = per_target
        if checkpoint is not None:
            from eneuro.utils.serializer import Serializer

            model_config = {"type": "TinySteeringRegressor" if self.model == "tiny" else type(model).__name__,
                            "image_size": list(self.image_size)}
            if self.model == "resnet18":
                model_config.update(out_features=len(self.target_names), target_names=list(self.target_names))
            if normalization:
                model_config["target_normalization"] = normalization
            Serializer.save_checkpoint_binary(model, optimizer, epochs, checkpoint, model_config=model_config)
            metrics["checkpoint"] = str(checkpoint)
        return metrics

    @staticmethod
    def _evaluate(model, loader, loss_fn, device: str, Config, *, deadline: float | None = None,
                  normalization: dict | None = None):
        """报告归一化空间的平均误差，并按原始单位拆分四个目标。"""
        names = tuple(normalization) if normalization else ("steering",)
        squared = np.zeros(len(names), dtype=np.float64)
        absolute = np.zeros(len(names), dtype=np.float64)
        loss_sum = 0.0
        count = 0
        with Config.using_config("enable_backprop", False), Config.using_config("train", False):
            for xb, yb in loader:
                _check_deadline(deadline)
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                diff = _as_numpy(pred.data - yb.data)
                if normalization:
                    squared += np.square(diff, dtype=np.float64).sum(axis=0)
                else:
                    loss_sum += float(_as_numpy(loss_fn(pred, yb).data)) * len(diff)
                absolute += np.abs(diff).sum(axis=0, dtype=np.float64)
                count += len(diff)
        _check_deadline(deadline)
        mae = absolute / count
        if normalization is None:
            return loss_sum / count, float(mae[0]), None
        mse = squared / count
        per_target = {
            name: {"mae": float(mae[i] * normalization[name]["scale"]),
                   "rmse": float(np.sqrt(mse[i]) * normalization[name]["scale"])}
            for i, name in enumerate(names)
        }
        return float(mse.mean()), float(mae.mean()), per_target

