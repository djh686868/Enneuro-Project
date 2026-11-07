"""基于 ZIP 数据集的车辆行为克隆调参执行器。

本文件负责把 `car_behavioral_cloning_data.zip` 适配为 AutoML agent 可以
调用的执行器。实现刻意采用“惰性读取”：每个 trial 只打开一次 ZIP，图像
在 DataLoader 请求某个样本时才解码，因此不会把 24,108 张图像全部展开到
内存中。执行器支持 center/left/right 三路摄像头，也支持对左右摄像头的
转向标签施加标准修正量。
"""

from __future__ import annotations

import csv
import io
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import TrialResult


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


class ZipSteeringDataset:
    """直接从 ZIP 中按需读取图像的车辆转向数据集。

    `rows` 保存 CSV 解析出的成员路径和标签，`indices` 只保存当前数据
    划分的行号。这样训练、验证、测试集可以共享同一份元数据，而不会复制
    图像本身。
    """

    def __init__(self, archive: zipfile.ZipFile, rows: Sequence[dict[str, Any]],
                 indices: Sequence[int], image_size: tuple[int, int] = (64, 32)):
        self.archive = archive
        self.rows = rows
        self.indices = list(indices)
        self.image_size = (int(image_size[0]), int(image_size[1]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        from eneuro.base import Tensor

        row = self.rows[self.indices[index]]
        # ZIP 内保存的是压缩后的 JPEG 字节；先解压为内存缓冲区，再交给
        # OpenCV 解码，整个过程不需要在磁盘上创建临时图片文件。
        # 每次只读取当前样本的压缩字节，避免把全部图像一次性展开到内存。
        raw = self.archive.read(row["member"])
        try:
            import cv2  # type: ignore

            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV 无法解码 ZIP 中的图像")
            # OpenCV 默认返回 BGR，模型输入约定为 RGB；随后统一缩放为
            # C×H×W、float32、[0, 1]，与 EnNeuro Tensor 接口保持一致。
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        except ImportError as exc:
            raise RuntimeError("车辆 ZIP 执行需要安装 opencv-python") from exc
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        target = np.asarray([row["steering"]], dtype=np.float32)
        return Tensor(image), Tensor(target)


def read_steering_rows(archive_path: str | Path, *, camera: str | Sequence[str] = "center",
                       max_samples: int | None = None, camera_correction: float = 0.2) -> list[dict[str, Any]]:
    """读取并校验 driving_log.csv，但不解码图像。

    参数中的 `camera` 可以是单个摄像头名称，也可以是多个名称。left/right
    图像通常需要相对 center 图像施加转向修正，这里分别加上和减去
    `camera_correction`，并把标签裁剪到 [-1, 1]。
    """

    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        csv_name = next((name for name in ("data/driving_log.csv", "driving_log.csv") if name in names), None)
        if csv_name is None:
            raise FileNotFoundError("driving_log.csv not found in archive")
        # 先把单值参数规范化成元组，后续逻辑就能统一处理单路和三路数据。
        # 统一成 tuple 后，单路 center 和三路 camera 可以复用同一套解析逻辑。
        cameras = (camera,) if isinstance(camera, str) else tuple(camera)
        if not cameras or any(c not in {"center", "left", "right"} for c in cameras):
            raise ValueError("camera must be center, left, right, or a sequence of those names")
        rows: list[dict[str, Any]] = []
        with archive.open(csv_name, "r") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            for raw_row in csv.DictReader(text):
                try:
                    base_steering = float((raw_row.get("steering") or "").strip())
                except ValueError:
                    # 个别坏行不应使整个搜索任务失败，跳过即可。
                    continue
                # 一条驾驶记录最多扩展为三条样本；left/right 的标签在下面做修正。
                for selected_camera in cameras:
                    image_ref = (raw_row.get(selected_camera) or "").strip()
                    if not image_ref:
                        continue
                    member = image_ref.replace("\\", "/")
                    if not member.startswith("data/"):
                        member = "data/" + member
                    if member not in names:
                        continue
                    correction = 0.0
                    if selected_camera == "left":
                        correction = float(camera_correction)
                    elif selected_camera == "right":
                        correction = -float(camera_correction)
                    # 保存 camera 字段便于报告和后续诊断；训练本身只使用
                    # member 和 steering 两个字段。
                    # 保留 camera 字段方便报告和排查，训练本身主要使用 member/steering。
                    rows.append({
                        "member": member,
                        "steering": float(np.clip(base_steering + correction, -1.0, 1.0)),
                        "camera": selected_camera,
                    })
                    if max_samples is not None and len(rows) >= max_samples:
                        break
                if max_samples is not None and len(rows) >= max_samples:
                    break
    if not rows:
        raise ValueError("no valid steering rows found in archive")
    return rows


class TinySteeringRegressor:
    """用于完整 ZIP 冒烟测试的小型 EnNeuro 回归模型。

    首轮端到端验收的重点是验证调参 agent 的闭环，而不是比较复杂 CNN 的
    精度。这里使用一个延迟初始化的 Linear 头直接回归转向角，既能真正
    调用 EnNeuro Trainer，又能把每个 trial 的运行时间控制在可接受范围内。
    将来可以在相同 Executor 接口中替换为更大的 CNN。
    """

    def __new__(cls):
        # Linear 是延迟初始化的，第一次前向时会根据展平后的图像维度创建权重。
        from eneuro.base import functions as F
        from eneuro.nn.module import Conv2d, Linear, Module

        class _Model(Module):
            def __init__(self):
                super().__init__()
                self.fc = Linear(1)

            def forward(self, x):
                x = F.flatten(x)
                return self.fc(x)

        return _Model()


class CarBehavioralCloningExecutor:
    """在车辆 ZIP 数据上执行一次真实的 EnNeuro 回归 trial。

    AutoML 搜索器只需要调用 `execute(config)`。执行器内部负责固定随机
    种子、读取 CSV、建立时间顺序划分、训练模型并返回验证 MSE；同时把
    测试 MSE/MAE、样本数量和耗时写入结果，供最终报告使用。
    """

    def __init__(self, archive_path: str | Path, *, epochs: int = 1,
                 seed: int = 0, device: str = "cpu", backend: str = "auto",
                 image_size: tuple[int, int] = (64, 32), max_samples: int | None = None,
                 cameras: Sequence[str] = ("center",), camera_correction: float = 0.2,
                 train_ratio: float = 0.8, val_ratio: float = 0.1):
        if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio and val_ratio must leave a test split")
        self.archive_path = Path(archive_path)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = str(device).lower()
        self.backend = str(backend).lower()
        self.image_size = image_size
        self.max_samples = max_samples
        self.cameras = tuple(cameras)
        self.camera_correction = float(camera_correction)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.calls = 0

    def execute(self, config: Mapping[str, Any]) -> TrialResult:
        """执行一个车辆回归 trial；异常会转换成失败结果交给搜索器处理。"""
        # 每个 trial 使用可预测但不同的种子，使重复运行同一搜索仍可复现，
        # 同时避免所有 trial 使用完全相同的初始化噪声。
        trial_seed = self.seed + self.calls
        self.calls += 1
        try:
            return self._execute(config, trial_seed)
        except Exception as exc:
            return TrialResult(float("inf"), status="failed",
                               error=f"{type(exc).__name__}: {exc}")

    def _execute(self, config: Mapping[str, Any], trial_seed: int) -> TrialResult:
        from eneuro.base import Config, as_Tensor
        from eneuro.data import DataLoader
        from eneuro.nn.loss import meanSquaredError
        from eneuro.nn.optim import Adam, MomentumSGD, SGD
        from eneuro.train.trainer import Trainer

        np.random.seed(trial_seed)
        random.seed(trial_seed)
        if self.device in {"cuda", "gpu"}:
            try:
                from eneuro.base.cuda import set_backend

                if self.backend != "auto":
                    # backend=auto 保留 EnNeuro 默认选择；显式指定时才切换
                    # cupy/rawmodule，避免执行器改变调用方的全局偏好。
                    set_backend(self.backend)
            except Exception:
                # CUDA 后端不可用时让实际训练路径给出失败结果，搜索器会把
                # 该 trial 记为 failed，而不是让整个调参进程直接退出。
                pass

        # 先解析索引和标签，真正的 JPEG 解码延迟到 DataLoader 请求样本时。
        rows = read_steering_rows(
            self.archive_path,
            camera=self.cameras,
            max_samples=self.max_samples,
            camera_correction=self.camera_correction,
        )
        n = len(rows)
        train_end = max(1, int(n * self.train_ratio))
        val_end = max(train_end + 1, int(n * (self.train_ratio + self.val_ratio)))
        # 按 CSV 原始时间顺序切分，避免相邻视频帧被随机打散到不同集合。
        # 末尾的边界保护保证小数据集至少能留下验证和测试样本。
        # 按 CSV 原始顺序切分，避免相邻视频帧随机泄漏到不同集合。
        train_indices = range(0, min(train_end, n - 2))
        val_indices = range(min(train_end, n - 2), min(val_end, n - 1))
        test_indices = range(min(val_end, n - 1), n)

        start = time.perf_counter()
        with zipfile.ZipFile(self.archive_path) as archive:
            train_ds = ZipSteeringDataset(archive, rows, list(train_indices), self.image_size)
            val_ds = ZipSteeringDataset(archive, rows, list(val_indices), self.image_size)
            test_ds = ZipSteeringDataset(archive, rows, list(test_indices), self.image_size)
            batch_size = int(config["batch_size"])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            model = TinySteeringRegressor().to(self.device)
            sample_x, _ = train_ds[0]
            # Linear 层是延迟初始化的，先用一个带 batch 维的样本完成形状
            # 推断，再创建 optimizer，保证 optimizer 能看到完整参数。
            sample_batch = as_Tensor(np.expand_dims(sample_x.data, axis=0)).to(self.device)
            model(sample_batch)
            optimizer_name = str(config.get("optimizer", "adam")).lower()
            optimizer_cls = {"adam": Adam, "sgd": SGD, "momentum_sgd": MomentumSGD}.get(optimizer_name)
            if optimizer_cls is None:
                raise ValueError(f"unknown optimizer: {optimizer_name}")
            optimizer = optimizer_cls(model.params(), lr=float(config["lr"]))
            trainer = Trainer(model, meanSquaredError, optimizer)

            train_mse = float("nan")
            val_mse = float("inf")
            val_mae = float("inf")
            for epoch in range(max(1, self.epochs)):
                # Trainer._one_step 返回当前 epoch 的训练损失；验证阶段单独
                # 调用 _evaluate，避免把验证数据参与反向传播。
                trainer._epoch = epoch
                train_mse, _ = trainer._one_step(train_loader, batch_size=batch_size,
                                                 training=True, verbose=False, device=self.device)
                val_mse, val_mae = self._evaluate(model, val_loader, meanSquaredError, self.device, Config)
            test_mse, test_mae = self._evaluate(model, test_loader, meanSquaredError, self.device, Config)

        return TrialResult(float(val_mse), {
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "val_mae": float(val_mae),
            "test_mse": float(test_mse),
            "test_mae": float(test_mae),
            "samples": n,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "elapsed_s": time.perf_counter() - start,
            "device": self.device,
            "image_size": list(self.image_size),
        })

    @staticmethod
    def _evaluate(model, loader, loss_fn, device: str, Config) -> tuple[float, float]:
        """在一个数据划分上计算平均 MSE 和 MAE。"""
        loss_sum = 0.0
        abs_sum = 0.0
        count = 0
        with Config.using_config("enable_backprop", False), Config.using_config("train", False):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                # 只把最终的预测和标签搬回 CPU 做统计；训练计算仍留在
                # 原设备上，避免每个 batch 产生不必要的数据拷贝。
                pred_np = _as_numpy(pred.data).reshape(-1)
                y_np = _as_numpy(yb.data).reshape(-1)
                n = len(y_np)
                loss_sum += float(_as_numpy(loss.data)) * n
                abs_sum += float(np.abs(pred_np - y_np).sum())
                count += n
        return loss_sum / max(count, 1), abs_sum / max(count, 1)

