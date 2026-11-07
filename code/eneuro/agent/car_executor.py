"""ZIP-backed behavioral-cloning executor for the basic AutoML agent.

The executor deliberately keeps the dataset lazy: the archive is opened once
per trial and each image is decoded only when its batch is requested.  This
avoids materializing all 24k images in host memory while still evaluating the
complete camera streams selected from ``driving_log.csv``.
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


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except ImportError:
        pass
    return np.asarray(value)


class ZipSteeringDataset:
    """Lazy camera dataset backed directly by the supplied ZIP file."""

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
        raw = self.archive.read(row["member"])
        try:
            import cv2  # type: ignore

            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not decode image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        except ImportError as exc:
            raise RuntimeError("Car ZIP execution requires opencv-python") from exc
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        target = np.asarray([row["steering"]], dtype=np.float32)
        return Tensor(image), Tensor(target)


def read_steering_rows(archive_path: str | Path, *, camera: str | Sequence[str] = "center",
                       max_samples: int | None = None, camera_correction: float = 0.2) -> list[dict[str, Any]]:
    """Read and validate CSV rows without decoding image data."""

    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        csv_name = next((name for name in ("data/driving_log.csv", "driving_log.csv") if name in names), None)
        if csv_name is None:
            raise FileNotFoundError("driving_log.csv not found in archive")
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
                    continue
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
    """Small EnNeuro regressor used to keep full-archive runs practical.

    The first end-to-end agent test should exercise search and execution rather
    than spend most of its time compiling convolution kernels.  A single lazy
    linear head over resized images is therefore the default model; the ZIP
    reader and Trainer path remain the same as for a larger CNN.
    """

    def __new__(cls):
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
    """Run one real EnNeuro regression trial on the complete ZIP dataset."""

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
                    set_backend(self.backend)
            except Exception:
                pass

        rows = read_steering_rows(
            self.archive_path,
            camera=self.cameras,
            max_samples=self.max_samples,
            camera_correction=self.camera_correction,
        )
        n = len(rows)
        train_end = max(1, int(n * self.train_ratio))
        val_end = max(train_end + 1, int(n * (self.train_ratio + self.val_ratio)))
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
        loss_sum = 0.0
        abs_sum = 0.0
        count = 0
        with Config.using_config("enable_backprop", False), Config.using_config("train", False):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                pred_np = _as_numpy(pred.data).reshape(-1)
                y_np = _as_numpy(yb.data).reshape(-1)
                n = len(y_np)
                loss_sum += float(_as_numpy(loss.data)) * n
                abs_sum += float(np.abs(pred_np - y_np).sum())
                count += n
        return loss_sum / max(count, 1), abs_sum / max(count, 1)

