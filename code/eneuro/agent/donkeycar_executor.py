"""在历史 DonkeyCar JPEG 数据上复用车辆回归训练、计时和外层评估。"""

from __future__ import annotations

from functools import cached_property
from math import isfinite
from pathlib import Path

from .car_executor import CarBehavioralCloningExecutor


class DonkeyCarSteeringExecutor(CarBehavioralCloningExecutor):
    """从 `数字ID_转向角.jpg` 读取原始转向角，用单输出 ResNet18 回归。"""

    def __init__(self, data_dir: str | Path, *, epochs: int = 1, refit_epochs: int | None = None,
                 seed: int = 0, device: str = "cuda", backend: str = "cupy", device_id: int = 0,
                 image_size: tuple[int, int] = (160, 120), max_samples: int | None = None):
        super().__init__(
            data_dir, epochs=epochs, refit_epochs=refit_epochs, seed=seed, device=device,
            backend=backend, device_id=device_id, image_size=image_size, max_samples=max_samples,
            train_ratio=0.70, val_ratio=0.15, model="resnet18",
        )
        self.target_names = ("steering",)

    @cached_property
    def _splits(self):
        # 历史数据把标签放在文件名末尾。保留原值，仅在加载 minibatch 时解码图像。
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"DonkeyCar 数据目录不存在：{self.data_dir}")
        rows = []
        for path in self.data_dir.glob("*.jpg"):
            if not path.is_file():
                continue
            try:
                image_id, angle = path.stem.rsplit("_", 1)
                group, steering = int(image_id), float(angle)
            except ValueError:
                continue
            if isfinite(steering):
                rows.append({"image_path": path, "steering": steering, "group": group})
        rows.sort(key=lambda row: (row["group"], row["image_path"].name))
        rows = rows[:self.max_samples]
        groups = sorted({row["group"] for row in rows})
        n = len(groups)
        if n < 3:
            raise ValueError(f"DonkeyCar 数据至少需要 3 个有效图像 ID：{self.data_dir}")

        # ID 相同的样本总在同一集合，相邻 ID 尽量放在连续的训练或留出段。
        train_end = min(max(1, int(n * self.train_ratio)), n - 2)
        val_end = min(max(train_end + 1, int(n * (self.train_ratio + self.val_ratio))), n - 1)
        train_last, val_last = groups[train_end - 1], groups[val_end - 1]
        train, val, test = [], [], []
        for i, row in enumerate(rows):
            group = row["group"]
            split = train if group <= train_last else val if group <= val_last else test
            split.append(i)
        return rows, train, val, test

    def split_metadata(self) -> dict:
        rows, train, val, test = self._splits
        splits = {}
        for name, indices in (("train", train), ("validation", val), ("test", test)):
            ids = {rows[i]["group"] for i in indices}
            splits[name] = {"images": len(indices), "image_ids": len(ids),
                            "first_image_id": min(ids), "last_image_id": max(ids)}
        return splits
