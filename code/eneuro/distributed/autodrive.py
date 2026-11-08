"""Offline steering-angle data and a PS-compatible AutoDriveNet variant.

The layer widths follow serving.AutoDriveNet. BatchNorm is omitted because the
synchronous PS does not synchronize its per-worker running statistics. Every
weight is initialized in the constructor so the PS can validate the manifest.
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ..base import functions as F
from ..nn.module import Conv2d, Linear, Module
from ..nn.optim import SGD


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / 'tests' / 'testdata' / 'data'
IMAGE_SIZE = (160, 120)  # Pillow uses width, height.


class SteeringDataset:
    """Read center-camera images lazily; each label has shape (1,) for MSE."""

    def __init__(self, data_dir=DEFAULT_DATA_DIR, limit=None):
        root = Path(data_dir)
        with (root / 'driving_log.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if limit is not None:
            if limit < 1 or limit > len(rows):
                raise ValueError('limit must be within the driving log')
            # Cover the full recorded drive while keeping a fixed-size benchmark.
            rows = [rows[i] for i in np.linspace(0, len(rows) - 1, limit, dtype=int)]
        self.samples = [(root / row['center'].strip(), np.float32(row['steering']))
                        for row in rows]
        if not self.samples:
            raise ValueError('driving log has no samples')
        missing = next((path for path, _ in self.samples if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(missing)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, steering = self.samples[index]
        with Image.open(path) as image:
            resized = image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
        return pixels.transpose(2, 0, 1), np.asarray([steering], dtype=np.float32)


class PSAutoDriveNet(Module):
    """Five-convolution steering regressor based on serving.AutoDriveNet."""

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(24, kernel_size=5, stride=2, in_channels=3)
        self.conv2 = Conv2d(36, kernel_size=5, stride=2, in_channels=24)
        self.conv3 = Conv2d(48, kernel_size=5, stride=2, in_channels=36)
        self.conv4 = Conv2d(64, kernel_size=3, stride=1, in_channels=48)
        self.conv5 = Conv2d(64, kernel_size=3, stride=1, in_channels=64)
        # Spatial sizes: 120x160 -> 58x78 -> 27x37 -> 12x17 -> 10x15 -> 8x13.
        self.fc1 = Linear(100, in_size=64 * 8 * 13)
        self.fc2 = Linear(50, in_size=100)
        self.fc3 = Linear(10, in_size=50)
        self.fc4 = Linear(1, in_size=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def make_autodrive_model():
    np.random.seed(42)
    return PSAutoDriveNet()


def make_steering_dataset(limit=None, data_dir=DEFAULT_DATA_DIR):
    return SteeringDataset(data_dir, limit)


def make_optimizer(parameters):
    return SGD(parameters, lr=0.01)
