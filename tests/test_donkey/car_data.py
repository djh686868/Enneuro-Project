"""读取 DonkeyCar 和 Car Behavioral Cloning 的转向角样本。"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import preprocess_image
from eneuro.agent import read_steering_rows


class SteeringDataset:
    """保存图像路径和转向值，由 DataLoader 按需解码。"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, angle = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert('RGB').resize((160, 120), Image.Resampling.BILINEAR)
            pixels = preprocess_image(np.asarray(rgb))
        return pixels, np.array([angle], dtype=np.float32)


def _read_donkeycar_samples(data_dir):
    samples = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg'}:
            continue
        frame, separator, label = path.stem.partition('_')
        if not separator or not frame.isdecimal():
            continue
        try:
            steering = float(label)
        except ValueError:
            continue
        if math.isfinite(steering):
            samples.append((int(frame), path, steering))
    samples.sort()
    return [(path, np.float32(steering)) for _, path, steering in samples]


def load_steering_dataset(data_dir, max_samples=0):
    """按采集顺序加载样本元数据，max_samples 非零时均匀抽样。"""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'找不到图像目录 {root}')
    if max_samples < 0:
        raise ValueError('max_samples 必须非负')

    csv_dir = root
    if not (root / 'driving_log.csv').is_file():
        csv_dir = root / 'data'
    if (csv_dir / 'driving_log.csv').is_file():
        rows = read_steering_rows(csv_dir, camera='center')
        samples = [(row['image_path'], np.float32(row['steering'])) for row in rows]
    else:
        samples = _read_donkeycar_samples(root)
        if not samples:
            raise ValueError(f'{root} 未找到 driving_log.csv 或形如 973_-0.1111.jpg 的图像')

    if 0 < max_samples < len(samples):
        indices = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
        samples = [samples[i] for i in indices]
    return SteeringDataset(samples)
