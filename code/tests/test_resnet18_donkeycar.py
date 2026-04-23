# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
import ctypes
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


if os.name == 'nt':
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
            ('PrivateUsage', ctypes.c_size_t),
        ]


sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import as_Tensor, Config  # noqa: E402
from eneuro.base import Tensor  # noqa: E402
from eneuro.base import functions as F  # noqa: E402
from eneuro.data import Dataset, DataLoader  # noqa: E402
from eneuro.nn.loss import meanSquaredError  # noqa: E402
from eneuro.nn.module import Module, Conv2d, BatchNorm, Linear, ResidualBlock, Sequential  # noqa: E402
from eneuro.nn.optim import Adam  # noqa: E402
from eneuro.train.trainer import Trainer  # noqa: E402
from eneuro.utils import save_checkpoint, load_checkpoint  # noqa: E402


def _get_cpu_time_seconds_fallback():
    if os.name != 'nt':
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    ok = kernel32.GetProcessTimes(
        handle,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    )
    if not ok:
        return None

    def _to_100ns(ft):
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    return (_to_100ns(kernel) + _to_100ns(user)) / 1e7


def _get_memory_info_fallback():
    if os.name != 'nt':
        return (None, None)
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return (None, None)
    return (int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))


def _memory_mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def collect_process_metrics(start_wall, start_cpu_time, start_rss, fallback_proc_time_start=None, fallback_trace_started=False):
    wall = time.perf_counter() - start_wall
    result = {
        'wall_s': wall,
        'cpu_avg_percent': None,
        'mem_peak_mb': None,
    }

    if psutil is None:
        end_cpu_time = _get_cpu_time_seconds_fallback()
        if start_cpu_time is not None and end_cpu_time is not None and wall > 0:
            logical_cores = os.cpu_count() or 1
            result['cpu_avg_percent'] = ((end_cpu_time - start_cpu_time) / wall) * 100.0 / logical_cores
        elif fallback_proc_time_start is not None and wall > 0:
            logical_cores = os.cpu_count() or 1
            cpu_sec = time.process_time() - fallback_proc_time_start
            result['cpu_avg_percent'] = (cpu_sec / wall) * 100.0 / logical_cores

        rss_now, peak_rss = _get_memory_info_fallback()
        if peak_rss is not None:
            result['mem_peak_mb'] = _memory_mb(peak_rss)
        elif start_rss is not None and rss_now is not None:
            result['mem_peak_mb'] = _memory_mb(max(start_rss, rss_now))
        elif fallback_trace_started and tracemalloc.is_tracing():
            _, peak_bytes = tracemalloc.get_traced_memory()
            result['mem_peak_mb'] = _memory_mb(peak_bytes)
            tracemalloc.stop()

        return result

    proc = psutil.Process(os.getpid())
    end_cpu = proc.cpu_times()
    cpu_time_delta = (end_cpu.user + end_cpu.system) - start_cpu_time
    logical_cores = psutil.cpu_count(logical=True) or 1
    if wall > 0:
        result['cpu_avg_percent'] = (cpu_time_delta / wall) * 100.0 / logical_cores

    mem_info = proc.memory_info()
    peak_bytes = getattr(mem_info, 'peak_wset', None)
    if peak_bytes is None:
        peak_bytes = max(mem_info.rss, start_rss)
    result['mem_peak_mb'] = _memory_mb(peak_bytes)
    return result


class ResNet18Steering(Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=3)
        self.bn1 = BatchNorm(64)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.fc = Linear(1)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        downsample = (stride != 1) or (self.in_channels != out_channels)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1, downsample=False))
        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.pooling(x, kernel_size=3, stride=2, pad=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.global_average_pooling(x)
        x = F.flatten(x)
        x = self.fc(x)
        return x


class DonkeycarDataset(Dataset):
    def __init__(self, x, y):
        self._x = x.astype(np.float32)
        self._y = y.astype(np.float32)
        super().__init__(train=True)

    def prepare(self):
        self.data = self._x
        self.label = self._y

    def __getitem__(self, index):
        return Tensor(self.data[index]), Tensor(np.array(self.label[index], dtype=np.float32))


def find_data_dir(explicit_dir=None):
    if explicit_dir:
        p = Path(explicit_dir)
        if p.exists():
            return p
        raise FileNotFoundError("指定的数据目录不存在: {}".format(p))

    code_root = Path(__file__).resolve().parents[1]
    candidates = [
        code_root / 'tests' / 'testdata' / 'data',
        code_root / 'testdata' / 'data',
        code_root.parent / 'tests' / 'testdata' / 'data',
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("找不到数据目录，尝试路径:\n{}".format("\n".join(str(c) for c in candidates)))


def parse_angle_from_name(file_path):
    stem = file_path.stem
    if '_' not in stem:
        raise ValueError("文件名不符合 id_angle 格式: {}".format(file_path.name))
    _, angle_str = stem.rsplit('_', 1)
    return float(angle_str)


def load_donkeycar_images(data_dir, image_size=64):
    image_paths = sorted(list(data_dir.glob('*.jpg')))
    if not image_paths:
        raise ValueError("目录中未找到jpg图像: {}".format(data_dir))

    xs = []
    ys = []

    for p in image_paths:
        try:
            angle = parse_angle_from_name(p)
        except Exception:
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        xs.append(img)
        ys.append(np.float32(angle))

    if not xs:
        raise ValueError("没有成功解析任何图像样本")

    x = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    return x, y


def split_dataset(x, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("划分比例需满足: train_ratio>0, val_ratio>0, train_ratio+val_ratio<1")

    n = len(x)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    return (
        (x[train_idx], y[train_idx]),
        (x[val_idx], y[val_idx]),
        (x[test_idx], y[test_idx]),
    )


def count_learnable_params(model):
    total = 0
    for p in model.params():
        if getattr(p, 'requires_grad', True) and getattr(p, 'data', None) is not None:
            total += int(np.prod(p.data.shape))
    return total


def train_one_epoch(model, optimizer, loader):
    mse_sum = 0.0
    sample_num = 0

    with Config.using_config('train', True):
        for xb, yb in loader:
            pred = model(as_Tensor(xb))
            loss = meanSquaredError(pred, yb)

            model.cleargrads()
            loss.backward()
            optimizer.step()

            batch_n = len(yb)
            mse_sum += float(loss.data) * batch_n
            sample_num += batch_n

    return mse_sum / max(sample_num, 1)


def evaluate(model, loader):
    mse_sum = 0.0
    mae_sum = 0.0
    sample_num = 0

    with Config.using_config('train', False):
        for xb, yb in loader:
            pred = model(as_Tensor(xb))
            loss = meanSquaredError(pred, yb)

            pred_np = pred.data.reshape(-1)
            yb_np = yb.data.reshape(-1)
            batch_n = len(yb)
            mse_sum += float(loss.data) * batch_n
            mae_sum += float(np.abs(pred_np - yb_np).sum())
            sample_num += batch_n

    mse = mse_sum / max(sample_num, 1)
    mae = mae_sum / max(sample_num, 1)
    return mse, mae


def main():
    default_ckpt = Path(__file__).resolve().parent / 'checkpoints' / 'resnet18_donkeycar_checkpoint.json'
    default_best_ckpt = Path(__file__).resolve().parent / 'checkpoints' / 'resnet18_donkeycar_best_checkpoint.json'

    parser = argparse.ArgumentParser(description='DonkeyCar图像转向角回归训练 (ResNet18)')
    parser.add_argument('--data-dir', type=str, default='D:/Data/data', help='数据目录，默认自动查找')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--max-samples', type=int, default=0, help='限制样本量以快速启动训练，<=0表示全量')
    parser.add_argument('--save-checkpoint', type=str, default=str(default_ckpt), help='训练结束后保存checkpoint路径')
    parser.add_argument('--best-checkpoint', type=str, default=str(default_best_ckpt), help='验证集最优模型checkpoint路径')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='从已有checkpoint加载并继续训练')
    parser.add_argument('--save-each-epoch', action='store_true', help='每个epoch结束后都保存checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = find_data_dir(args.data_dir)
    print('[INFO]使用数据目录:', data_dir)

    x_all, y_all = load_donkeycar_images(data_dir, image_size=args.image_size)
    print('[INFO]加载样本数:', len(x_all))

    if args.max_samples > 0 and len(x_all) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        chosen = rng.choice(len(x_all), size=args.max_samples, replace=False)
        x_all = x_all[chosen]
        y_all = y_all[chosen]
        print('[INFO]为快速训练抽样样本数:', len(x_all))

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split_dataset(
        x_all,
        y_all,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print('[INFO] 数据划分: train={}, val={}, test={}'.format(len(x_tr), len(x_va), len(x_te)))

    train_loader = DataLoader(DonkeycarDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DonkeycarDataset(x_va, y_va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DonkeycarDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = ResNet18Steering()
    optimizer = Adam(model.params(), lr=args.lr)
    trainer = Trainer(model, meanSquaredError, optimizer)

    resumed_epoch = 0
    if args.load_checkpoint:
        load_path = Path(args.load_checkpoint)
        if not load_path.exists():
            raise FileNotFoundError('checkpoint不存在: {}'.format(load_path))
        resumed_epoch = load_checkpoint(str(load_path), model, optimizer)
        print('[INFO] 已加载checkpoint: {} (epoch={})'.format(load_path, resumed_epoch))

    learnable_params = count_learnable_params(model)
    print('[METRIC] learnable_params={}'.format(learnable_params))

    proc = psutil.Process(os.getpid()) if psutil is not None else None
    fallback_proc_time_start = time.process_time()
    fallback_trace_started = False
    if proc is not None:
        start_cpu = proc.cpu_times()
        start_cpu_time = start_cpu.user + start_cpu.system
        start_rss = proc.memory_info().rss
    else:
        start_cpu_time = _get_cpu_time_seconds_fallback()
        start_rss, _ = _get_memory_info_fallback()
        if start_rss is None:
            tracemalloc.start()
            fallback_trace_started = True

    start_wall = time.perf_counter()

    total_target_epoch = resumed_epoch + args.epochs
    best_val_mse = float('inf')
    best_epoch = resumed_epoch
    for local_epoch in range(args.epochs):
        epoch = resumed_epoch + local_epoch
        trainer._epoch = epoch
        tr_mse, _ = trainer._one_step(
            train_loader,
            batch_size=args.batch_size,
            training=True,
            verbose=False,
            device='cpu',
        )
        va_mse, va_mae = evaluate(model, val_loader)
        print('[EPOCH {}/{}] train_mse={:.6f} val_mse={:.6f} val_mae={:.6f}'.format(
            epoch + 1,
            total_target_epoch,
            tr_mse,
            va_mse,
            va_mae,
        ))

        if args.best_checkpoint and va_mse < best_val_mse:
            best_val_mse = va_mse
            best_epoch = epoch + 1
            best_path = Path(args.best_checkpoint)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, str(best_path))
            print('[INFO] 已保存best checkpoint: {} (val_mse={:.6f})'.format(best_path, va_mse))

        if args.save_each_epoch and args.save_checkpoint:
            save_path = Path(args.save_checkpoint)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, str(save_path))
            print('[INFO] 已保存checkpoint:', save_path)

    te_mse, te_mae = evaluate(model, test_loader)
    metrics = collect_process_metrics(
        start_wall,
        start_cpu_time,
        start_rss,
        fallback_proc_time_start=fallback_proc_time_start,
        fallback_trace_started=fallback_trace_started,
    )

    print('[RESULT] test_mse={:.6f} test_mae={:.6f}'.format(te_mse, te_mae))
    print('[METRIC] wall_time_s={:.4f}'.format(metrics['wall_s']))
    if metrics['mem_peak_mb'] is not None:
        print('[METRIC] peak_memory_mb={:.2f}'.format(metrics['mem_peak_mb']))
    else:
        print('[METRIC] peak_memory_mb=N/A')
    if metrics['cpu_avg_percent'] is not None:
        print('[METRIC] cpu_avg_percent={:.2f}'.format(metrics['cpu_avg_percent']))
    else:
        print('[METRIC] cpu_avg_percent=N/A')
    if args.best_checkpoint and best_val_mse < float('inf'):
        print('[METRIC] best_val_mse={:.6f} best_epoch={}'.format(best_val_mse, best_epoch))

    if args.save_checkpoint:
        save_path = Path(args.save_checkpoint)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, resumed_epoch + args.epochs, str(save_path))
        print('[INFO] 已保存checkpoint:', save_path)


if __name__ == '__main__':
    main()
