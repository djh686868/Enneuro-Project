# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import cupy as cp  # type: ignore
    has_cupy = True
except ImportError:
    cp = None
    has_cupy = False

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import as_Tensor, Config  # noqa: E402
from eneuro.base import Tensor  # noqa: E402
from eneuro.base import functions as F  # noqa: E402
from eneuro.base.cuda import set_backend, launch_counts  # noqa: E402
from eneuro.data import Dataset, DataLoader  # noqa: E402
from eneuro.nn.loss import meanSquaredError  # noqa: E402
from eneuro.nn.module import Module, Conv2d, BatchNorm, Linear, ResidualBlock, Sequential  # noqa: E402
from eneuro.nn.optim import Adam  # noqa: E402
from eneuro.profiling import process_snapshot  # noqa: E402
from eneuro.train.trainer import Trainer  # noqa: E402
from eneuro.utils import save_checkpoint, load_checkpoint  # noqa: E402


def _resolve_device(requested):
    device = (requested or 'cpu').lower()
    if device in ('cuda', 'gpu'):
        if not has_cupy:
            print('[WARN] CuPy 未安装，已自动回退到 CPU 运行。')
            return 'cpu'
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            print('[WARN] GPU 不可用，已自动回退到 CPU 运行。原因: {}'.format(exc))
            return 'cpu'
        if gpu_count <= 0:
            print('[WARN] 未检测到可用 GPU，已自动回退到 CPU 运行。')
            return 'cpu'
    return device


def _to_numpy(x):
    if has_cupy and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _backend_name(arr):
    if has_cupy and isinstance(arr, cp.ndarray):
        return 'cupy'
    return 'numpy'


def _device_label(device):
    if device in ('cuda', 'gpu'):
        return 'gpu'
    return 'cpu'


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
        # ``astype`` defaults to copy=True even when the source is already
        # float32.  For 6200 RGB images at 256x256 that needless copy is
        # several GiB per split and can push the process into paging.
        self._x = np.asarray(x, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
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


def evaluate(model, loader, device='cpu'):
    mse_sum = 0.0
    mae_sum = 0.0
    sample_num = 0

    with Config.using_config('train', False):
        for xb, yb in loader:
            xb = as_Tensor(xb).to(device)
            yb = as_Tensor(yb).to(device)
            pred = model(xb)
            loss = meanSquaredError(pred, yb)

            pred_np = _to_numpy(pred.data).reshape(-1)
            yb_np = _to_numpy(yb.data).reshape(-1)
            batch_n = len(yb)
            mse_sum += float(loss.data) * batch_n
            mae_sum += float(np.abs(pred_np - yb_np).sum())
            sample_num += batch_n

    mse = mse_sum / max(sample_num, 1)
    mae = mae_sum / max(sample_num, 1)
    return mse, mae


def main():
    # Binary checkpoints avoid JSON list expansion (multi-GiB for a full
    # ResNet18 + Adam state) and are atomic on interruption.
    default_ckpt = Path(__file__).resolve().parent / 'checkpoints' / 'resnet18_donkeycar_checkpoint.pkl'
    default_best_ckpt = Path(__file__).resolve().parent / 'checkpoints' / 'resnet18_donkeycar_best_checkpoint.pkl'

    parser = argparse.ArgumentParser(description='DonkeyCar图像转向角回归训练 (ResNet18)')
    parser.add_argument('--data-dir', type=str, default=r'D:\Enneuro\tests\testdata\data', help='数据目录，默认自动查找')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--max-samples', type=int, default=0, help='限制样本量以快速启动训练，<=0表示全量')
    parser.add_argument('--save-checkpoint', type=str, default=str(default_ckpt), help='训练结束后保存checkpoint路径')
    parser.add_argument('--best-checkpoint', type=str, default=str(default_best_ckpt), help='验证集最优模型checkpoint路径')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='从已有checkpoint加载并继续训练')
    parser.add_argument('--save-each-epoch', action='store_true', help='每个epoch结束后都保存checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备: cpu/cuda')
    parser.add_argument('--backend', type=str, default=os.environ.get('ENNEURO_CUDA_BACKEND', 'cupy'), choices=('cupy', 'rawmodule', 'auto'), help='CUDA算子后端')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_backend(args.backend)
    print('[INFO] cuda_backend={}'.format(args.backend))

    np.random.seed(args.seed)

    device = _resolve_device(args.device)

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

    # The three split arrays own their storage.  Release the original stacked
    # dataset before model construction so host memory is not kept at roughly
    # two copies of the complete image set for the entire training run.
    del x_all, y_all

    train_loader = DataLoader(DonkeycarDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DonkeycarDataset(x_va, y_va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DonkeycarDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = ResNet18Steering().to(device)
    optimizer = Adam(model.params(), lr=args.lr)
    trainer = Trainer(model, meanSquaredError, optimizer)

    first_param = next(model.params(), None)
    if first_param is not None and getattr(first_param, 'data', None) is not None:
        print('[INFO] model_param_backend={}'.format(_backend_name(first_param.data)))
    sample_x, _ = train_loader.dataset[0]
    print('[INFO] sample_input_backend={}'.format(_backend_name(sample_x.data)))
    print('[INFO] requested_device={} resolved_device={}'.format(args.device, device))

    resumed_epoch = 0
    if args.load_checkpoint:
        load_path = Path(args.load_checkpoint)
        if not load_path.exists():
            raise FileNotFoundError('checkpoint不存在: {}'.format(load_path))
        resumed_epoch = load_checkpoint(str(load_path), model, optimizer)
        print('[INFO] 已加载checkpoint: {} (epoch={})'.format(load_path, resumed_epoch))

    learnable_params = count_learnable_params(model)
    print('[METRIC] learnable_params={}'.format(learnable_params))

    before = process_snapshot()
    start_wall = time.perf_counter()

    total_target_epoch = resumed_epoch + args.epochs
    best_val_mse = float('inf')
    best_epoch = resumed_epoch
    for local_epoch in range(args.epochs):
        epoch = resumed_epoch + local_epoch
        trainer._epoch = epoch
        epoch_start = time.perf_counter()
        tr_mse, _ = trainer._one_step(
            train_loader,
            batch_size=args.batch_size,
            training=True,
            verbose=False,
            device=device,
        )
        tr_mse = float(tr_mse)
        va_mse, va_mae = evaluate(model, val_loader, device=device)
        epoch_duration = time.perf_counter() - epoch_start
        print('[EPOCH {}/{}] train_mse={:.6f} val_mse={:.6f} val_mae={:.6f} epoch_time_s={:.4f}'.format(
            epoch + 1,
            total_target_epoch,
            tr_mse,
            va_mse,
            va_mae,
            epoch_duration,
        ))

        # 注：已按要求注释掉保存当前最佳参数的逻辑
        # if args.best_checkpoint and va_mse < best_val_mse:
        #     best_val_mse = va_mse
        #     best_epoch = epoch + 1
        #     best_path = Path(args.best_checkpoint)
        #     best_path.parent.mkdir(parents=True, exist_ok=True)
        #     save_checkpoint(model, optimizer, epoch + 1, str(best_path))
        #     print('[INFO] 已保存best checkpoint: {} (val_mse={:.6f})'.format(best_path, va_mse))

        if args.save_each_epoch and args.save_checkpoint:
            save_path = Path(args.save_checkpoint)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, str(save_path))
            print('[INFO] 已保存checkpoint:', save_path)

    te_mse, te_mae = evaluate(model, test_loader, device=device)
    print('[METRIC] cuda_launch_counts={}'.format(launch_counts()))
    wall_s = time.perf_counter() - start_wall
    after = process_snapshot()
    cpu_time = after['cpu_time_s'] - before['cpu_time_s']
    peak_rss = after['rss_lifetime_peak_bytes']
    if peak_rss is None:
        peak_rss = max((rss for rss in (before['rss_bytes'], after['rss_bytes']) if rss is not None), default=None)

    device_label = _device_label(device)
    print('[RESULT] device={} test_mse={:.6f} test_mae={:.6f}'.format(device_label, te_mse, te_mae))
    print('[METRIC] wall_time_s={:.4f}'.format(wall_s))
    if peak_rss is not None:
        print('[METRIC] peak_memory_mb={:.2f}'.format(peak_rss / 1024 ** 2))
    else:
        print('[METRIC] peak_memory_mb=N/A')
    if wall_s > 0:
        cpu_percent = 100 * cpu_time / wall_s / (os.cpu_count() or 1)
        print('[METRIC] cpu_avg_percent={:.2f}'.format(cpu_percent))
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
