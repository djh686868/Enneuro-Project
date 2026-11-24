"""使用 Trainer 和异步 DataLoader 训练车辆转向角回归模型。"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))

from car_data import SteeringDataset, load_steering_dataset
from model import ResNet18AutoDrive
from eneuro.data import DataLoader
from eneuro.nn.loss import meanSquaredError
from eneuro.nn.optim import Adam
from eneuro.train import Trainer
from eneuro.utils.serializer import Serializer


class TimedDataLoader(DataLoader):
    """测量调用线程取批的等待时间及完整训练或验证阶段的耗时。"""

    def __init__(self, *args, synchronize=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.synchronize = synchronize

    def __iter__(self):
        self.samples_seen = 0
        self.batches_seen = 0
        self.wait_seconds = 0.0
        self.first_batch_seconds = 0.0
        phase_start = request_start = time.perf_counter()
        batches = super().__iter__()
        try:
            for images, labels in batches:
                batch_wait = time.perf_counter() - request_start
                if self.samples_seen == 0:
                    self.first_batch_seconds = batch_wait
                self.wait_seconds += batch_wait
                self.samples_seen += len(images)
                self.batches_seen += 1
                yield images, labels
                request_start = time.perf_counter()
        finally:
            batches.close()
        if self.synchronize is not None:
            self.synchronize()
        self.elapsed_seconds = time.perf_counter() - phase_start

    def timing_summary(self):
        samples_per_second = self.samples_seen / self.elapsed_seconds
        wait_percent = 100 * self.wait_seconds / self.elapsed_seconds
        later_wait = 1e3 * (self.wait_seconds - self.first_batch_seconds)
        return (f'{self.samples_seen} 张 / {self.elapsed_seconds:.2f} s = '
                f'{samples_per_second:.1f} 张/s，'
                f'取批等待 {self.wait_seconds:.3f} s（{wait_percent:.1f}%），'
                f'首批 {1e3 * self.first_batch_seconds:.1f} ms，'
                f'后续 {self.batches_seen - 1} 批累计等待 {later_wait:.1f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data',
                        help='DonkeyCar JPEG 目录，或包含 driving_log.csv 和 IMG/ 的目录或其父目录')
    parser.add_argument('--device', choices=['cpu', 'cupy', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2, help='后台线程数，0 为同步加载')
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=0, help='均匀抽样上限，0 为全部')
    parser.add_argument('--save-checkpoint', type=Path, help='训练结束后保存二进制 .pkl')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('轮数、批大小和学习率必须为正数')
    if args.workers < 0 or args.prefetch_factor < 1:
        parser.error('workers 必须非负，prefetch-factor 必须为正数')
    if not 0 <= args.seed < 2**32:
        parser.error('seed 须在 [0, 2**32) 内')
    if args.max_samples < 0 or args.max_samples == 1:
        parser.error('max-samples 须为 0 或至少 2')
    if args.save_checkpoint and args.save_checkpoint.suffix.lower() != '.pkl':
        parser.error('save-checkpoint 需要使用 .pkl 扩展名')
    try:
        dataset = load_steering_dataset(args.data_dir, args.max_samples)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if len(dataset) < 2:
        parser.error('训练和验证至少需要两张图像')

    random.seed(args.seed)
    np.random.seed(args.seed)
    synchronize = None
    device = args.device
    if device != 'cpu':
        try:
            import cupy as cp
        except ImportError:
            parser.error('GPU 训练需要安装 CuPy')
        if not cp.is_available():
            parser.error('当前环境未检测到可用的 CUDA 设备')
        cp.random.seed(args.seed)
        if device == 'cupy':
            from eneuro.base.cuda import set_backend
            set_backend('cupy')
            device = 'cuda'

        def synchronize():
            cp.cuda.get_current_stream().synchronize()

    # 连续驾驶帧高度相似，留末尾约 20% 作为验证段。
    split = len(dataset) * 4 // 5
    train_dataset = SteeringDataset(dataset.samples[:split])
    val_dataset = SteeringDataset(dataset.samples[split:])
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          prefetch_factor=args.prefetch_factor, synchronize=synchronize)
    train_loader = TimedDataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = TimedDataLoader(val_dataset, shuffle=False, **loader_options)

    model = ResNet18AutoDrive().to(device)
    optimizer = Adam(model.params(), lr=args.lr)
    train_loss_sum = 0.0
    epoch_started = time.perf_counter()

    def collect_train_loss(metrics):
        nonlocal train_loss_sum
        start = metrics['batch'] * args.batch_size
        count = min(args.batch_size, len(train_dataset) - start)
        loss = metrics['batch_loss']
        train_loss_sum += (math.nan if loss is None else loss) * count

    def report_epoch(metrics):
        nonlocal train_loss_sum, epoch_started
        val_mse = metrics['val_loss']
        elapsed = time.perf_counter() - epoch_started
        print(f"第 {metrics['epoch']}/{args.epochs} 轮 | "
              f"训练 MSE={train_loss_sum / len(train_dataset):.6f} | "
              f"验证 MSE={val_mse:.6f} | 验证 RMSE={math.sqrt(val_mse):.6f} | "
              f"{elapsed:.2f} 秒", flush=True)
        print(f'  训练 {train_loader.timing_summary()}\n'
              f'  验证 {val_loader.timing_summary()}', flush=True)
        train_loss_sum = 0.0
        epoch_started = time.perf_counter()

    trainer = Trainer(model, meanSquaredError, optimizer,
                      on_batch_end=collect_train_loss, on_epoch_end=report_epoch)
    print(f'设备={args.device}，加载线程={args.workers}，'
          f'训练={len(train_dataset)} 张，验证={len(val_dataset)} 张，'
          f'batch_size={args.batch_size}', flush=True)
    trainer.fit(train_loader, val_loader, epochs=args.epochs,
                batch_size=args.batch_size, verbose=False, device=device)

    if args.save_checkpoint:
        Serializer.save_checkpoint_binary(model, optimizer, args.epochs, args.save_checkpoint)
        print(f'已保存训练检查点 {args.save_checkpoint.resolve()}')


if __name__ == '__main__':
    main()
