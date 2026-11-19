"""Train EnNeuro steering regression on a Udacity driving_log.csv dataset."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eneuro.base import Config, Tensor, as_Tensor  # noqa: E402
from eneuro.base.cuda import set_backend  # noqa: E402
from eneuro.data import Dataset, DataLoader  # noqa: E402
from eneuro.nn.loss import meanSquaredError  # noqa: E402
from eneuro.nn.optim import Adam  # noqa: E402
from eneuro.utils import load_checkpoint, save_checkpoint  # noqa: E402
from test_resnet18_donkeycar import ResNet18Steering  # noqa: E402


def read_log(data_dir):
    csv_path = data_dir / 'driving_log.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    rows = []
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'center', 'left', 'right', 'steering'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{csv_path} needs columns: {sorted(required)}')
        for line, row in enumerate(reader, start=2):
            angle = float(row['steering'].strip())
            if not np.isfinite(angle):
                raise ValueError(f'non-finite steering value at CSV line {line}')
            paths = []
            for camera in ('center', 'left', 'right'):
                path = Path(row[camera].strip())
                if not path.is_file():
                    path = data_dir / path.name
                    if not path.is_file():
                        path = data_dir / 'IMG' / path.name
                if not path.is_file():
                    raise FileNotFoundError(f'CSV line {line}: {camera}: {row[camera]}')
                paths.append(path)
            rows.append((paths, angle))
    if len(rows) < 10:
        raise ValueError('at least 10 complete driving_log.csv rows are required')
    return rows


class DrivingDataset(Dataset):
    def __init__(self, entries, height, width, augment=False):
        self.entries = entries
        self.height = height
        self.width = width
        self.augment = augment
        super().__init__(train=augment)

    def prepare(self):
        self.data = self.entries

    def __getitem__(self, index):
        path, angle = self.data[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'cannot decode image: {path}')
        # Remove most of the sky and the car hood while retaining the road.
        image = image[50:140]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
            angle = -angle
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        target = np.asarray([angle], dtype=np.float32)
        return Tensor(image), Tensor(target)


def make_entries(rows, correction):
    entries = []
    for paths, angle in rows:
        entries.extend((
            (paths[0], angle),
            (paths[1], float(np.clip(angle + correction, -1.0, 1.0))),
            (paths[2], float(np.clip(angle - correction, -1.0, 1.0))),
        ))
    return entries


def run_epoch(model, loader, optimizer=None, device='cuda'):
    training = optimizer is not None
    total_squared = 0.0
    total_absolute = 0.0
    total = 0
    context = Config.using_config('train', training)
    with context:
        for batch_index, (xb, yb) in enumerate(loader, start=1):
            xb = as_Tensor(xb).to(device)
            yb = as_Tensor(yb).to(device)
            if training:
                pred = model(xb)
                loss = meanSquaredError(pred, yb)
                model.cleargrads()
                loss.backward()
                optimizer.step()
            else:
                with Config.no_grad():
                    pred = model(xb)
                    loss = meanSquaredError(pred, yb)
            count = len(yb)
            total_squared += float(loss.data) * count
            if not training:
                diff = pred.data - yb.data
                total_absolute += float(abs(diff).sum())
            total += count
            if training and (batch_index == 1 or batch_index % 100 == 0):
                print(f'[BATCH] {batch_index}/{len(loader)} mse={total_squared / total:.6f}', flush=True)
    return {'mse': total_squared / total, 'mae': total_absolute / total if not training else None}


def make_optimizer(model, lr):
    params = {}
    model._flatten_params(params)
    return Adam([params[name] for name in sorted(params)], lr=lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='folder containing driving_log.csv and IMG')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--camera-correction', type=float, default=0.2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--backend', choices=('cupy', 'rawmodule', 'auto'), default='rawmodule')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='checkpoint to resume; --epochs is the total target epoch count')
    parser.add_argument('--max-train-rows', type=int, default=0, help='for a short smoke run only')
    parser.add_argument('--max-eval-rows', type=int, default=0, help='for a short smoke run only')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.height < 16 or args.width < 16:
        parser.error('epochs and batch size must be positive; image dimensions must be >= 16')

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_backend(args.backend)
    rows = read_log(args.data_dir)
    n_train = int(len(rows) * 0.7)
    n_val = int(len(rows) * 0.15)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]
    if args.max_train_rows:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_eval_rows:
        val_rows = val_rows[:args.max_eval_rows]
        test_rows = test_rows[:args.max_eval_rows]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError('the train, validation and test splits must each have samples')
    train_entries = make_entries(train_rows, args.camera_correction)
    val_entries = [(paths[0], angle) for paths, angle in val_rows]
    test_entries = [(paths[0], angle) for paths, angle in test_rows]
    train_loader = DataLoader(DrivingDataset(train_entries, args.height, args.width, augment=True),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DrivingDataset(val_entries, args.height, args.width),
                            batch_size=args.batch_size)
    test_loader = DataLoader(DrivingDataset(test_entries, args.height, args.width),
                             batch_size=args.batch_size)

    model = ResNet18Steering().to(args.device)
    optimizer = make_optimizer(model, args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'data_dir': str(args.data_dir.resolve()), 'csv_rows': len(rows),
        'train_rows': len(train_rows), 'val_rows': len(val_rows), 'test_rows': len(test_rows),
        'train_images_per_epoch': len(train_entries), 'camera_correction': args.camera_correction,
        'crop_top': 50, 'crop_bottom': 140, 'height': args.height, 'width': args.width,
        'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': args.lr,
        'device': args.device, 'backend': args.backend, 'seed': args.seed,
        'split': 'chronological 70/15/15; train all cameras; evaluate center camera',
    }
    config['zero_baseline'] = {
        'val_mse': float(np.mean([angle * angle for _, angle in val_entries])),
        'val_mae': float(np.mean([abs(angle) for _, angle in val_entries])),
        'test_mse': float(np.mean([angle * angle for _, angle in test_entries])),
        'test_mae': float(np.mean([abs(angle) for _, angle in test_entries])),
    }
    print('[CONFIG] ' + json.dumps(config, ensure_ascii=False), flush=True)
    history = []
    best_val_mse = float('inf')
    resumed_epoch = 0
    if args.resume_from:
        resumed_epoch = load_checkpoint(str(args.resume_from), model, optimizer)
        # Old checkpoints index Adam buffers by an unordered parameter list.
        # The weights are keyed by name, but optimizer buffers cannot safely
        # be matched across processes. Continue with fresh Adam buffers.
        optimizer = make_optimizer(model, args.lr)
        metrics_path = args.output_dir / 'metrics.json'
        if metrics_path.is_file():
            history = json.loads(metrics_path.read_text(encoding='utf-8')).get('history', [])
            if history:
                best_val_mse = min(item['val_mse'] for item in history)
        if args.epochs <= resumed_epoch:
            parser.error(f'--epochs must exceed resumed epoch {resumed_epoch}')
        print(f'[RESUME] epoch={resumed_epoch} checkpoint={args.resume_from} optimizer=fresh_adam', flush=True)
    for epoch in range(resumed_epoch + 1, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, train_loader, optimizer=optimizer, device=args.device)
        val = run_epoch(model, val_loader, device=args.device)
        record = {'epoch': epoch, 'train_mse': train['mse'], 'val_mse': val['mse'],
                  'val_mae': val['mae'], 'duration_s': time.perf_counter() - start}
        history.append(record)
        print('[EPOCH] ' + json.dumps(record), flush=True)
        checkpoint = args.output_dir / 'latest.pkl'
        save_checkpoint(model, optimizer, epoch, str(checkpoint), model_config=config)
        if val['mse'] < best_val_mse:
            best_val_mse = val['mse']
            save_checkpoint(model, optimizer, epoch, str(args.output_dir / 'best.pkl'), model_config=config)
        (args.output_dir / 'metrics.json').write_text(
            json.dumps({'config': config, 'history': history}, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
    best_checkpoint = args.output_dir / 'best.pkl'
    best_epoch = min(history, key=lambda item: item['val_mse'])['epoch']
    load_checkpoint(str(best_checkpoint), model, optimizer)
    test = run_epoch(model, test_loader, device=args.device)
    report = {'config': config, 'history': history, 'best_epoch': best_epoch,
              'test_best_checkpoint': test}
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[TEST] ' + json.dumps(test), flush=True)


if __name__ == '__main__':
    main()
