# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import time
import psutil
from pathlib import Path

import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from PIL import Image


class SteeringDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.images = []
        self.angles = []
        self.transform = transform
        self.prepare()

    def prepare(self):
        pattern = re.compile(r'^(\d+)_([-+]?\d*\.?\d+)\.jpg$')
        for fname in os.listdir(self.root_dir):
            if not fname.lower().endswith('.jpg'):
                continue
            match = pattern.match(fname)
            if not match:
                continue
            angle_str = match.group(2)
            try:
                angle = float(angle_str)
            except ValueError:
                continue
            if angle < -1.0 or angle > 1.0:
                continue
            img_path = self.root_dir / fname
            self.images.append(str(img_path))
            self.angles.append(angle)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        angle = self.angles[index]
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img, dtype=np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        img = paddle.to_tensor(img, dtype='float32')
        label = self._default_target_transform(angle)
        label = paddle.to_tensor(label, dtype='int64')
        return img, label

    @staticmethod
    def _default_target_transform(angle):
        num_classes = 20
        low, high = -1.0, 1.0
        bin_width = (high - low) / num_classes
        idx = int((angle - low) // bin_width)
        if idx == num_classes:
            idx = num_classes - 1
        idx = max(0, min(idx, num_classes - 1))
        return idx


class BasicBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(out_channels)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm2D(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet18(paddle.nn.Layer):
    def __init__(self, num_classes=20):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = paddle.nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(64)
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        
        self.avgpool = paddle.nn.AdaptiveAvgPool2D((1, 1))
        self.fc = paddle.nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = paddle.nn.Sequential(
                paddle.nn.Conv2D(self.in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                paddle.nn.BatchNorm2D(out_channels)
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return paddle.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


def progress_bar(current, total, epoch, loss, acc, width=30):
    percent = current / total
    filled = int(width * percent)
    bar = '=' * filled + '-' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% ({current:5}/{total:5}) '
        f' | loss={loss:.4f} | acc={acc:.3f}'
    )
    sys.stdout.flush()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def print_gpu_mem():
    if paddle.device.is_compiled_with_cuda():
        mem_info = paddle.device.cuda.memory_allocated()
        mem_reserved = paddle.device.cuda.memory_reserved()
        print(f"GPU memory: allocated={mem_info/1024**2:.1f}MB, reserved={mem_reserved/1024**2:.1f}MB")


def train():
    print_gpu_mem()
    
    batch_size = 16
    num_workers = 2
    
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / 'data'
    
    dataset = SteeringDataset(
        root_dir=str(data_path),
        transform=lambda x: x.transpose(2, 0, 1)
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0)
    
    use_cuda = paddle.device.is_compiled_with_cuda()
    device = paddle.device.set_device('gpu' if use_cuda else 'cpu')
    model = ResNet18(num_classes=20)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-4)
    loss_fn = paddle.nn.CrossEntropyLoss()
    
    total_params = count_params(model)
    print(f"Model parameters: {total_params:,}")
    
    mem_peak = 0
    gpu_mem_history = []
    
    num_epoch = 1
    
    for epoch in range(num_epoch):
        model.train()
        tic = time.time()
        loss_sum, acc_sum, mse_sum, sample_num = 0., 0, 0., 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if images.shape[0] != batch_size:
                break
            
            optimizer.clear_grad()
            y_pre = model(images)
            loss = loss_fn(y_pre, labels)
            loss.backward()
            optimizer.step()
            
            pred_classes = paddle.argmax(y_pre, axis=1)
            batch_acc = float((pred_classes == labels).astype('float32').mean().numpy())
            
            loss_sum += float(loss.numpy()) * len(images)
            if not np.isnan(batch_acc):
                acc_sum += batch_acc * len(images)
                mse_sum += float(((pred_classes - labels)**2).astype('float32').mean().numpy()) * len(images)
            sample_num += len(images)
            
            display_acc = (acc_sum / sample_num) if sample_num > 0 and not np.isnan(acc_sum) else np.nan
            progress_bar(batch_idx * batch_size + len(images), len(dataset), epoch, float(loss.numpy()), display_acc)
            
            current_mem = psutil.Process().memory_info().rss / 1024**2
            mem_peak = max(mem_peak, current_mem)
            
            if paddle.device.is_compiled_with_cuda():
                gpu_mem_history.append(paddle.device.cuda.memory_allocated() / 1024**2)
        
        toc = time.time()
        duration = toc - tic
        print(f"\nEpoch completed in {duration:.4f}s\n")
    
    print_gpu_mem()
    
    gpu_peak = max(gpu_mem_history) if gpu_mem_history else 0
    gpu_avg = np.mean(gpu_mem_history) if gpu_mem_history else 0
    
    avg_mse = mse_sum / sample_num if sample_num > 0 else 0.
    
    result = {
        'framework': 'PaddlePaddle',
        'duration': float(duration),
        'params': int(total_params),
        'mem_peak': float(mem_peak),
        'gpu_peak': float(gpu_peak),
        'gpu_avg': float(gpu_avg),
        'avg_mse': float(avg_mse),
        'num_workers': 0,
        'batch_size': int(batch_size),
        'num_samples': int(len(dataset))
    }
    
    results_dir = current_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'benchmark_paddle.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print("\nBenchmark results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


if __name__ == '__main__':
    train()