# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import time
import psutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def default_transform(x):
    return x.transpose(2, 0, 1)


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
        img = torch.from_numpy(img).float()
        label = self._default_target_transform(angle)
        label = torch.tensor(label, dtype=torch.long)
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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=20):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(blocks - 1):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def progress_bar(current, total, epoch, loss, acc, width=30):
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% ({current:5}/{total:5}) '
        f' | loss={loss:.4f} | acc={acc:.3f}'
    )
    sys.stdout.flush()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def print_gpu_mem():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU memory: allocated={allocated:.1f}MB, cached={cached:.1f}MB")


def train():
    print_gpu_mem()
    
    batch_size = 16
    num_workers = 2
    
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / 'data'
    
    dataset = SteeringDataset(
        root_dir=str(data_path),
        transform=default_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if images.size(0) != batch_size:
                break
            
            optimizer.zero_grad()
            y_pre = model(images)
            loss = loss_fn(y_pre, labels)
            loss.backward()
            optimizer.step()
            
            pred_classes = torch.argmax(y_pre, dim=1)
            batch_acc = (pred_classes == labels).float().mean().item()
            
            loss_sum += loss.item() * len(images)
            if not np.isnan(batch_acc):
                acc_sum += batch_acc * len(images)
                mse_sum += ((pred_classes - labels)**2).float().mean().item() * len(images)
            sample_num += len(images)
            
            display_acc = (acc_sum / sample_num) if sample_num > 0 and not np.isnan(acc_sum) else np.nan
            progress_bar(batch_idx * batch_size + len(images), len(dataset), epoch, loss.item(), display_acc)
            
            current_mem = psutil.Process().memory_info().rss / 1024**2
            mem_peak = max(mem_peak, current_mem)
            
            if torch.cuda.is_available():
                gpu_mem_history.append(torch.cuda.memory_allocated() / 1024**2)
        
        toc = time.time()
        duration = toc - tic
        print(f"\nEpoch completed in {duration:.4f}s\n")
    
    print_gpu_mem()
    
    gpu_peak = max(gpu_mem_history) if gpu_mem_history else 0
    gpu_avg = np.mean(gpu_mem_history) if gpu_mem_history else 0
    
    avg_mse = mse_sum / sample_num if sample_num > 0 else 0.
    
    result = {
        'framework': 'PyTorch',
        'duration': duration,
        'params': total_params,
        'mem_peak': mem_peak,
        'gpu_peak': gpu_peak,
        'gpu_avg': gpu_avg,
        'avg_mse': avg_mse,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'num_samples': len(dataset)
    }
    
    results_dir = current_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'benchmark_pytorch.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print("\nBenchmark results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


if __name__ == '__main__':
    train()