# -*- coding: utf-8 -*-
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])
    return img


class AutoDriveDatasetPyTorch(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode.lower()
        self.transform = transform
        assert self.mode in {"train", "val"}
        
        current_dir = Path(__file__).resolve().parent
        if self.mode == "train":
            file_path = str(current_dir / "train.txt")
        else:
            file_path = str(current_dir / "val.txt")
        
        self.file_list = []
        with open(file_path, "r") as f:
            files = f.readlines()
            for file in files:
                file = file.strip()
                if not file:
                    continue
                parts = file.split(" ")
                self.file_list.append([parts[0], float(parts[1])])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        img = torch.from_numpy(img).float()
        label = torch.tensor([self.file_list[index][1]], dtype=torch.float32)
        return img, label


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


class ResNet18AutoDrive(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18AutoDrive, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
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
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def progress_bar(current, total, epoch, loss, width=30):
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% ({current:5}/{total:5}) | loss={loss:.6f}'
    )
    sys.stdout.flush()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train():
    batch_size = 32
    total_epochs = 100
    lr = 1e-4
    num_workers = 2

    current_dir = Path(__file__).resolve().parent
    save_folder = str(current_dir / "results")
    os.makedirs(save_folder, exist_ok=True)

    train_dataset = AutoDriveDatasetPyTorch(mode="train", transform=preprocess_image)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = AutoDriveDatasetPyTorch(mode="val", transform=preprocess_image)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18AutoDrive(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    total_params = count_params(model)
    print(f"Model parameters: {total_params:,}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}, Epochs: {total_epochs}, LR: {lr}")

    train_loss_history = []
    val_loss_history = []
    epoch_times = []

    best_val_loss = float('inf')

    for epoch in range(total_epochs):
        model.train()
        tic = time.time()
        train_loss_sum = 0.0
        train_sample_num = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(images)
            train_sample_num += len(images)

            avg_loss = train_loss_sum / train_sample_num if train_sample_num > 0 else 0.0
            progress_bar(batch_idx * batch_size + len(images), len(train_dataset), epoch, avg_loss)

        toc = time.time()
        epoch_time = toc - tic
        epoch_times.append(epoch_time)

        avg_train_loss = train_loss_sum / train_sample_num if train_sample_num > 0 else 0.0
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_sample_num = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss_sum += loss.item() * len(images)
                val_sample_num += len(images)

        avg_val_loss = val_loss_sum / val_sample_num if val_sample_num > 0 else 0.0
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_folder, "model_pytorch.pt"))

        print(f"\nEpoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.4f}s")

    torch.save(model.state_dict(), os.path.join(save_folder, "model_pytorch_final.pt"))
    print(f"\nTraining completed. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_folder}/model_pytorch.pt")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Training Visualization (PyTorch - Regression)', fontsize=16)

    axes[0].plot(train_loss_history, label='Train Loss')
    axes[0].plot(val_loss_history, label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epoch_times)
    axes[1].set_title('Time Consumption per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].grid(True)

    axes[2].plot(train_loss_history, label='Train Loss')
    axes[2].plot(val_loss_history, label='Val Loss')
    axes[2].set_title('Loss Curve (Zoomed)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE Loss')
    axes[2].set_ylim([0, 0.05])
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(save_folder, "training_curves_pytorch.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_folder}/training_curves_pytorch.png")
    plt.close()


if __name__ == '__main__':
    train()