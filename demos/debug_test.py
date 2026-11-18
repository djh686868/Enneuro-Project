import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

print("Step 1: Importing basic modules...")
from eneuro.base import Tensor, Config
from eneuro.base.functions import Conv2d
from eneuro.nn.loss import MSELoss
from eneuro.nn.optim import Adam
print("   OK")

print("\nStep 2: Importing model...")
t0 = time.time()
from model import AutoDriveNet
t1 = time.time()
print(f"   OK (time: {t1-t0:.2f}s)")

print("\nStep 3: Creating model...")
t0 = time.time()
model = AutoDriveNet(in_channels=3, num_classes=1)
t1 = time.time()
print(f"   OK (time: {t1-t0:.2f}s)")

print("\nStep 4: Importing dataset...")
t0 = time.time()
from dataset import AutoDriveDataset, preprocess_image
t1 = time.time()
print(f"   OK (time: {t1-t0:.2f}s)")

print("\nStep 5: Loading dataset (first 100 samples)...")
t0 = time.time()
train_dataset = AutoDriveDataset(mode='train', transform=preprocess_image)
t1 = time.time()
print(f"   Dataset size: {len(train_dataset)}")
print(f"   OK (time: {t1-t0:.2f}s)")

print("\nStep 6: Getting first sample...")
t0 = time.time()
img, label = train_dataset[0]
t1 = time.time()
print(f"   Image shape: {img.shape}")
print(f"   Label: {label}")
print(f"   OK (time: {t1-t0:.2f}s)")

print("\nStep 7: Testing forward pass with single image...")
t0 = time.time()
Config.train = True
Conv2d.FFT_MIN_KERNEL_SIZE = 99
Conv2d.FFT_MIN_SPATIAL_SIZE = 9999
model.to('cpu')

img_tensor = Tensor(img.reshape(1, 3, 120, 160))
y_hat = model(img_tensor)
t1 = time.time()
print(f"   Output shape: {y_hat.shape}")
print(f"   OK (time: {t1-t0:.4f}s)")

print("\n=== Debug test completed ===")