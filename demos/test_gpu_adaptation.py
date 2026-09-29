#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, time
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

print("Step 1: Importing modules...")
import numpy as np
try:
    import cupy as cp
    HAS_GPU = True
    print(f"  CuPy available: {cp.__version__}")
except ImportError:
    HAS_GPU = False
    cp = None
    print("  CuPy not available")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eneuro.base import Tensor, Config
from eneuro.base.functions import Conv2d
from eneuro.nn.loss import MSELoss
from eneuro.nn.optim import Adam
from eneuro.data.dataloader import DataLoader

from model import AutoDriveNet
from dataset import AutoDriveDataset, preprocess_image
print("  All imports successful")

print("\nStep 2: Loading dataset...")
train_dataset = AutoDriveDataset(mode='train', transform=preprocess_image)
print(f"  Dataset size: {len(train_dataset)}")

print("\nStep 3: Creating DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print(f"  Number of batches: {len(train_loader)}")

print("\nStep 4: Testing model creation...")
model = AutoDriveNet(in_channels=3, num_classes=1)
print(f"  Model created successfully")

print("\nStep 5: Testing single batch...")
for Xb, yb in train_loader:
    print(f"  Input shape: {Xb.shape}")
    print(f"  Label shape: {yb.shape}")
    
    print("  Testing CPU forward pass...")
    Config.train = True
    Conv2d.FFT_MIN_KERNEL_SIZE = 99
    Conv2d.FFT_MIN_SPATIAL_SIZE = 9999
    model.to('cpu')
    
    Xb_t = Tensor(Xb)
    yb_t = Tensor(yb)
    
    t0 = time.time()
    y_hat = model(Xb_t)
    t1 = time.time()
    print(f"  Forward pass time: {t1-t0:.4f}s")
    print(f"  Output shape: {y_hat.shape}")
    
    loss_fn = MSELoss()
    loss = loss_fn(y_hat, yb_t)
    print(f"  Loss: {loss.data}")
    
    model.cleargrads()
    loss.backward()
    print(f"  Backward pass completed")
    
    optimizer = Adam(model.params(), lr=0.001)
    optimizer.step()
    print(f"  Optimizer step completed")
    
    break

print("\n=== Test completed successfully ===")