# -*- coding: utf-8 -*-
"""
GPU 适配性能对比脚本
在 DonkeyCar 数据集上训练 ResNet18，对比 CPU vs GPU 训练速度
只训练 1 个 Epoch 展示加速效果
"""
import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests', 'test_donkey'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from eneuro.base import Tensor, as_Tensor, Config, as_Tensor
from eneuro.base.functions import Conv2d
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from eneuro.data.dataloader import DataLoader
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, desc):
    """训练 1 个 Epoch，返回时间和 loss"""
    # 在 CPU 上暂时禁用 FFT 路径（numpy 兼容性问题）
    old_fft_kernel = Conv2d.FFT_MIN_KERNEL_SIZE
    old_fft_spatial = Conv2d.FFT_MIN_SPATIAL_SIZE
    if device == 'cpu':
        Conv2d.FFT_MIN_KERNEL_SIZE = 99  # 禁止 FFT
        Conv2d.FFT_MIN_SPATIAL_SIZE = 9999

    model.to(device)
    Config.train = True
    total_loss = 0.0
    n_batches = 0

    t_start = time.time()
    try:
        for Xb, yb in train_loader:
            Xb_t = as_Tensor(Xb).to(device)
            yb_t = as_Tensor(yb).to(device)

            y_hat = model(Xb_t)
            loss = loss_fn(y_hat, yb_t)

            model.cleargrads()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            n_batches += 1

            del y_hat, loss, Xb_t, yb_t
    finally:
        Conv2d.FFT_MIN_KERNEL_SIZE = old_fft_kernel
        Conv2d.FFT_MIN_SPATIAL_SIZE = old_fft_spatial

    elapsed = time.time() - t_start
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return elapsed, avg_loss


def main():
    print("=" * 70)
    print("GPU 适配性能对比 (DonkeyCar + ResNet18, 1 Epoch)")
    print("=" * 70)

    # 加载数据集
    print("\n[1] 加载 DonkeyCar 数据集...")
    train_dataset = AutoDriveDataset(mode='train', transform=preprocess_image)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"    训练集: {len(train_dataset)} 样本, {len(train_loader)} 个 batch")

    results = {}

    # ---- CPU 训练 ----
    print("\n[2] CPU 训练中...")
    model_cpu = ResNet18AutoDrive(in_channels=3, num_classes=1)
    optimizer_cpu = Adam(model_cpu.params(), lr=0.001)
    loss_fn_cpu = MSELoss()

    cpu_time, cpu_loss = train_one_epoch(
        model_cpu, train_loader, loss_fn_cpu, optimizer_cpu, 'cpu', 'CPU'
    )
    results['CPU'] = {'time': cpu_time, 'loss': cpu_loss}
    print(f"    CPU: {cpu_time:.2f}s, loss={cpu_loss:.6f}")

    # ---- GPU 训练 ----
    if HAS_CUPY:
        print("\n[3] GPU 训练中...")
        model_gpu = ResNet18AutoDrive(in_channels=3, num_classes=1)
        optimizer_gpu = Adam(model_gpu.params(), lr=0.001)
        loss_fn_gpu = MSELoss()

        gpu_time, gpu_loss = train_one_epoch(
            model_gpu, train_loader, loss_fn_gpu, optimizer_gpu, 'cuda', 'GPU'
        )
        results['GPU'] = {'time': gpu_time, 'loss': gpu_loss}

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"    GPU: {gpu_time:.2f}s, loss={gpu_loss:.6f}")
        print(f"\n    ==> GPU 加速比: {speedup:.2f}x")
    else:
        print("\n[3] GPU (CuPy) 不可用，跳过 GPU 测试")
        gpu_time = None
        speedup = None

    # ---- 生成对比报告 ----
    print("\n" + "=" * 70)
    print("  对比结果汇总")
    print("=" * 70)
    print(f"  {'设备':<10} {'训练时间':>12} {'Loss':>12} {'加速比':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'CPU':<10} {cpu_time:>10.2f}s {cpu_loss:>10.6f} {'1.00x':>10}")
    if gpu_time and speedup:
        print(f"  {'GPU':<10} {gpu_time:>10.2f}s {gpu_loss:>10.6f} {speedup:>8.2f}x")
    print("=" * 70)

    # 保存结果到文件
    report_path = os.path.join(OUTPUT_DIR, 'gpu_comparison.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GPU 适配性能对比报告\n")
        f.write("数据集: DonkeyCar (转向角回归)\n")
        f.write(f"模型: ResNet18, 训练集 {len(train_dataset)} 样本\n")
        f.write(f"Batch Size: 32, 训练 1 Epoch\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"CPU 训练时间: {cpu_time:.2f}s, Loss: {cpu_loss:.6f}\n")
        if gpu_time and speedup:
            f.write(f"GPU 训练时间: {gpu_time:.2f}s, Loss: {gpu_loss:.6f}\n")
            f.write(f"加速比: {speedup:.2f}x\n")
        else:
            f.write("GPU 不可用\n")
    print(f"\n[OK] 对比报告已保存: {report_path}")


if __name__ == '__main__':
    main()
