#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU vs CPU 性能对比演示
基于 DonkeyCar 数据集 + ResNet18，对比 CPU/Numpy 和 GPU/CuPy 的推理与训练速度
"""
import sys, os, time
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

import numpy as np
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

from eneuro.base import Tensor, Config


def build_resnet18_gpu_test():
    """
    使用 DonkeyCar 的 ResNet18AutoDrive 模型进行 GPU/CPU 对比
    """
    from model import ResNet18AutoDrive
    from dataset import AutoDriveDataset, preprocess_image
    from eneuro.nn.loss import MSELoss
    from eneuro.nn.optim import Adam

    print('=' * 70)
    print(' ResNet18 (DonkeyCar) GPU vs CPU 性能对比')
    print('=' * 70)

    # 加载数据集
    ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    n_samples = min(200, len(ds))  # 只用 200 个样本加速
    print(f'\nUsing {n_samples} samples')

    img_list, label_list = [], []
    for i in range(n_samples):
        img, label = ds[i]
        img_list.append(img)
        label_list.append(label)
    x_data = np.stack(img_list, axis=0)
    y_data = np.stack(label_list, axis=0).reshape(-1, 1)

    # ============= CPU 测试 =============
    print('\n--- CPU (NumPy) 测试 ---')
    model_cpu = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model_cpu.get_params_list(only_trainable=True), lr=0.001)

    # 预热
    xx = Tensor(x_data[:4])
    _ = model_cpu(xx)

    # 推理测速
    t0 = time.perf_counter()
    for i in range(0, n_samples, 32):
        batch_x = Tensor(x_data[i:i+32])
        with Config.using_config('train', False):
            _ = model_cpu(batch_x)
    cpu_infer_time = time.perf_counter() - t0
    cpu_infer_imgs_per_sec = n_samples / cpu_infer_time

    # 训练 1 Epoch 测速
    t0 = time.perf_counter()
    n_batches = 0
    for i in range(0, n_samples, 32):
        batch_x = Tensor(x_data[i:i+32])
        batch_y = Tensor(y_data[i:i+32])
        y_hat = model_cpu(batch_x)
        loss = loss_fn(y_hat, batch_y)
        model_cpu.cleargrads()
        loss.backward()
        optimizer.step()
        n_batches += 1
    cpu_train_time = time.perf_counter() - t0
    cpu_train_imgs_per_sec = n_samples / cpu_train_time

    print(f'  推理: {cpu_infer_time:.3f}s ({cpu_infer_imgs_per_sec:.1f} imgs/s)')
    print(f'  训练(1 epoch): {cpu_train_time:.3f}s ({cpu_train_imgs_per_sec:.1f} imgs/s)')

    # ============= GPU 测试 =============
    gpu_infer_time = None
    gpu_train_time = None
    speedup_infer = 0
    speedup_train = 0

    if HAS_GPU:
        print(f'\n--- GPU (CuPy / {cp.cuda.runtime.getDeviceProperties(0)[b"name"].decode()}) 测试 ---')
        model_gpu = ResNet18AutoDrive()
        model_gpu.to('cuda')

        # GPU 数据
        x_gpu = cp.array(x_data)
        y_gpu = cp.array(y_data)

        loss_fn = MSELoss()
        optimizer_gpu = Adam(model_gpu.get_params_list(only_trainable=True), lr=0.001)

        # 预热
        xx = Tensor(x_gpu[:4].copy())
        _ = model_gpu(xx)

        # 推理测速
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n_samples, 32):
            batch_x = Tensor(x_gpu[i:i+32].copy())
            with Config.using_config('train', False):
                _ = model_gpu(batch_x)
        cp.cuda.Stream.null.synchronize()
        gpu_infer_time = time.perf_counter() - t0
        gpu_infer_imgs_per_sec = n_samples / gpu_infer_time

        # 训练 1 Epoch 测速
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n_samples, 32):
            batch_x = Tensor(x_gpu[i:i+32].copy())
            batch_y = Tensor(y_gpu[i:i+32].copy())
            y_hat = model_gpu(batch_x)
            loss = loss_fn(y_hat, batch_y)
            model_gpu.cleargrads()
            loss.backward()
            optimizer_gpu.step()
        cp.cuda.Stream.null.synchronize()
        gpu_train_time = time.perf_counter() - t0
        gpu_train_imgs_per_sec = n_samples / gpu_train_time

        speedup_infer = cpu_infer_time / gpu_infer_time
        speedup_train = cpu_train_time / gpu_train_time

        print(f'  推理: {gpu_infer_time:.3f}s ({gpu_infer_imgs_per_sec:.1f} imgs/s)')
        print(f'  训练(1 epoch): {gpu_train_time:.3f}s ({gpu_train_imgs_per_sec:.1f} imgs/s)')
        print(f'\n  GPU 加速比 - 推理: {speedup_infer:.1f}x  |  训练: {speedup_train:.1f}x')
    else:
        print('\n--- GPU (CuPy) 不可用，跳过 GPU 测试 ---')

    # ============= 汇总 =============
    print('\n' + '=' * 70)
    print(' 性能对比汇总')
    print('=' * 70)
    print(f' {"指标":<22} {"CPU (NumPy)":<22} {"GPU (CuPy)":<22} {"加速比":<10}')
    print(f' {"-"*70}')
    print(f' {"推理速度 (imgs/s)":<22} {cpu_infer_imgs_per_sec:<22.1f} {gpu_infer_imgs_per_sec if gpu_infer_time else "N/A":<22} {speedup_infer:<10.1f}x')
    print(f' {"训练速度 (imgs/s)":<22} {cpu_train_imgs_per_sec:<22.1f} {gpu_train_imgs_per_sec if gpu_train_time else "N/A":<22} {speedup_train:<10.1f}x')
    print(f' {"推理总耗时":<22} {cpu_infer_time:<22.3f}s {gpu_infer_time if gpu_infer_time else "N/A":<22}')
    print(f' {"训练总耗时":<22} {cpu_train_time:<22.3f}s {gpu_train_time if gpu_train_time else "N/A":<22}')

    # ============= 生成对比图表 =============
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = script_dir / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = ['推理速度\n(imgs/s)', '训练速度\n(imgs/s)']
    cpu_vals = [cpu_infer_imgs_per_sec, cpu_train_imgs_per_sec]
    gpu_vals = [gpu_infer_imgs_per_sec if gpu_infer_time else 0, gpu_train_imgs_per_sec if gpu_train_time else 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图: 速度对比
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax1.bar(x - width/2, cpu_vals, width, label='CPU (NumPy)', color='#4472C4')
    bars2 = ax1.bar(x + width/2, gpu_vals, width, label='GPU (CuPy)', color='#ED7D31')
    ax1.set_ylabel('Images / Second')
    ax1.set_title('CPU vs GPU 吞吐量对比 (ResNet18 + DonkeyCar)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, cpu_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, gpu_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # 柱状图: 加速比
    speeds = [speedup_infer, speedup_train]
    colors = ['#2F5496' if s >= 1 else '#C00000' for s in speeds]
    bars = ax2.bar([0, 1], speeds, width * 2, color=colors)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('GPU 相对 CPU 加速比')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, speeds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = out_dir / 'gpu_vs_cpu_benchmark.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nChart saved: {path}')
    print('\n=== GPU benchmark demo complete! ===')


if __name__ == '__main__':
    build_resnet18_gpu_test()
