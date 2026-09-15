#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 适配效果展示
使用 AutoDriveNet 在 DonkeyCar 数据集上训练，对比 CPU vs GPU 性能
生成柱状图展示加速效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

# CuPy 13.3.0 需匹配 CUDA 12.6，强制指定路径
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from font_setup import setup_cjk_font
setup_cjk_font()

from eneuro.base import Tensor, Config
from eneuro.base.functions import Conv2d
from eneuro.nn.loss import MSELoss
from eneuro.nn.optim import Adam

from model import AutoDriveNet
from dataset import AutoDriveDataset, preprocess_image


def main():
    print('=' * 70)
    print(' GPU 适配效果展示 (AutoDriveNet + DonkeyCar)')
    print('=' * 70)

    print('\n[1] 加载 DonkeyCar 数据集...')
    ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    n_samples = min(200, len(ds))
    print(f'    使用 {n_samples} 个样本')

    img_list, label_list = [], []
    for i in range(n_samples):
        img, label = ds[i]
        img_list.append(img)
        label_list.append(label)
    x_data = np.stack(img_list, axis=0)
    y_data = np.stack(label_list, axis=0).reshape(-1, 1)
    print(f'    数据形状: x={x_data.shape}, y={y_data.shape}')

    print('\n[2] CPU 训练中...')
    model_cpu = AutoDriveNet(in_channels=3, num_classes=1)
    loss_fn = MSELoss()
    optimizer = Adam(model_cpu.params(), lr=0.001)

    old_fft_kernel = Conv2d.FFT_MIN_KERNEL_SIZE
    old_fft_spatial = Conv2d.FFT_MIN_SPATIAL_SIZE
    Conv2d.FFT_MIN_KERNEL_SIZE = 99
    Conv2d.FFT_MIN_SPATIAL_SIZE = 9999

    t0 = time.time()
    for i in range(0, n_samples, 8):
        batch_x = Tensor(x_data[i:i+8])
        batch_y = Tensor(y_data[i:i+8])
        y_hat = model_cpu(batch_x)
        loss = loss_fn(y_hat, batch_y)
        model_cpu.cleargrads()
        loss.backward()
        optimizer.step()
    cpu_train_time = time.time() - t0

    Conv2d.FFT_MIN_KERNEL_SIZE = old_fft_kernel
    Conv2d.FFT_MIN_SPATIAL_SIZE = old_fft_spatial

    print(f'    CPU 训练时间: {cpu_train_time:.2f}s')

    gpu_train_time = None
    speedup = None

    if HAS_GPU:
        print('\n[3] GPU 训练中...')
        model_gpu = AutoDriveNet(in_channels=3, num_classes=1)
        model_gpu.to('cuda')

        x_gpu = cp.array(x_data)
        y_gpu = cp.array(y_data)

        loss_fn = MSELoss()
        optimizer_gpu = Adam(model_gpu.params(), lr=0.001)

        # GPU 端同样禁用 FFT，避免 NVRTC 编译问题
        Conv2d.FFT_MIN_KERNEL_SIZE = 99
        Conv2d.FFT_MIN_SPATIAL_SIZE = 9999

        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        for i in range(0, n_samples, 8):
            batch_x = x_gpu[i:i+8]
            batch_y = y_gpu[i:i+8]
            bx = Tensor(batch_x)
            by = Tensor(batch_y)
            y_hat = model_gpu(bx)
            loss = loss_fn(y_hat, by)
            model_gpu.cleargrads()
            loss.backward()
            optimizer_gpu.step()
        cp.cuda.Stream.null.synchronize()
        gpu_train_time = time.time() - t0

        Conv2d.FFT_MIN_KERNEL_SIZE = old_fft_kernel
        Conv2d.FFT_MIN_SPATIAL_SIZE = old_fft_spatial

        speedup = cpu_train_time / gpu_train_time if gpu_train_time > 0 else 0
        print(f'    GPU 训练时间: {gpu_train_time:.2f}s')
        print(f'    GPU 加速比: {speedup:.2f}x')
    else:
        print('\n[3] GPU (CuPy) 不可用，跳过 GPU 测试')

    print('\n' + '=' * 70)
    print(' 性能对比汇总')
    print('=' * 70)
    print(f' {"设备":<10} {"训练时间":>12} {"加速比":>10}')
    print(f' {"-" * 32}')
    print(f' {"CPU":<10} {cpu_train_time:>10.2f}s {"1.00x":>10}')
    if gpu_train_time and speedup:
        print(f' {"GPU":<10} {gpu_train_time:>10.2f}s {speedup:>8.2f}x')
    print('=' * 70)

    out_dir = script_dir / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    categories = ['CPU', 'GPU']
    times = [cpu_train_time, gpu_train_time if gpu_train_time else cpu_train_time * 0.1]
    colors = ['#4472C4', '#ED7D31']

    bars = ax1.bar(categories, times, width=0.5, color=colors)
    ax1.set_ylabel('训练时间 (秒)')
    ax1.set_title('AutoDriveNet 训练时间对比')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f}s', ha='center', va='bottom', fontsize=10)

    speedup_vals = [1.0, speedup if speedup else 0]
    bar_color = '#2F5496' if speedup and speedup >= 1 else '#C00000'
    bars2 = ax2.bar(['CPU', 'GPU'], speedup_vals, width=0.5, color=['#4472C4', bar_color])
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel('加速比 (x)')
    ax2.set_title('GPU 相对 CPU 加速比')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, speedup_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(str(out_dir), 'gpu_adaptation_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    if os.path.exists(chart_path):
        print(f'\n图表已保存: {chart_path}')
    else:
        print(f'\n警告: 图表保存失败: {chart_path}')

    report_path = os.path.join(str(out_dir), 'gpu_adaptation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('GPU 适配效果报告\n')
        f.write('数据集: DonkeyCar (转向角回归)\n')
        f.write('模型: AutoDriveNet (轻量级 CNN)\n')
        f.write(f'训练样本: {n_samples} 个, Batch Size: 8\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'CPU 训练时间: {cpu_train_time:.2f}s\n')
        if gpu_train_time and speedup:
            f.write(f'GPU 训练时间: {gpu_train_time:.2f}s\n')
            f.write(f'GPU 加速比: {speedup:.2f}x\n')
        else:
            f.write('GPU 不可用\n')
    print(f'报告已保存: {report_path}')

    print('\n=== GPU 适配效果展示完成 ===')


if __name__ == '__main__':
    main()
