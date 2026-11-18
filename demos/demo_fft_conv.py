# -*- coding: utf-8 -*-
"""
FFT 大卷积核加速演示脚本
对比 FFT 卷积路径与 im2col 朴素路径在大核卷积场景下的性能差异
使用随机大张量模拟大卷积核推理场景，不依赖 DonkeyCar 数据集
"""
import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

import numpy as np
from eneuro.base import Tensor
from eneuro.base.functions import Conv2d

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    HAS_CUPY = False
    HAS_GPU = False


def test_fft_benchmark():
    print("=" * 70)
    print("FFT 大卷积核加速演示")
    print("=" * 70)

    # 测试配置: 各种大核尺寸
    test_configs = [
        # (N, C, H, W, OC, KH, KW) — 描述
        ((1, 3, 64, 64, 16, 5, 5), "5x5 Conv (中等核)"),
        ((1, 3, 64, 64, 16, 7, 7), "7x7 Conv (大核)"),
        ((1, 3, 64, 64, 16, 9, 9), "9x9 Conv (大核)"),
        ((1, 3, 64, 64, 16, 11, 11), "11x11 Conv (超大核)"),
        ((1, 16, 32, 32, 32, 5, 5), "5x5 Conv (多通道)"),
        ((1, 16, 32, 32, 32, 7, 7), "7x7 Conv (多通道)"),
    ]

    # 保存原始阈值
    old_kernel = Conv2d.FFT_MIN_KERNEL_SIZE
    old_spatial = Conv2d.FFT_MIN_SPATIAL_SIZE

    results = []

    print(f"\n{'配置':<22} {'路径':<10} {'时间(ms)':>12} {'相对加速':>10}")
    print("-" * 60)

    for (N, C, H, W, OC, KH, KW), desc in test_configs:
        np.random.seed(42)

        x = np.random.randn(N, C, H, W).astype(np.float32)
        w = np.random.randn(OC, C, KH, KW).astype(np.float32)
        b = np.random.randn(OC).astype(np.float32)

        # ---- 测试 FFT 路径 ----
        Conv2d.FFT_MIN_KERNEL_SIZE = 1  # 强制走 FFT
        Conv2d.FFT_MIN_SPATIAL_SIZE = 1

        layer_fft = Conv2d(stride=(1, 1), pad=(KH // 2, KW // 2), dilation=1)
        # 清缓存确保公平
        layer_fft._path_cache.clear()

        # warmup
        _ = layer_fft(Tensor(x), Tensor(w), Tensor(b))

        times_fft = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = layer_fft(Tensor(x), Tensor(w), Tensor(b))
            times_fft.append((time.perf_counter() - t0) * 1000)

        fft_time = np.mean(times_fft[1:])  # 去掉第一次 warmup
        fft_path = layer_fft._select_forward_path(x, w)

        # ---- 测试 im2col 路径 ----
        Conv2d.FFT_MIN_KERNEL_SIZE = 99  # 强制不走 FFT
        Conv2d.FFT_MIN_SPATIAL_SIZE = 999

        layer_im2col = Conv2d(stride=(1, 1), pad=(KH // 2, KW // 2), dilation=1)
        layer_im2col._path_cache.clear()

        _ = layer_im2col(Tensor(x), Tensor(w), Tensor(b))

        times_im2col = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = layer_im2col(Tensor(x), Tensor(w), Tensor(b))
            times_im2col.append((time.perf_counter() - t0) * 1000)

        im2col_time = np.mean(times_im2col[1:])
        im2col_path = layer_im2col._select_forward_path(x, w)

        speedup = im2col_time / fft_time if fft_time > 0 else 0

        results.append({
            'desc': desc, 'KH': KH, 'KW': KW, 'C': C, 'OC': OC, 'H': H, 'W': W,
            'fft_time': fft_time, 'fft_path': fft_path,
            'im2col_time': im2col_time, 'im2col_path': im2col_path,
            'speedup': speedup
        })

        print(f"  {desc:<20} FFT:    {fft_time:>8.3f}ms")
        print(f"  {'':20} im2col: {im2col_time:>8.3f}ms  {'':>2}{speedup:>6.2f}x")
        print()

    # 恢复阈值
    Conv2d.FFT_MIN_KERNEL_SIZE = old_kernel
    Conv2d.FFT_MIN_SPATIAL_SIZE = old_spatial

    # ---- 生成可视化柱状图 ----
    print("[生成对比图表...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from font_setup import setup_cjk_font
    setup_cjk_font()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图: 柱状图
    ax = axes[0]
    labels = [f"{r['KH']}x{r['KW']}\nC={r['C']},OC={r['OC']}" for r in results]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, [r['im2col_time'] for r in results], width,
                   label='im2col (朴素)', color='#FF6B6B', alpha=0.85)
    bars2 = ax.bar(x + width/2, [r['fft_time'] for r in results], width,
                   label='FFT (加速)', color='#4ECDC4', alpha=0.85)

    ax.set_ylabel('推理时间 (ms)', fontsize=12)
    ax.set_title('FFT vs im2col 大核卷积推理时间', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, t in zip(bars2, [r['fft_time'] for r in results]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max([r['im2col_time'] for r in results])*0.01,
                f'{t:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2D8B84')

    # 右图: 加速比
    ax = axes[1]
    speedups = [r['speedup'] for r in results]
    colors = ['#27AE60' if s > 1.0 else '#E74C3C' for s in speedups]
    bars = ax.bar(x, speedups, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='相等 (1.0x)')
    ax.set_ylabel('FFT 加速比 (im2col/fft)', fontsize=12)
    ax.set_title('FFT 加速比 (越高越好)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{s:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('FFT 大卷积核加速性能对比', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'fft_convolution_benchmark.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] 对比图已保存: {save_path}")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'fft_convolution_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FFT 大卷积核加速性能对比报告\n")
        f.write(f"GPU 可用: {HAS_GPU}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'配置':<24} {'FFT(ms)':>10} {'im2col(ms)':>12} {'加速比':>8}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['desc']:<24} {r['fft_time']:>8.3f} {r['im2col_time']:>10.3f} {r['speedup']:>6.2f}x\n")
        f.write("\n")
        avg_speedup = np.mean([r['speedup'] for r in results])
        f.write(f"平均加速比: {avg_speedup:.2f}x\n")
        f.write("\n说明: FFT 路径适用于 kernel_size >= 5 且空间尺寸 >= 32 的大核卷积\n")
    print(f"    [OK] 报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("FFT 卷积加速演示完成!")
    print("=" * 70)


if __name__ == '__main__':
    test_fft_benchmark()
