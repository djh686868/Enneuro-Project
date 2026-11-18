# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_angles(file_path):
    angles = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            angle = float(parts[1])
            angles.append(angle)
    return np.array(angles)


def analyze_distribution():
    current_dir = Path(__file__).resolve().parent
    train_file = str(current_dir / "train.txt")
    val_file = str(current_dir / "val.txt")
    
    train_angles = load_angles(train_file)
    val_angles = load_angles(val_file)
    all_angles = np.concatenate([train_angles, val_angles])
    
    print(f"训练集样本数: {len(train_angles)}")
    print(f"验证集样本数: {len(val_angles)}")
    print(f"总样本数: {len(all_angles)}")
    print(f"\n转向角统计:")
    print(f"  最小值: {all_angles.min():.4f}")
    print(f"  最大值: {all_angles.max():.4f}")
    print(f"  平均值: {all_angles.mean():.4f}")
    print(f"  中位数: {np.median(all_angles):.4f}")
    print(f"  标准差: {all_angles.std():.4f}")
    
    bin_width = 0.1
    bins = np.arange(-1.0, 1.01, bin_width)
    bin_labels = [f"{bins[i]:.1f}~{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    train_counts, _ = np.histogram(train_angles, bins=bins)
    val_counts, _ = np.histogram(val_angles, bins=bins)
    all_counts, _ = np.histogram(all_angles, bins=bins)
    
    train_percent = train_counts / len(train_angles) * 100
    val_percent = val_counts / len(val_angles) * 100
    all_percent = all_counts / len(all_angles) * 100
    
    print(f"\n各区间分布统计 (区间宽度: {bin_width}):")
    print(f"{'区间':<12} {'训练集数量':>10} {'训练集比例':>10} {'验证集数量':>10} {'验证集比例':>10} {'总计数量':>10} {'总计比例':>10}")
    print("-" * 90)
    for i in range(len(bin_labels)):
        print(f"{bin_labels[i]:<12} {train_counts[i]:>10} {train_percent[i]:>9.2f}% {val_counts[i]:>10} {val_percent[i]:>9.2f}% {all_counts[i]:>10} {all_percent[i]:>9.2f}%")
    
    save_folder = str(current_dir / "results")
    os.makedirs(save_folder, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].bar(range(len(bin_labels)), all_counts, color='#4E79A7', edgecolor='black')
    axes[0].set_title('All Data Steering Angle Distribution', fontsize=14)
    axes[0].set_xlabel('Steering Angle Range', fontsize=12)
    axes[0].set_ylabel('Sample Count', fontsize=12)
    axes[0].set_xticks(range(len(bin_labels)))
    axes[0].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(range(len(bin_labels)), train_counts, color='#F28E2B', edgecolor='black')
    axes[1].set_title('Training Set Steering Angle Distribution', fontsize=14)
    axes[1].set_xlabel('Steering Angle Range', fontsize=12)
    axes[1].set_ylabel('Sample Count', fontsize=12)
    axes[1].set_xticks(range(len(bin_labels)))
    axes[1].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(range(len(bin_labels)), val_counts, color='#59A14F', edgecolor='black')
    axes[2].set_title('Validation Set Steering Angle Distribution', fontsize=14)
    axes[2].set_xlabel('Steering Angle Range', fontsize=12)
    axes[2].set_ylabel('Sample Count', fontsize=12)
    axes[2].set_xticks(range(len(bin_labels)))
    axes[2].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "data_distribution_histogram.png"), dpi=150, bbox_inches='tight')
    print(f"\n直方图已保存到 {save_folder}/data_distribution_histogram.png")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = range(len(bin_labels))
    width = 0.25
    ax2.bar([i - width for i in x], train_counts, width=width, label='Train', color='#F28E2B', edgecolor='black')
    ax2.bar(x, val_counts, width=width, label='Validation', color='#59A14F', edgecolor='black')
    ax2.bar([i + width for i in x], all_counts, width=width, label='All', color='#4E79A7', edgecolor='black')
    ax2.set_title('Steering Angle Distribution Comparison', fontsize=14)
    ax2.set_xlabel('Steering Angle Range', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "data_distribution_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"对比图已保存到 {save_folder}/data_distribution_comparison.png")
    
    plt.close('all')


if __name__ == '__main__':
    analyze_distribution()