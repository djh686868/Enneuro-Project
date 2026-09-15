# -*- coding: utf-8 -*-
"""
早停机制 (Early Stopping) 演示脚本
在 DonkeyCar 数据集上训练 ResNet18，展示早停在验证集 loss 不下降时自动中断
"""
import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests', 'test_donkey'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eneuro.base import Tensor, as_Tensor, Config, as_Tensor
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from eneuro.data.dataloader import DataLoader
from eneuro.train.trainer import Trainer
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive, AutoDriveNet
from font_setup import setup_cjk_font

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def main():
    print("=" * 70)
    print("早停机制 (Early Stopping) 演示")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载 DonkeyCar 数据集...")
    train_dataset = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_dataset = AutoDriveDataset(mode='val', transform=preprocess_image)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    device = 'cuda' if HAS_CUPY else 'cpu'

    # 训练 1: 无早停 (训练 30 Epoch)
    print("\n[2] 训练 (无早停, 30 Epoch)...")
    model1 = AutoDriveNet(in_channels=3, num_classes=1)
    opt1 = Adam(model1.params(), lr=0.001)
    loss_fn = MSELoss()

    no_es_losses = []
    start = time.time()
    for epoch in range(30):
        # train
        Config.train = True
        model1.to(device)
        for Xb, yb in train_loader:
            Xb_t = as_Tensor(Xb).to(device)
            yb_t = as_Tensor(yb).to(device)
            y_hat = model1(Xb_t)
            loss = loss_fn(y_hat, yb_t)
            model1.cleargrads()
            loss.backward()
            opt1.step()
            del y_hat, loss, Xb_t, yb_t

        # val
        Config.train = False
        total_val = 0.0
        with Config.using_config('enable_backprop', False):
            for Xb, yb in val_loader:
                Xb_t = as_Tensor(Xb).to(device)
                yb_t = as_Tensor(yb).to(device)
                y_hat = model1(Xb_t)
                total_val += float(loss_fn(y_hat, yb_t).data) * len(Xb)
                del y_hat, Xb_t, yb_t
        val_loss = total_val / len(val_dataset)
        no_es_losses.append(val_loss)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch+1:2d}/30, Val Loss: {val_loss:.6f}")
    no_es_time = time.time() - start
    print(f"    无早停完成, 耗时: {no_es_time:.1f}s, 最终 Val Loss: {no_es_losses[-1]:.6f}")

    # 训练 2: 有早停 (patience=5)
    print("\n[3] 训练 (有早停, patience=5)...")
    model2 = AutoDriveNet(in_channels=3, num_classes=1)
    opt2 = Adam(model2.params(), lr=0.001)

    es_losses = []
    best_loss = float('inf')
    best_weights = None
    patience = 5
    no_improve = 0
    stopped_epoch = None

    start = time.time()
    for epoch in range(30):
        Config.train = True
        model2.to(device)
        for Xb, yb in train_loader:
            Xb_t = as_Tensor(Xb).to(device)
            yb_t = as_Tensor(yb).to(device)
            y_hat = model2(Xb_t)
            loss = loss_fn(y_hat, yb_t)
            model2.cleargrads()
            loss.backward()
            opt2.step()
            del y_hat, loss, Xb_t, yb_t

        Config.train = False
        total_val = 0.0
        with Config.using_config('enable_backprop', False):
            for Xb, yb in val_loader:
                Xb_t = as_Tensor(Xb).to(device)
                yb_t = as_Tensor(yb).to(device)
                y_hat = model2(Xb_t)
                total_val += float(loss_fn(y_hat, yb_t).data) * len(Xb)
                del y_hat, Xb_t, yb_t
        val_loss = total_val / len(val_dataset)
        es_losses.append(val_loss)

        # 早停检查
        if val_loss < best_loss - 0.0001:
            best_loss = val_loss
            best_weights = {}
            for p in model2.params():
                if p.data is not None:
                    best_weights[id(p)] = p.data.copy()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 3 == 0 or no_improve >= patience - 1:
            status = "Improved" if no_improve == 0 else f"NoImprove={no_improve}/{patience}"
            print(f"    Epoch {epoch+1:2d}/30, Val Loss: {val_loss:.6f}, {status}")

        if no_improve >= patience:
            stopped_epoch = epoch
            if best_weights:
                for p in model2.params():
                    if p.data is not None and id(p) in best_weights:
                        p.data[:] = best_weights[id(p)]
            print(f"\n    >>> 早停触发于 Epoch {epoch+1}, 恢复最佳权重 (Val Loss={best_loss:.6f})")
            break

    es_time = time.time() - start
    if stopped_epoch is None:
        stopped_epoch = 29
    print(f"    早停完成, 耗时: {es_time:.1f}s, 最终 Val Loss: {best_loss:.6f}")

    # 绘制对比图
    print("\n[4] 生成早停对比曲线...")
    setup_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 无早停
    ax = axes[0]
    ax.plot(range(1, len(no_es_losses) + 1), no_es_losses, 'b-o', markersize=3, linewidth=1.2)
    ax.axhline(y=min(no_es_losses), color='green', linestyle='--', alpha=0.6, label=f'Min={min(no_es_losses):.6f}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('无早停 — 训练全部 30 Epoch', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, -0.15, f'耗时: {no_es_time:.1f}s | 最终 Loss: {no_es_losses[-1]:.6f}',
            transform=ax.transAxes, ha='center', fontsize=10, color='#666')

    # 右图: 有早停
    ax = axes[1]
    colors = ['#4ECDC4' if l == min(es_losses) else '#FF6B6B' if (i >= stopped_epoch - patience + 1) else '#45B7D1'
              for i, l in enumerate(es_losses)]
    ax.plot(range(1, len(es_losses) + 1), es_losses, '-', color='#888', linewidth=1)
    ax.scatter(range(1, len(es_losses) + 1), es_losses, c=colors, s=30, zorder=5)
    ax.axvline(x=stopped_epoch + 1, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label=f'早停 @ Epoch {stopped_epoch+1}')
    ax.axhline(y=best_loss, color='green', linestyle='--', alpha=0.6, label=f'Best={best_loss:.6f}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(f'有早停 (patience={patience}) — 提前终止', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, -0.15, f'耗时: {es_time:.1f}s (节约 {(1-es_time/no_es_time)*100:.0f}%) | 最佳 Loss: {best_loss:.6f}',
            transform=ax.transAxes, ha='center', fontsize=10, color='#666')

    plt.suptitle('早停机制 (Early Stopping) 效果对比 — DonkeyCar + ResNet18',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'early_stopping_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] 对比图已保存: {save_path}")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'early_stopping_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("早停机制对比报告\n")
        f.write(f"patience={patience}, min_delta=0.0001\n")
        f.write("=" * 60 + "\n")
        f.write(f"无早停: {len(no_es_losses)} Epoch, {no_es_time:.1f}s, Loss: {no_es_losses[-1]:.6f}\n")
        f.write(f"有早停: {stopped_epoch+1} Epoch, {es_time:.1f}s, Loss: {best_loss:.6f}\n")
        f.write(f"时间节约: {(1-es_time/no_es_time)*100:.0f}%\n")
    print(f"    [OK] 报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("早停演示完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
