# -*- coding: utf-8 -*-
"""
L1/L2 正则化效果对比演示脚本
在 DonkeyCar 数据集上训练 ResNet18，对比无正则化 / L1 / L2 的权重分布和训练效果
仅训练少量 Epoch 展示正则化对权重的影响
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


def train_n_epochs(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=5):
    """训练 n Epoch，返回每个 epoch 的 train/val loss 和最终权重"""
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        Config.train = True
        model.to(device)
        epoch_loss = 0.0
        n_batches = 0
        for Xb, yb in train_loader:
            Xb_t = as_Tensor(Xb).to(device)
            yb_t = as_Tensor(yb).to(device)
            y_hat = model(Xb_t)
            loss = loss_fn(y_hat, yb_t)
            model.cleargrads()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data)
            n_batches += 1
            del y_hat, loss, Xb_t, yb_t
        train_losses.append(epoch_loss / n_batches)

        # Validation
        Config.train = False
        total_val = 0.0
        with Config.using_config('enable_backprop', False):
            for Xb, yb in val_loader:
                Xb_t = as_Tensor(Xb).to(device)
                yb_t = as_Tensor(yb).to(device)
                total_val += float(loss_fn(model(Xb_t), yb_t).data) * len(Xb)
                del Xb_t, yb_t
        val_losses.append(total_val / len(val_loader.dataset))

        if epoch % 2 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

    # 收集所有权重
    weights = []
    for p in model.params():
        if p.data is not None and p.name == 'W':
            w = p.data
            if hasattr(w, 'get'):
                w = w.get()  # cupy -> numpy
            weights.append(w.flatten())

    all_weights = np.concatenate(weights) if weights else np.array([])
    return train_losses, val_losses, all_weights


def main():
    print("=" * 70)
    print("L1 / L2 正则化效果对比演示")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载 DonkeyCar 数据集...")
    train_dataset = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_dataset = AutoDriveDataset(mode='val', transform=preprocess_image)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    device = 'cuda' if HAS_CUPY else 'cpu'

    EPOCHS = 8
    results = {}

    # ---- 无正则化 ----
    print("\n[2] 训练: 无正则化...")
    model_none = AutoDriveNet(in_channels=3, num_classes=1)
    opt_none = Adam(model_none.params(), lr=0.001, l1_lambda=0.0, l2_lambda=0.0)
    loss_fn = MSELoss()
    t_loss, v_loss, w_none = train_n_epochs(model_none, train_loader, val_loader, loss_fn, opt_none, device, EPOCHS)
    results['None'] = {'train': t_loss, 'val': v_loss, 'weights': w_none}
    print(f"    权重统计: mean={w_none.mean():.6f}, std={w_none.std():.6f}, sparsity={np.sum(np.abs(w_none) < 1e-4)/len(w_none)*100:.1f}%")

    # ---- L1 正则化 ----
    print("\n[3] 训练: L1 正则化 (lambda=0.001)...")
    model_l1 = AutoDriveNet(in_channels=3, num_classes=1)
    opt_l1 = Adam(model_l1.params(), lr=0.001, l1_lambda=0.001, l2_lambda=0.0)
    t_loss_l1, v_loss_l1, w_l1 = train_n_epochs(model_l1, train_loader, val_loader, loss_fn, opt_l1, device, EPOCHS)
    results['L1 (lambda=0.001)'] = {'train': t_loss_l1, 'val': v_loss_l1, 'weights': w_l1}
    print(f"    权重统计: mean={w_l1.mean():.6f}, std={w_l1.std():.6f}, sparsity={np.sum(np.abs(w_l1) < 1e-4)/len(w_l1)*100:.1f}%")

    # ---- L2 正则化 ----
    print("\n[4] 训练: L2 正则化 (lambda=0.01)...")
    model_l2 = AutoDriveNet(in_channels=3, num_classes=1)
    opt_l2 = Adam(model_l2.params(), lr=0.001, l1_lambda=0.0, l2_lambda=0.01)
    t_loss_l2, v_loss_l2, w_l2 = train_n_epochs(model_l2, train_loader, val_loader, loss_fn, opt_l2, device, EPOCHS)
    results['L2 (lambda=0.01)'] = {'train': t_loss_l2, 'val': v_loss_l2, 'weights': w_l2}
    print(f"    权重统计: mean={w_l2.mean():.6f}, std={w_l2.std():.6f}, sparsity={np.sum(np.abs(w_l2) < 1e-4)/len(w_l2)*100:.1f}%")

    # ---- 生成可视化 ----
    print("\n[5] 生成可视化图表...")
    setup_cjk_font()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 左上: Loss 曲线
    ax = axes[0, 0]
    ax.plot(range(1, EPOCHS+1), results['None']['val'], 'o-', color='#3498DB', linewidth=2, markersize=5, label='无正则化')
    ax.plot(range(1, EPOCHS+1), results['L1 (lambda=0.001)']['val'], 's-', color='#E74C3C', linewidth=2, markersize=5, label='L1')
    ax.plot(range(1, EPOCHS+1), results['L2 (lambda=0.01)']['val'], '^-', color='#2ECC71', linewidth=2, markersize=5, label='L2')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('验证集 Loss 曲线', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 右上: 最终权重分布直方图
    ax = axes[0, 1]
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    alphas = [0.5, 0.5, 0.5]
    for (name, res), c, a in zip(results.items(), colors, alphas):
        w = res['weights']
        w_clipped = np.clip(w, -0.5, 0.5)
        ax.hist(w_clipped, bins=80, alpha=a, color=c, label=name, density=True)
    ax.set_xlabel('权重值', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('权重分布直方图 (范围 [-0.5, 0.5])', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 左下: 稀疏度对比
    ax = axes[1, 0]
    sparsities = [np.sum(np.abs(results[k]['weights']) < 1e-4) / len(results[k]['weights']) * 100 for k in results]
    bar_colors = ['#3498DB', '#E74C3C', '#2ECC71']
    bars = ax.bar(list(results.keys()), sparsities, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1)
    ax.set_ylabel('稀疏度 (%)', fontsize=12)
    ax.set_title('权重稀疏度对比 (|w| < 1e-4)', fontsize=13, fontweight='bold')
    for bar, s in zip(bars, sparsities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{s:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 右下: 权重标准差对比
    ax = axes[1, 1]
    stds = [results[k]['weights'].std() for k in results]
    bars = ax.bar(list(results.keys()), stds, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1)
    ax.set_ylabel('权重标准差', fontsize=12)
    ax.set_title('权重标准差对比 (L2 抑制大权重)', fontsize=13, fontweight='bold')
    for bar, s in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{s:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('L1 / L2 正则化效果对比 — DonkeyCar + ResNet18', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'l1_l2_regularization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] 对比图已保存: {save_path}")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'l1_l2_regularization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("L1 / L2 正则化效果对比报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'正则化':<22} {'均值':>10} {'标准差':>10} {'稀疏度':>10} {'范数':>12}\n")
        f.write("-" * 70 + "\n")
        for name in results:
            w = results[name]['weights']
            l1_norm = np.sum(np.abs(w))
            l2_norm = np.sqrt(np.sum(w**2))
            f.write(f"{name:<22} {w.mean():>8.4f} {w.std():>8.4f} "
                    f"{np.sum(np.abs(w)<1e-4)/len(w)*100:>8.1f}% "
                    f"L1={l1_norm:.2f} L2={l2_norm:.2f}\n")
        f.write("\n")
        f.write("L1 正则化: 促进稀疏性 (更多权重接近 0)\n")
        f.write("L2 正则化: 抑制大权重 (权重分布更集中)\n")
    print(f"    [OK] 报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("L1/L2 正则化演示完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
