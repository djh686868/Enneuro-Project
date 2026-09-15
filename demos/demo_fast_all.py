# -*- coding: utf-8 -*-
"""
快速版演示脚本 — GPU对比 + 早停 + L1L2正则化
使用小批量数据（100个样本）和少量Epoch（3-5个），快速展示效果
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
from font_setup import setup_cjk_font

from eneuro.base import Tensor, Config, as_Tensor
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from eneuro.data.dataloader import DataLoader
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive, AutoDriveNet

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    HAS_CUPY = False
    HAS_GPU = False

# 使用小数据集加速
MAX_TRAIN = 200
MAX_VAL = 80


def make_small_loader(dataset, loader, max_n):
    """只取前 max_n 个 batch"""
    data = []
    for i, (xb, yb) in enumerate(loader):
        data.append((xb, yb))
        if i >= max_n - 1:
            break
    return data


def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =========================== 1. GPU 对比 ===========================
def run_gpu_compare():
    print_header("1. GPU 适配性能对比")
    ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    data = make_small_loader(ds, dl, 4)

    for dev in ['cpu', 'cuda'] if HAS_GPU else ['cpu']:
        if dev == 'cuda' and not HAS_GPU:
            continue
        model = ResNet18AutoDrive(in_channels=3, num_classes=1)
        opt = Adam(model.params(), lr=0.001)
        loss_fn = MSELoss()
        model.to(dev)
        Config.train = True

        t0 = time.time()
        for Xb, yb in data:
            Xb_t = as_Tensor(Xb).to(dev)
            yb_t = as_Tensor(yb).to(dev)
            y_hat = model(Xb_t)
            loss = loss_fn(y_hat, yb_t)
            model.cleargrads()
            loss.backward()
            opt.step()
            del y_hat, loss, Xb_t, yb_t
        elapsed = time.time() - t0

        print(f"  {dev.upper():>4}: {elapsed:.2f}s ({len(data)} batches x 64)")
        if dev == 'cpu':
            cpu_time = elapsed
        else:
            gpu_time = elapsed

    if HAS_GPU:
        speedup = cpu_time / gpu_time
        print(f"\n  ==> GPU 加速比: {speedup:.2f}x")
        return cpu_time, gpu_time, speedup
    return cpu_time, None, None


# =========================== 2. 早停 ===========================
def run_early_stop():
    print_header("2. 早停机制对比")

    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    train_data = make_small_loader(train_ds, train_dl, 4)
    val_data = make_small_loader(val_ds, val_dl, 2)
    device = 'cuda' if HAS_GPU else 'cpu'

    def train_val(model, opt, loss_fn, epochs, do_early_stop=False):
        val_losses = []
        best_loss = float('inf')
        best_w = None
        no_impr = 0
        patience = 3
        stopped = None

        for ep in range(epochs):
            Config.train = True
            model.to(device)
            for Xb, yb in train_data:
                Xb_t = as_Tensor(Xb).to(device)
                yb_t = as_Tensor(yb).to(device)
                y_hat = model(Xb_t)
                loss = loss_fn(y_hat, yb_t)
                model.cleargrads()
                loss.backward()
                opt.step()
                del y_hat, loss, Xb_t, yb_t

            Config.train = False
            total_v = 0.0
            with Config.using_config('enable_backprop', False):
                for Xb, yb in val_data:
                    Xb_t = as_Tensor(Xb).to(device)
                    yb_t = as_Tensor(yb).to(device)
                    total_v += float(loss_fn(model(Xb_t), yb_t).data)
                    del Xb_t, yb_t
            v_loss = total_v / len(val_data)
            val_losses.append(v_loss)

            if do_early_stop:
                if v_loss < best_loss - 0.0001:
                    best_loss = v_loss
                    best_w = {id(p): p.data.copy() for p in model.params() if p.data is not None}
                    no_impr = 0
                else:
                    no_impr += 1
                if no_impr >= patience:
                    stopped = ep
                    if best_w:
                        for p in model.params():
                            if p.data is not None and id(p) in best_w:
                                p.data[:] = best_w[id(p)]
                    break

        return val_losses, stopped, best_loss if do_early_stop else val_losses[-1]

    print("  训练无早停 (5 Epoch)...")
    m1 = AutoDriveNet(in_channels=3, num_classes=1)
    o1 = Adam(m1.params(), lr=0.001)
    losses1, _, _ = train_val(m1, o1, MSELoss(), 5, do_early_stop=False)

    print("  训练有早停 (patience=3)...")
    m2 = AutoDriveNet(in_channels=3, num_classes=1)
    o2 = Adam(m2.params(), lr=0.001)
    losses2, stopped, best_l = train_val(m2, o2, MSELoss(), 15, do_early_stop=True)

    # 绘图
    setup_cjk_font()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(range(1, len(losses1)+1), losses1, 'b-o', ms=4)
    ax[0].set_title('无早停 (5 Epoch)', fontweight='bold')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Val Loss')
    ax[0].grid(alpha=0.3)
    ax[0].text(0.5, -0.15, f'最终 Loss: {losses1[-1]:.4f}', transform=ax[0].transAxes, ha='center', color='#666')

    ax[1].plot(range(1, len(losses2)+1), losses2, 'go-', ms=4)
    if stopped:
        ax[1].axvline(stopped+1, color='r', ls='--', label=f'早停@{stopped+1}')
    ax[1].set_title(f'有早停 (patience=3)', fontweight='bold')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Val Loss')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    if stopped:
        ax[1].text(0.5, -0.15, f'早停于 E{stopped+1}, Best Loss: {best_l:.4f}', transform=ax[1].transAxes, ha='center', color='#666')

    plt.suptitle('早停机制 — AutoDriveNet + DonkeyCar', fontweight='bold', fontsize=14)
    plt.tight_layout()
    sp = os.path.join(OUTPUT_DIR, 'early_stopping_comparison.png')
    plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {sp}")


# =========================== 3. L1/L2 正则化 ===========================
def run_l1l2():
    print_header("3. L1/L2 正则化对比")
    ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    data = make_small_loader(ds, dl, 4)
    device = 'cuda' if HAS_GPU else 'cpu'

    results = {}
    for name, l1, l2 in [('None', 0, 0), ('L1(0.01)', 0.01, 0), ('L2(0.01)', 0.0, 0.01)]:
        m = AutoDriveNet(in_channels=3, num_classes=1)
        opt = Adam(m.params(), lr=0.001, l1_lambda=l1, l2_lambda=l2)
        loss_fn = MSELoss()

        Config.train = True
        m.to(device)
        for ep in range(5):
            for Xb, yb in data:
                Xb_t = as_Tensor(Xb).to(device)
                yb_t = as_Tensor(yb).to(device)
                y_hat = m(Xb_t)
                loss = loss_fn(y_hat, yb_t)
                m.cleargrads()
                loss.backward()
                opt.step()
                del y_hat, loss, Xb_t, yb_t

        weights = []
        for p in m.params():
            if p.data is not None and p.name == 'W':
                w = p.data.get() if hasattr(p.data, 'get') else p.data
                weights.append(w.flatten())
        all_w = np.concatenate(weights)
        spar = np.sum(np.abs(all_w) < 1e-4) / len(all_w) * 100
        results[name] = {'w': all_w, 'sparsity': spar, 'std': all_w.std(), 'mean': all_w.mean()}
        print(f"  {name:>10}: mean={all_w.mean():.4f}, std={all_w.std():.4f}, sparsity={spar:.1f}%")

    # 绘图
    setup_cjk_font()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    for (name, r), c in zip(results.items(), colors):
        w = np.clip(r['w'], -0.3, 0.3)
        ax[0].hist(w, bins=60, alpha=0.4, color=c, label=name)

    ax[0].set_title('权重分布', fontweight='bold')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    bars = ax[1].bar(list(results.keys()), [results[k]['sparsity'] for k in results], color=colors)
    ax[1].set_title('稀疏度 (|w|<1e-4)', fontweight='bold')
    ax[1].set_ylabel('%')
    for b, (k, r) in zip(bars, results.items()):
        ax[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{r["sparsity"]:.1f}%', ha='center', fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3)

    plt.suptitle('L1/L2 正则化 — AutoDriveNet + DonkeyCar', fontweight='bold', fontsize=14)
    plt.tight_layout()
    sp = os.path.join(OUTPUT_DIR, 'l1_l2_regularization.png')
    plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {sp}")


if __name__ == '__main__':
    t0 = time.time()
    run_gpu_compare()
    run_early_stop()
    run_l1l2()
    print(f"\nTotal: {time.time()-t0:.1f}s")
