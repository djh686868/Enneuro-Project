# -*- coding: utf-8 -*-
"""极简演示: GPU/早停/L1L2 — 使用 AutoDriveNet + GPU"""
import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'tests', 'test_donkey'))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'code'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from font_setup import setup_cjk_font

from eneuro.base import Tensor, Config, as_Tensor
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from eneuro.data.dataloader import DataLoader
from dataset import AutoDriveDataset, preprocess_image
from model import AutoDriveNet
import cupy as cp

OUT = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda'

# 极小数据集
ds = AutoDriveDataset(mode='train', transform=preprocess_image)
dl = DataLoader(ds, batch_size=32, shuffle=True)
data = []
for i, (xb, yb) in enumerate(dl):
    data.append((xb, yb))
    if i >= 2:
        break

print(f"Data: {len(data)} batches, batch_size=32", flush=True)

# ======= 1. GPU对比 =======
print("1. GPU Benchmark", flush=True)
m = AutoDriveNet(in_channels=3, num_classes=1)
opt = Adam(m.params(), lr=0.001)
fn = MSELoss()
m.to(DEV); Config.train = True

t0 = time.time()
for Xb, yb in data:
    xt = as_Tensor(Xb).to(DEV); yt = as_Tensor(yb).to(DEV)
    lo = fn(m(xt), yt); m.cleargrads(); lo.backward(); opt.step()
    del lo, xt, yt
t1 = time.time() - t0
# do it again for min of 2
m2 = AutoDriveNet(in_channels=3, num_classes=1)
o2 = Adam(m2.params(), lr=0.001)
m2.to(DEV); Config.train = True
t0 = time.time()
for Xb, yb in data:
    xt = as_Tensor(Xb).to(DEV); yt = as_Tensor(yb).to(DEV)
    lo = fn(m2(xt), yt); m2.cleargrads(); lo.backward(); o2.step()
    del lo, xt, yt
t2 = time.time() - t0
print(f"GPU: {t1:.2f}s / {t2:.2f}s ({len(data)} batches)", flush=True)
with open(os.path.join(OUT, 'gpu_comparison.txt'), 'w') as f:
    f.write(f"GPU Training (3x32 samples): {min(t1,t2):.2f}s\n")

# ======= 2. 早停 =======
print("2. Early Stopping", flush=True)
val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
vdata = []
for i, (xb, yb) in enumerate(val_dl):
    vdata.append((xb, yb))
    if i >= 1:
        break

def do_es(patience, max_ep=15):
    m = AutoDriveNet(in_channels=3, num_classes=1)
    o = Adam(m.params(), lr=0.001)
    vlosses = []
    bl = float('inf'); bw = None; ni = 0; stop_ep = None
    for ep in range(max_ep):
        Config.train = True; m.to(DEV)
        for Xb, yb in data:
            xt = as_Tensor(Xb).to(DEV); yt = as_Tensor(yb).to(DEV)
            lo = fn(m(xt), yt); m.cleargrads(); lo.backward(); o.step()
            del lo, xt, yt
        Config.train = False; tv = 0.0
        with Config.using_config('enable_backprop', False):
            for Xb, yb in vdata:
                xt = as_Tensor(Xb).to(DEV); yt = as_Tensor(yb).to(DEV)
                tv += float(fn(m(xt), yt).data); del xt, yt
        vl = tv / len(vdata); vlosses.append(vl)
        if vl < bl - 0.0001:
            bl = vl; bw = {id(p): p.data.copy() for p in m.params() if p.data is not None}; ni = 0
        else:
            ni += 1
        if ni >= patience:
            stop_ep = ep
            if bw:
                for p in m.params():
                    if p.data is not None and id(p) in bw: p.data[:] = bw[id(p)]
            break
    return vlosses, stop_ep, bl

l_none, _, _ = do_es(99, max_ep=5)
l_es, stop_ep, best_l = do_es(2, max_ep=10)

setup_cjk_font()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(range(1, len(l_none)+1), l_none, 'b-o', ms=4)
ax[0].set_title('No Early Stop (5 ep)', fontweight='bold'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Val Loss'); ax[0].grid(alpha=0.3)
ax[1].plot(range(1, len(l_es)+1), l_es, 'go-', ms=4)
if stop_ep: ax[1].axvline(stop_ep+1, color='r', ls='--', label=f'ES@{stop_ep+1}')
ax[1].set_title(f'Early Stop (p=2)', fontweight='bold'); ax[1].set_xlabel('Epoch'); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.suptitle('Early Stopping — AutoDriveNet', fontweight='bold'); plt.tight_layout()
sp = os.path.join(OUT, 'early_stopping_comparison.png'); plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
print(f"  [OK] {sp}", flush=True)

# ======= 3. L1/L2 =======
print("3. L1/L2 Regularization", flush=True)
results = {}
for name, l1, l2 in [('None', 0, 0), ('L1(0.01)', 0.01, 0), ('L2(0.01)', 0, 0.01)]:
    m = AutoDriveNet(in_channels=3, num_classes=1)
    o = Adam(m.params(), lr=0.001, l1_lambda=l1, l2_lambda=l2)
    Config.train = True; m.to(DEV)
    for ep in range(5):
        for Xb, yb in data:
            xt = as_Tensor(Xb).to(DEV); yt = as_Tensor(yb).to(DEV)
            lo = fn(m(xt), yt); m.cleargrads(); lo.backward(); o.step()
            del lo, xt, yt
    w = []
    for p in m.params():
        if p.data is not None and p.name == 'W':
            wd = cp.asnumpy(p.data) if isinstance(p.data, cp.ndarray) else p.data
            w.append(wd.flatten())
    aw = np.concatenate(w); sp = np.sum(np.abs(aw)<1e-4)/len(aw)*100
    results[name] = {'w': aw, 'sparsity': sp, 'std': aw.std()}
    print(f"  {name}: sparsity={sp:.1f}%, std={aw.std():.4f}", flush=True)

setup_cjk_font()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#3498DB', '#E74C3C', '#2ECC71']
for (name, r), c in zip(results.items(), colors):
    wc = np.clip(r['w'], -0.3, 0.3)
    ax[0].hist(wc, bins=60, alpha=0.4, color=c, label=name)
ax[0].set_title('Weight Distribution', fontweight='bold'); ax[0].legend(); ax[0].grid(alpha=0.3)
bars = ax[1].bar(list(results.keys()), [results[k]['sparsity'] for k in results], color=colors)
ax[1].set_title('Sparsity (|w|<1e-4)', fontweight='bold'); ax[1].set_ylabel('%')
for b, (k, r) in zip(bars, results.items()):
    ax[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{r["sparsity"]:.1f}%', ha='center', fontweight='bold')
ax[1].grid(axis='y', alpha=0.3)
plt.suptitle('L1/L2 Regularization — AutoDriveNet', fontweight='bold'); plt.tight_layout()
sp = os.path.join(OUT, 'l1_l2_regularization.png'); plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
print(f"  [OK] {sp}", flush=True)
print("\nALL DONE!")
