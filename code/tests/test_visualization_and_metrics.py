"""
tests/test_visualization_and_metrics.py

综合测试：可视化模块 + 模型评估模块
- 使用真实 MNIST 数据集（code/tests/testdata/MNIST_data/mnist.pkl）
- 使用框架自带 DataLoader / Trainer / Evaluator
- 覆盖 Visualizer（训练曲线、混淆矩阵）
- 覆盖 metrics（accuracy top-k、precision_recall_f1、ClassificationReport）
- 覆盖 GradCAMVisualizer（热力图叠加可视化）

运行方式（在 code/ 目录下）：
    python tests/test_visualization_and_metrics.py
    python tests/test_visualization_and_metrics.py --no-gradcam   # 跳过 Grad-CAM（更快）
    python tests/test_visualization_and_metrics.py --epochs 3     # 自定义训练轮数
"""

import sys, os, argparse, time, pickle
from pathlib import Path
import numpy as np

# ── 路径设置 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # code/
sys.path.insert(0, str(ROOT))

# ── 框架导入 ────────────────────────────────────────────────────────────────
import eneuro
from eneuro.data.dataset  import Dataset
from eneuro.data.dataloader import DataLoader
from eneuro.nn.module     import Module, Conv2d, Linear, BatchNorm
from eneuro.nn.optim      import Adam
from eneuro.nn.loss       import crossEntropyError
from eneuro.train         import Trainer, Evaluator
from eneuro.train.meters  import AverageMeter
from eneuro.train.metrics import (
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    ClassificationReport,
)
from eneuro.utils.visualization import Visualizer, GradCAMVisualizer
from eneuro.base import functions as F
from eneuro.base.core import Tensor

# ── CLI 参数 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     type=int,  default=2,     help='训练轮数')
parser.add_argument('--batch-size', type=int,  default=128,   help='批大小')
parser.add_argument('--subset',     type=int,  default=10000, help='取前 N 个训练样本（加速测试）')
parser.add_argument('--device',     type=str,  default='cpu', help='cpu 或 cuda')
parser.add_argument('--no-gradcam', action='store_true',      help='跳过 Grad-CAM 测试')
parser.add_argument('--save-dir',   type=str,  default='.',   help='图表保存目录')
args = parser.parse_args()

SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEP  = '=' * 64
sep  = '-' * 64
PASS = '[PASS]'
FAIL = '[FAIL]'
_failures = []

def check(name, cond, detail=''):
    if cond:
        print(f'  {PASS}  {name}')
    else:
        print(f'  {FAIL}  {name}  {detail}')
        _failures.append(name)

# ════════════════════════════════════════════════════════════════════════════
# 1. 数据集
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  1. 加载 MNIST 数据集')
print(SEP)

PKL = ROOT / 'tests' / 'testdata' / 'MNIST_data' / 'mnist.pkl'
assert PKL.exists(), f'找不到 MNIST 文件: {PKL}'

with open(PKL, 'rb') as f:
    raw = pickle.load(f)

X_train_raw = raw['train_img'].astype(np.float32) / 255.0    # (60000, 784)
y_train_raw = raw['train_label'].astype(np.int64)
X_test_raw  = raw['test_img'].astype(np.float32)  / 255.0    # (10000, 784)
y_test_raw  = raw['test_label'].astype(np.int64)

# 展平 → (N,1,28,28)
def to_nchw(X):
    return X.reshape(-1, 1, 28, 28)

X_train_raw = to_nchw(X_train_raw)
X_test_raw  = to_nchw(X_test_raw)

# 取子集加速测试
N_TRAIN = min(args.subset, len(X_train_raw))
X_train_raw = X_train_raw[:N_TRAIN]
y_train_raw = y_train_raw[:N_TRAIN]

print(f'  训练集: {X_train_raw.shape}  ({N_TRAIN} 样本)')
print(f'  测试集: {X_test_raw.shape}  ({len(X_test_raw)} 样本)')


class MNISTDataset(Dataset):
    """MNIST 数据集，继承框架 Dataset 基类"""
    def __init__(self, images, labels, transform=None):
        self.data   = images   # (N, 1, 28, 28) float32
        self.label  = labels   # (N,) int64
        self.transform = transform or (lambda x: x)
        self.target_transform = lambda x: x

    def prepare(self):          # Dataset.prepare() 由基类调用，此处已在 __init__ 完成
        pass

    def __getitem__(self, idx):
        img   = self.transform(self.data[idx])     # (1,28,28) numpy
        label = int(self.label[idx])
        # DataLoader.Tensor.stack 要求元素为 Tensor
        return Tensor(img), Tensor(np.array(label, dtype=np.int64))

    def __len__(self):
        return len(self.data)


train_dataset = MNISTDataset(X_train_raw, y_train_raw)
test_dataset  = MNISTDataset(X_test_raw,  y_test_raw)

# 使用框架 DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=256,             shuffle=False, drop_last=False)

check('DataLoader 可迭代', hasattr(train_loader, '__iter__'))
check('DataLoader 有 .dataset', hasattr(train_loader, 'dataset'))

# ════════════════════════════════════════════════════════════════════════════
# 2. 定义 LeNet 模型（使用框架 Module / Conv2d / Linear）
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  2. 定义 LeNet 模型')
print(SEP)


class LeNet(Module):
    """
    LeNet-5 变体，使用框架 Module 基类，参数自动注册。
    输入: (N, 1, 28, 28)  输出: (N, 10)
    """
    def __init__(self):
        super().__init__()
        # Layer 的 __setattr__ 会自动将 Conv2d / Linear 加入 _params
        self.conv1 = Conv2d(out_channels=6,  kernel_size=5, stride=1, pad=0)
        self.conv2 = Conv2d(out_channels=16, kernel_size=5, stride=1, pad=0)
        self.fc1   = Linear(out_size=120, in_size=16 * 4 * 4)
        self.fc2   = Linear(out_size=84)
        self.fc3   = Linear(out_size=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))           # (N,6,24,24)
        x = F.pooling(x, kernel_size=2, stride=2)  # (N,6,12,12)
        x = F.relu(self.conv2(x))           # (N,16,8,8)
        x = F.pooling(x, kernel_size=2, stride=2)  # (N,16,4,4)
        x = F.flatten(x)                    # (N,256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = LeNet()
total_params = sum(p.data.size for p in model.params() if p.data is not None)
print(f'  LeNet 总参数量: {total_params:,}')
check('模型参数可遍历', total_params > 0)

# 验证前向传播形状
dummy = Tensor(np.zeros((2, 1, 28, 28), dtype=np.float32))
out   = model(dummy)
check(f'前向输出形状 (2,10)', out.shape == (2, 10), f'实际: {out.shape}')

# ════════════════════════════════════════════════════════════════════════════
# 3. 训练（Trainer.fit + Visualizer）
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print(f'  3. 训练 {args.epochs} 个 epoch（device={args.device}）')
print(SEP)

visualizer = Visualizer(num_classes=10)
optimizer  = Adam(model.params(), lr=1e-3)
trainer    = Trainer(model, crossEntropyError, optimizer, visualizer=visualizer)

t0 = time.time()
trainer.fit(
    train_loader,
    test_loader,
    epochs      = args.epochs,
    batch_size  = args.batch_size,
    verbose     = True,
    device      = args.device,
)
train_time = time.time() - t0
print(f'\n  总训练时间: {train_time:.1f} s')

check('train_loss 已记录', len(visualizer.train_loss) == args.epochs)
check('val_loss   已记录', len(visualizer.val_loss)   == args.epochs)
check('epoch_times 已记录', len(visualizer.epoch_times) == args.epochs)
check('val_loss 最终值合理', visualizer.val_loss[-1] < 5.0)

# ════════════════════════════════════════════════════════════════════════════
# 4. Evaluator 评估 + 收集全量预测标签
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  4. 模型评估（Evaluator）')
print(SEP)

evaluator = Evaluator(model, crossEntropyError, visualizer=visualizer)
# visualizer 已在 Trainer 训练过程中记录了训练集预测；
# 重置 y_true/y_pred 只保留测试集的结果，用于混淆矩阵
visualizer.y_true.clear()
visualizer.y_pred.clear()

test_loss, test_acc = evaluator.evaluate(
    test_loader, batch_size=256, verbose=True, device=args.device
)
print(f'\n  Test Loss: {test_loss:.4f}   Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)')

check('Evaluator 返回 loss 合理', 0 < test_loss < 10)
check('Evaluator 返回 acc  合理', 0 <= test_acc <= 1.0)
check('y_true 非空（混淆矩阵数据）', len(visualizer.y_true) > 0)

# ════════════════════════════════════════════════════════════════════════════
# 5. 指标计算（metrics 模块）
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  5. 指标计算（accuracy / precision_recall_f1 / ClassificationReport）')
print(SEP)

y_pred_all = np.array(visualizer.y_pred, dtype=np.int64)
y_true_all = np.array(visualizer.y_true, dtype=np.int64)

# 5-a. Top-1 / Top-5（需要 logits，这里用 one-hot 模拟 logits，验证接口）
#      对于已收集的类别索引，top-1 直接计算
top1_result = accuracy(y_pred_all, y_true_all, topk=(1,))
top1 = top1_result[1]
print(f'  Top-1 Accuracy : {top1:.4f}  ({top1*100:.2f}%)')
check('Top-1 准确率与 Evaluator 一致', abs(top1 - test_acc) < 0.01,
      f'metrics={top1:.4f}  evaluator={test_acc:.4f}')

# 5-b. Precision / Recall / F1
for avg in ('macro', 'micro', 'weighted'):
    p, r, f = precision_recall_f1(y_pred_all, y_true_all, num_classes=10, average=avg)
    print(f'  {avg:8s}  P={p:.4f}  R={r:.4f}  F1={f:.4f}')
    check(f'{avg} F1 ∈ [0,1]', 0.0 <= f <= 1.0 + 1e-9)

# 5-c. 混淆矩阵
cm = confusion_matrix(y_pred_all, y_true_all, num_classes=10)
check('混淆矩阵形状 (10,10)', cm.shape == (10, 10))
check('混淆矩阵总和 = 测试集大小', cm.sum() == len(y_true_all),
      f'cm.sum={cm.sum()} N={len(y_true_all)}')
check('混淆矩阵对角线均非负', (np.diag(cm) >= 0).all())
diag_sum = np.diag(cm).sum()
print(f'  混淆矩阵对角线之和: {diag_sum} / {cm.sum()} = {diag_sum/cm.sum():.4f}')

# 5-d. ClassificationReport
MNIST_NAMES = [str(i) for i in range(10)]
report = ClassificationReport(y_pred_all, y_true_all, num_classes=10, class_names=MNIST_NAMES)
print(f'\n{sep}')
print(str(report))
print(sep)
check('report accuracy 与 metrics top-1 一致',
      abs(report.acc - top1) < 1e-9)
check('report to_dict 包含所有 10 类', all(str(i) in report.to_dict() for i in range(10)))

# ════════════════════════════════════════════════════════════════════════════
# 6. 可视化（Visualizer.plot_all）
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  6. Visualizer — 训练曲线 + 混淆矩阵')
print(SEP)

curve_path = SAVE_DIR / 'test_training_visualization.png'
try:
    visualizer.plot_all(save_path=str(curve_path), show=False)
    check('plot_all 正常执行', True)
    check('曲线图文件已保存', curve_path.exists())
    print(f'  已保存: {curve_path}')
except Exception as e:
    check('plot_all 正常执行', False, str(e))

# ════════════════════════════════════════════════════════════════════════════
# 7. GradCAMVisualizer
# ════════════════════════════════════════════════════════════════════════════
if not args.no_gradcam:
    print(f'\n{SEP}')
    print('  7. GradCAMVisualizer — Grad-CAM 热力图')
    print(SEP)

    try:
        # GradCAMVisualizer 自动选择最后一个 conv 层（conv2）
        gc_viz = GradCAMVisualizer(model, target_layer=model.conv2)

        # 取测试集前 3 个样本
        sample_imgs  = Tensor(X_test_raw[:3])
        sample_labels = y_test_raw[:3]

        gradcam_path = SAVE_DIR / 'test_gradcam_result.png'
        result = gc_viz.visualize(
            sample_imgs,
            class_idx  = None,       # 使用预测类别
            save_path  = str(gradcam_path),
            show       = False,
        )
        check('GradCAMVisualizer.visualize 正常执行', True)
        check('返回字典含 heatmap', 'heatmap' in result)
        check('热力图维度正确', result['heatmap'].ndim == 2)
        check('热力图文件已保存', gradcam_path.exists())
        print(f'  已保存: {gradcam_path}')
        print(f'  热力图形状: {result["heatmap"].shape}')
        print(f'  预测类别: {result["predicted_class"]} | 真实类别: {sample_labels[0]}')

    except Exception as e:
        check('GradCAMVisualizer.visualize 正常执行', False, str(e))
        import traceback
        traceback.print_exc()
else:
    print(f'\n  [跳过] GradCAMVisualizer（--no-gradcam）')

# ════════════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
if _failures:
    print(f'  {FAIL}  {len(_failures)} 项测试失败:')
    for f in _failures:
        print(f'      - {f}')
    sys.exit(1)
else:
    print(f'  {PASS}  全部测试通过！')
print(f'{SEP}\n')
