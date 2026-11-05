"""
tests/test_metrics.py

eneuro.train.metrics 单元测试

测试覆盖：
  1. accuracy      — top-1 完美预测 / 全错 / top-5 逻辑回归场景
  2. confusion_matrix — 基本形状与数值
  3. precision_recall_f1 — 四种 average 模式，与 sklearn 交叉验证
  4. ClassificationReport — 字符串与字典输出
  5. Tensor 输入兼容性
  6. 边界情况（单类别、全错）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

try:
    from sklearn.metrics import (
        precision_recall_fscore_support as sk_prf,
        accuracy_score as sk_acc,
        confusion_matrix as sk_cm,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from eneuro.train.metrics import (
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    ClassificationReport,
)
from eneuro.base.core import Tensor

# ─────────────────────────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"
_failures = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")
        _failures.append(name)

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. accuracy — top-1 / top-5
# ─────────────────────────────────────────────────────────────────────────────
section("1. accuracy()")

# 1a. 完美 top-1（类别索引输入）
y_pred_cls = np.array([0, 1, 2, 3, 4])
y_true     = np.array([0, 1, 2, 3, 4])
res = accuracy(y_pred_cls, y_true, topk=(1,))
check("top-1 完美预测 = 1.0", abs(res[1] - 1.0) < 1e-9)

# 1b. 全错
y_pred_cls2 = np.array([1, 2, 3, 4, 0])
res2 = accuracy(y_pred_cls2, y_true, topk=(1,))
check("top-1 全错 = 0.0", abs(res2[1] - 0.0) < 1e-9)

# 1c. 50% 准确
y_pred_cls3 = np.array([0, 2, 2, 0, 4])
res3 = accuracy(y_pred_cls3, y_true, topk=(1,))
check("top-1 50% ≈ 0.6", abs(res3[1] - 0.6) < 1e-9)

# 1d. logits 输入，top-1 和 top-5
N, C = 100, 10
rng = np.random.default_rng(42)
logits = rng.random((N, C)).astype(np.float32)
y_true_rand = rng.integers(0, C, size=N)
res4 = accuracy(logits, y_true_rand, topk=(1, 5))
check("logits top-1 ∈ [0,1]", 0.0 <= res4[1] <= 1.0)
check("logits top-5 ∈ [0,1]", 0.0 <= res4[5] <= 1.0)
check("top-5 >= top-1",       res4[5] >= res4[1] - 1e-9)

# 1e. Tensor 输入
y_pred_tensor = Tensor(y_pred_cls.astype(np.float32))
y_true_tensor = Tensor(y_true.astype(np.float32))
res5 = accuracy(y_pred_tensor, y_true_tensor, topk=(1,))
check("Tensor 输入 top-1 = 1.0", abs(res5[1] - 1.0) < 1e-9)

# ─────────────────────────────────────────────────────────────────────────────
# 2. confusion_matrix
# ─────────────────────────────────────────────────────────────────────────────
section("2. confusion_matrix()")

y_p = np.array([0, 1, 2, 0, 1, 2])
y_t = np.array([0, 1, 2, 1, 0, 2])
cm = confusion_matrix(y_p, y_t, num_classes=3)
check("形状 (3,3)",        cm.shape == (3, 3))
check("对角线 [0,0]=1",    cm[0, 0] == 1)
check("对角线 [1,1]=1",    cm[1, 1] == 1)
check("对角线 [2,2]=2",    cm[2, 2] == 2)
check("cm[0,1]=1 (0被预测为1)", cm[0, 1] == 1)   # 真实0，预测1
check("cm[1,0]=1 (1被预测为0)", cm[1, 0] == 1)   # 真实1，预测0
check("总和 = N",          cm.sum() == len(y_t))

# 2b. num_classes 自动推断
cm_auto = confusion_matrix(y_p, y_t)
check("auto num_classes 形状正确", cm_auto.shape == (3, 3))

# 2c. 对照 sklearn
if HAS_SKLEARN:
    cm_sk = sk_cm(y_t, y_p)
    check("与 sklearn confusion_matrix 一致", np.array_equal(cm, cm_sk))
else:
    print(f"  ⓘ  sklearn 未安装，跳过交叉验证")

# ─────────────────────────────────────────────────────────────────────────────
# 3. precision_recall_f1
# ─────────────────────────────────────────────────────────────────────────────
section("3. precision_recall_f1()")

# 固定样例：3 类，有不均衡
y_pred_prf = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
y_true_prf = np.array([0, 1, 1, 2, 2, 0, 0, 1, 2, 1])

for avg in ('macro', 'micro', 'weighted'):
    p, r, f = precision_recall_f1(y_pred_prf, y_true_prf, num_classes=3, average=avg)
    check(f"average='{avg}' p∈[0,1]", 0.0 <= p <= 1.0 + 1e-9)
    check(f"average='{avg}' r∈[0,1]", 0.0 <= r <= 1.0 + 1e-9)
    check(f"average='{avg}' f∈[0,1]", 0.0 <= f <= 1.0 + 1e-9)

# 'none' 返回数组
p_arr, r_arr, f_arr = precision_recall_f1(y_pred_prf, y_true_prf, num_classes=3, average='none')
check("average='none' 返回长度=3", len(p_arr) == 3 and len(f_arr) == 3)

# 对照 sklearn（macro）
if HAS_SKLEARN:
    sk_p, sk_r, sk_f, _ = sk_prf(y_true_prf, y_pred_prf, average='macro', zero_division=0)
    p_m, r_m, f_m = precision_recall_f1(y_pred_prf, y_true_prf, num_classes=3, average='macro')
    check("macro precision 与 sklearn 误差<1e-4", abs(p_m - sk_p) < 1e-4,
          f"ours={p_m:.6f} sklearn={sk_p:.6f}")
    check("macro recall    与 sklearn 误差<1e-4", abs(r_m - sk_r) < 1e-4,
          f"ours={r_m:.6f} sklearn={sk_r:.6f}")
    check("macro f1        与 sklearn 误差<1e-4", abs(f_m - sk_f) < 1e-4,
          f"ours={f_m:.6f} sklearn={sk_f:.6f}")

    sk_p2, sk_r2, sk_f2, _ = sk_prf(y_true_prf, y_pred_prf, average='weighted', zero_division=0)
    p_w, r_w, f_w = precision_recall_f1(y_pred_prf, y_true_prf, num_classes=3, average='weighted')
    check("weighted f1 与 sklearn 误差<1e-4", abs(f_w - sk_f2) < 1e-4,
          f"ours={f_w:.6f} sklearn={sk_f2:.6f}")

    sk_p3, sk_r3, sk_f3, _ = sk_prf(y_true_prf, y_pred_prf, average='micro', zero_division=0)
    p_i, r_i, f_i = precision_recall_f1(y_pred_prf, y_true_prf, num_classes=3, average='micro')
    check("micro f1 与 sklearn 误差<1e-4", abs(f_i - sk_f3) < 1e-4,
          f"ours={f_i:.6f} sklearn={sk_f3:.6f}")
else:
    print(f"  ⓘ  sklearn 未安装，跳过与 sklearn 的数值交叉验证")

# 完美预测时 F1 应为 1
y_perfect = np.arange(5)
p_perf, r_perf, f_perf = precision_recall_f1(y_perfect, y_perfect, num_classes=5, average='macro')
check("完美预测 macro-F1 ≈ 1.0", abs(f_perf - 1.0) < 1e-4,
      f"f1={f_perf:.6f}")

# 全错（3类循环错位）
y_wrong = np.array([1, 2, 0])
y_gt    = np.array([0, 1, 2])
_, _, f_wrong = precision_recall_f1(y_wrong, y_gt, num_classes=3, average='macro')
check("全错 macro-F1 = 0", abs(f_wrong - 0.0) < 1e-4, f"f1={f_wrong:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ClassificationReport
# ─────────────────────────────────────────────────────────────────────────────
section("4. ClassificationReport()")

names = ['cat', 'dog', 'bird']
report = ClassificationReport(y_pred_prf, y_true_prf, num_classes=3, class_names=names)

s = str(report)
check("报告包含 'cat'",          'cat' in s)
check("报告包含 'precision'",    'precision' in s)
check("报告包含 'macro avg'",    'macro avg' in s)
check("报告包含 'weighted avg'", 'weighted avg' in s)
check("报告包含 'accuracy'",     'accuracy' in s)

d = report.to_dict()
check("to_dict 包含 'cat'",          'cat' in d)
check("to_dict 包含 'accuracy'",     'accuracy' in d)
check("to_dict 包含 'macro avg'",    'macro avg' in d)
check("to_dict 包含 'weighted avg'", 'weighted avg' in d)
check("to_dict 包含 'micro avg'",    'micro avg' in d)
check("accuracy ∈ [0,1]",           0.0 <= d['accuracy'] <= 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 大规模随机数据 + sklearn 全面对比
# ─────────────────────────────────────────────────────────────────────────────
section("5. 大规模随机数据对比 (N=1000, C=10)")

if HAS_SKLEARN:
    rng2 = np.random.default_rng(0)
    N2, C2 = 1000, 10
    y_p_big = rng2.integers(0, C2, size=N2)
    y_t_big = rng2.integers(0, C2, size=N2)

    for avg_mode in ('macro', 'micro', 'weighted'):
        sk_pv, sk_rv, sk_fv, _ = sk_prf(y_t_big, y_p_big, average=avg_mode, zero_division=0)
        our_p, our_r, our_f = precision_recall_f1(y_p_big, y_t_big, num_classes=C2, average=avg_mode)
        check(f"N=1000 {avg_mode} P 误差<1e-4", abs(our_p - sk_pv) < 1e-4,
              f"ours={our_p:.6f} sk={sk_pv:.6f}")
        check(f"N=1000 {avg_mode} R 误差<1e-4", abs(our_r - sk_rv) < 1e-4,
              f"ours={our_r:.6f} sk={sk_rv:.6f}")
        check(f"N=1000 {avg_mode} F 误差<1e-4", abs(our_f - sk_fv) < 1e-4,
              f"ours={our_f:.6f} sk={sk_fv:.6f}")

    our_acc = accuracy(y_p_big, y_t_big, topk=(1,))[1]
    sk_acc_v = sk_acc(y_t_big, y_p_big)
    check("N=1000 accuracy 误差<1e-6", abs(our_acc - sk_acc_v) < 1e-6,
          f"ours={our_acc:.6f} sk={sk_acc_v:.6f}")
else:
    print("  ⓘ  sklearn 未安装，跳过大规模对比")

# ─────────────────────────────────────────────────────────────────────────────
# 打印报告示例
# ─────────────────────────────────────────────────────────────────────────────
section("ClassificationReport 输出示例（3类）")
print(report)

# ─────────────────────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if _failures:
    print(f"  {FAIL}  {len(_failures)} 项测试失败：")
    for f in _failures:
        print(f"      - {f}")
    sys.exit(1)
else:
    print(f"  {PASS}  全部测试通过！")
print(f"{'='*60}\n")
