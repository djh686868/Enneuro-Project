"""
eneuro/train/metrics.py

分类任务评价指标：
  - accuracy          Top-K 准确率
  - confusion_matrix  混淆矩阵
  - precision_recall_f1  Precision / Recall / F1（macro / micro / weighted / per-class）
  - ClassificationReport  格式化报告（类 sklearn.metrics.classification_report）

所有函数接受 numpy ndarray（类别索引，int），不依赖 Tensor。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    """将 Tensor / cupy 数组 / list 统一转为 numpy ndarray (int64)。"""
    if hasattr(arr, 'data'):          # eneuro Tensor
        arr = arr.data
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr, dtype=np.int64).ravel()


# ---------------------------------------------------------------------------
# Top-K Accuracy
# ---------------------------------------------------------------------------

def accuracy(logits_or_pred, y_true, topk=(1,)):
    """
    计算 Top-K 准确率。

    参数
    ----
    logits_or_pred : array-like, shape (N,) 或 (N, C)
        若为 (N,)，视为已预测的类别索引。
        若为 (N, C)，按每行最大的 k 个索引判断。
    y_true : array-like, shape (N,)
        真实类别索引。
    topk : tuple of int
        需要计算的 K 值，默认 (1,)。

    返回
    ----
    dict  {k: float}，每个 k 对应的准确率（0~1）。

    示例
    ----
    >>> accuracy(logits, labels, topk=(1, 5))
    {1: 0.83, 5: 0.96}
    """
    arr = logits_or_pred
    # eneuro Tensor → 取底层数组
    if type(arr).__name__ == 'Tensor' and hasattr(arr, 'data'):
        arr = arr.data
    # cupy → numpy
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
    except ImportError:
        pass
    arr = np.asarray(arr)

    y_true_np = _to_numpy(y_true)
    N = len(y_true_np)

    results = {}

    if arr.ndim == 1:
        # 已经是类别预测
        pred_cls = arr.astype(np.int64).ravel()
        for k in topk:
            if k == 1:
                results[k] = float((pred_cls == y_true_np).mean())
            else:
                # 无 logits，top-k 无意义，退化为 top-1
                results[k] = results.get(1, float((pred_cls == y_true_np).mean()))
        return results

    # arr: (N, C)
    # argsort 降序，取前 max(topk) 列
    max_k = max(topk)
    # 降序排列每行的列索引
    top_indices = np.argsort(arr, axis=1)[:, ::-1][:, :max_k]   # (N, max_k)

    for k in topk:
        top_k_pred = top_indices[:, :k]                          # (N, k)
        correct = np.any(top_k_pred == y_true_np[:, None], axis=1)
        results[k] = float(correct.mean())

    return results


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def confusion_matrix(y_pred, y_true, num_classes=None):
    """
    计算混淆矩阵。

    参数
    ----
    y_pred : array-like, shape (N,)  预测类别索引
    y_true : array-like, shape (N,)  真实类别索引
    num_classes : int | None
        类别数。为 None 时自动推断为 max(y_true, y_pred) + 1。

    返回
    ----
    cm : ndarray, shape (num_classes, num_classes)
        cm[i, j] = 真实类别为 i、预测为 j 的样本数。
    """
    y_pred_np = _to_numpy(y_pred)
    y_true_np = _to_numpy(y_true)

    if num_classes is None:
        num_classes = int(max(y_true_np.max(), y_pred_np.max())) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_np, y_pred_np):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

def precision_recall_f1(y_pred, y_true, num_classes=None, average='macro', eps=1e-8):
    """
    计算 Precision、Recall、F1-score。

    参数
    ----
    y_pred : array-like, shape (N,)
    y_true : array-like, shape (N,)
    num_classes : int | None
    average : str
        'macro'    : 各类等权平均
        'micro'    : 全局 TP/FP/FN 汇总后计算
        'weighted' : 按真实样本数加权平均
        'none'     : 返回每个类别的独立结果
    eps : float  防止除零

    返回
    ----
    若 average != 'none'：
        (precision, recall, f1)  三个 float
    若 average == 'none'：
        (precision_arr, recall_arr, f1_arr)  三个 ndarray, shape (num_classes,)
    """
    y_pred_np = _to_numpy(y_pred)
    y_true_np = _to_numpy(y_true)

    if num_classes is None:
        num_classes = int(max(y_true_np.max(), y_pred_np.max())) + 1

    # 逐类统计 TP、FP、FN
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    support = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        pred_c = (y_pred_np == c)
        true_c = (y_true_np == c)
        tp[c] = (pred_c & true_c).sum()
        fp[c] = (pred_c & ~true_c).sum()
        fn[c] = (~pred_c & true_c).sum()
        support[c] = true_c.sum()

    per_precision = tp / (tp + fp + eps)
    per_recall    = tp / (tp + fn + eps)
    per_f1        = 2 * per_precision * per_recall / (per_precision + per_recall + eps)

    if average == 'none':
        return per_precision, per_recall, per_f1

    if average == 'micro':
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        p = tp_sum / (tp_sum + fp_sum + eps)
        r = tp_sum / (tp_sum + fn_sum + eps)
        f = 2 * p * r / (p + r + eps)
        return float(p), float(r), float(f)

    if average == 'macro':
        return float(per_precision.mean()), float(per_recall.mean()), float(per_f1.mean())

    if average == 'weighted':
        w = support / (support.sum() + eps)
        return (
            float((per_precision * w).sum()),
            float((per_recall    * w).sum()),
            float((per_f1        * w).sum()),
        )

    raise ValueError(f"average must be 'macro'/'micro'/'weighted'/'none', got '{average}'")


# ---------------------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------------------

class ClassificationReport:
    """
    格式化输出各类别及汇总的 Precision / Recall / F1 / Support，
    风格与 sklearn.metrics.classification_report 一致。

    用法
    ----
    report = ClassificationReport(y_pred, y_true, num_classes=10)
    print(report)           # 打印报告字符串
    d = report.to_dict()    # 获取字典形式结果
    """

    def __init__(self, y_pred, y_true, num_classes=None, class_names=None, eps=1e-8):
        self.y_pred = _to_numpy(y_pred)
        self.y_true = _to_numpy(y_true)

        if num_classes is None:
            num_classes = int(max(self.y_true.max(), self.y_pred.max())) + 1
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.eps = eps

        self._compute()

    def _compute(self):
        per_p, per_r, per_f1 = precision_recall_f1(
            self.y_pred, self.y_true, self.num_classes, average='none', eps=self.eps
        )
        support = np.array(
            [(self.y_true == c).sum() for c in range(self.num_classes)], dtype=np.float64
        )
        acc_val = accuracy(self.y_pred, self.y_true, topk=(1,))[1]

        self.per_precision = per_p
        self.per_recall    = per_r
        self.per_f1        = per_f1
        self.support       = support
        self.acc           = acc_val

        # 汇总
        self.macro_p,    self.macro_r,    self.macro_f1    = precision_recall_f1(
            self.y_pred, self.y_true, self.num_classes, average='macro',    eps=self.eps)
        self.micro_p,    self.micro_r,    self.micro_f1    = precision_recall_f1(
            self.y_pred, self.y_true, self.num_classes, average='micro',    eps=self.eps)
        self.weighted_p, self.weighted_r, self.weighted_f1 = precision_recall_f1(
            self.y_pred, self.y_true, self.num_classes, average='weighted', eps=self.eps)

    def __str__(self):
        col_w = max(len(n) for n in self.class_names) + 2
        header = f"{'':>{col_w}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}"
        lines = [header, '']

        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:>{col_w}}  {self.per_precision[i]:9.4f}  "
                f"{self.per_recall[i]:9.4f}  {self.per_f1[i]:9.4f}  "
                f"{int(self.support[i]):9d}"
            )

        total = int(self.support.sum())
        lines.append('')
        lines.append(
            f"{'accuracy':>{col_w}}  {'':9}  {'':9}  {self.acc:9.4f}  {total:9d}"
        )
        lines.append(
            f"{'macro avg':>{col_w}}  {self.macro_p:9.4f}  "
            f"{self.macro_r:9.4f}  {self.macro_f1:9.4f}  {total:9d}"
        )
        lines.append(
            f"{'weighted avg':>{col_w}}  {self.weighted_p:9.4f}  "
            f"{self.weighted_r:9.4f}  {self.weighted_f1:9.4f}  {total:9d}"
        )
        return '\n'.join(lines)

    def to_dict(self):
        d = {}
        for i, name in enumerate(self.class_names):
            d[name] = {
                'precision': float(self.per_precision[i]),
                'recall':    float(self.per_recall[i]),
                'f1-score':  float(self.per_f1[i]),
                'support':   int(self.support[i]),
            }
        d['accuracy']     = float(self.acc)
        d['macro avg']    = {'precision': self.macro_p,    'recall': self.macro_r,    'f1-score': self.macro_f1}
        d['micro avg']    = {'precision': self.micro_p,    'recall': self.micro_r,    'f1-score': self.micro_f1}
        d['weighted avg'] = {'precision': self.weighted_p, 'recall': self.weighted_r, 'f1-score': self.weighted_f1}
        return d
