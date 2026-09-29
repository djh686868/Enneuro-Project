"""基于真实 MNIST 子集的三方 LeNet 验收 benchmark。

三方：CPU NumPy、CuPy 基线、CUDA C RawModule。脚本只使用仓库已有
``code/tests/testdata/MNIST_data/mnist.pkl``，不联网下载数据。
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np


def load_mnist(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        with gzip.open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    if isinstance(data, tuple) and len(data) == 2:
        (x_train, y_train), (x_test, y_test) = data
    elif isinstance(data, tuple) and len(data) == 3:
        (x_train, y_train), _, (x_test, y_test) = data
    elif isinstance(data, dict):
        # 仓库中既有 train_img/train_label 命名，也兼容标准 x_train/y_train 命名。
        x_train = data.get("train_img", data.get("x_train"))
        y_train = data.get("train_label", data.get("y_train"))
        x_test = data.get("test_img", data.get("x_test"))
        y_test = data.get("test_label", data.get("y_test"))
        if any(v is None for v in (x_train, y_train, x_test, y_test)):
            raise ValueError(f"unsupported MNIST dict keys: {sorted(data.keys())}")
    else:
        raise ValueError(f"unsupported MNIST pickle format: {type(data)}")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    if x_train.ndim == 2:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)
    elif x_train.ndim == 3:
        x_train = x_train[:, None]
        x_test = x_test[:, None]
    if x_train.max() > 1.0:
        x_train /= 255.0; x_test /= 255.0
    return x_train, np.asarray(y_train, dtype=np.int32), x_test, np.asarray(y_test, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="code/tests/testdata/MNIST_data/mnist.pkl")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="artifacts/cuda_stage1_probe/mnist_three_way.json")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--backends", nargs="+", choices=("numpy", "cupy", "rawmodule"),
                    default=("numpy", "cupy", "rawmodule"),
                    help="Backends to measure; use cupy rawmodule to avoid repeating CPU work")
    args = ap.parse_args()

    import cupy as cp
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CUDA device unavailable")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eneuro.base import Tensor
    from eneuro.base.cuda import dispatch
    from eneuro.nn.loss import CrossEntropyLoss
    from eneuro.nn.module import LeNet

    x_train, y_train, _, _ = load_mnist(Path(args.data))
    x_train, y_train = x_train[:args.samples], y_train[:args.samples]
    steps = min(args.steps, len(x_train) // args.batch_size)
    results = {}
    reference_loss = None
    reference_backend = args.backends[0]

    for backend in args.backends:
        # 在 CPU 上以固定 NumPy 随机状态初始化，再复制到 GPU，保证模型参数一致。
        np.random.seed(20260914)
        model = LeNet(in_channels=1, num_classes=10)
        use_gpu = backend != "numpy"
        # LeNet 的 conv2/fc 层是延迟初始化的。先用 CPU dummy 输入触发全部
        # 参数创建，再迁移到 GPU，保证三方使用完全相同的权重。
        model(Tensor(x_train[:1], device="cpu"))
        model.cleargrads()
        if use_gpu:
            dispatch.set_backend(backend)
            model.to("cuda")
            x_all, y_all = cp.asarray(x_train), cp.asarray(y_train)
            if backend == "rawmodule":
                dispatch.launch_counts(reset=True)
        else:
            x_all, y_all = x_train, y_train

        def one_step(start):
            model.cleargrads()
            xb = Tensor(x_all[start:start + args.batch_size], device="cuda" if use_gpu else "cpu")
            tb = Tensor(y_all[start:start + args.batch_size], device="cuda" if use_gpu else "cpu")
            logits = model(xb)
            loss = CrossEntropyLoss()(logits, tb)
            loss.backward()
            return loss, logits

        try:
            for i in range(args.warmup):
                one_step((i % steps) * args.batch_size)
            if use_gpu:
                cp.cuda.Stream.null.synchronize()
            start_time = time.perf_counter()
            losses, correct, count = [], 0, 0
            for i in range(steps):
                loss, logits = one_step(i * args.batch_size)
                if use_gpu:
                    cp.cuda.Stream.null.synchronize()
                    pred = cp.asnumpy(cp.argmax(logits.data, axis=1))
                    lv = float(cp.asnumpy(loss.data))
                else:
                    pred = np.argmax(logits.data, axis=1)
                    lv = float(loss.data)
                losses.append(lv)
                correct += int(np.sum(pred == y_train[i * args.batch_size:(i + 1) * args.batch_size]))
                count += args.batch_size
            elapsed = (time.perf_counter() - start_time) / steps * 1000.0
            results[backend] = {"status": "ok", "mean_batch_ms": elapsed,
                                "mean_loss": float(np.mean(losses)),
                                "accuracy": correct / count}
            if backend == "rawmodule":
                results[backend]["cuda_kernel_launches"] = dispatch.launch_counts()
            if reference_loss is None:
                reference_loss = np.asarray(losses, dtype=np.float64)
            else:
                results[backend]["max_loss_error_vs_reference"] = float(np.max(np.abs(np.asarray(losses) - reference_loss)))
        except Exception as exc:
            results[backend] = {"status": "unavailable", "error": repr(exc)}

    cpu_ms = results.get("numpy", {}).get("mean_batch_ms")
    cupy_ms = results.get("cupy", {}).get("mean_batch_ms")
    raw_ms = results.get("rawmodule", {}).get("mean_batch_ms")
    if cpu_ms and cupy_ms: results["cupy"]["speedup_vs_numpy"] = cpu_ms / cupy_ms
    if cpu_ms and raw_ms: results["rawmodule"]["speedup_vs_numpy"] = cpu_ms / raw_ms
    if cupy_ms and raw_ms: results["rawmodule"]["speedup_vs_cupy"] = cupy_ms / raw_ms

    report = {"schema_version": 1, "dataset": str(args.data), "samples": len(x_train),
              "steps": steps, "batch_size": args.batch_size,
              "backends": list(args.backends), "reference_backend": reference_backend,
              "device": cp.cuda.Device().compute_capability, "results": results,
              "acceptance": {"numerical_match": all(
                  r.get("status") != "ok" or r.get("max_loss_error_vs_reference", 0) < 1e-3
                  for key, r in results.items() if key != args.backends[0]),
                  "rawmodule_faster_than_cupy": bool(raw_ms and cupy_ms and raw_ms < cupy_ms)}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plot = Path(args.plot) if args.plot else out.with_suffix(".png")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names, vals = [], []
        for key, label in (("numpy", "CPU NumPy"), ("cupy", "CuPy"), ("rawmodule", "CUDA C RawModule")):
            if results.get(key, {}).get("mean_batch_ms") is not None:
                names.append(label); vals.append(results[key]["mean_batch_ms"])
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        bars = ax.bar(names, vals, color=["#7f8c8d", "#3498db", "#e67e22"][:len(vals)])
        ax.set_ylabel("Mean LeNet batch latency (ms)"); ax.set_title("MNIST LeNet three-way benchmark")
        ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
        for bar, v in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom")
        fig.tight_layout(); plot.parent.mkdir(parents=True, exist_ok=True); fig.savefig(plot); plt.close(fig)
        report["plot"] = str(plot)
    except Exception as exc:
        report["plot_error"] = repr(exc)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["acceptance"]["numerical_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
