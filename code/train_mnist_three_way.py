"""完整 MNIST LeNet 训练对比：CPU NumPy、CuPy、CUDA C RawModule。"""
from __future__ import annotations
import argparse, gzip, json, pickle, sys, time
from pathlib import Path
import numpy as np

def load_mnist(path):
    try:
        with open(path, "rb") as f: data = pickle.load(f)
    except Exception:
        with gzip.open(path, "rb") as f: data = pickle.load(f, encoding="latin1")
    if isinstance(data, tuple) and len(data) == 2: (xtr, ytr), (xte, yte) = data
    elif isinstance(data, tuple) and len(data) == 3: (xtr, ytr), _, (xte, yte) = data
    elif isinstance(data, dict):
        xtr, ytr = data.get("train_img", data.get("x_train")), data.get("train_label", data.get("y_train"))
        xte, yte = data.get("test_img", data.get("x_test")), data.get("test_label", data.get("y_test"))
    else: raise ValueError(f"unsupported MNIST format: {type(data)}")
    def prep(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2: x = x.reshape(-1, 1, 28, 28)
        elif x.ndim == 3: x = x[:, None]
        return x / 255.0 if x.max() > 1 else x
    return prep(xtr), np.asarray(ytr, dtype=np.int32), prep(xte), np.asarray(yte, dtype=np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="code/tests/testdata/MNIST_data/mnist.pkl")
    ap.add_argument("--train-samples", type=int, default=4096)
    ap.add_argument("--test-samples", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--out", default="artifacts/cuda_stage1_probe/mnist_training_three_way.json")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    import cupy as cp
    if int(cp.cuda.runtime.getDeviceCount()) < 1: raise RuntimeError("CUDA device unavailable")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eneuro.base import Tensor
    from eneuro.base.cuda import dispatch
    from eneuro.nn.loss import CrossEntropyLoss
    from eneuro.nn.module import LeNet
    from eneuro.nn.optim import Adam
    xtr, ytr, xte, yte = load_mnist(Path(args.data))
    xtr, ytr, xte, yte = xtr[:args.train_samples], ytr[:args.train_samples], xte[:args.test_samples], yte[:args.test_samples]
    rng = np.random.default_rng(20260914)
    batches = len(xtr) // args.batch_size
    results = {}
    for backend in ("numpy", "cupy", "rawmodule"):
        np.random.seed(20260914)
        model = LeNet(1, 10)
        # 先在 CPU 初始化所有延迟层，确保三方参数完全相同。
        model(Tensor(xtr[:1], device="cpu")); model.cleargrads()
        use_gpu = backend != "numpy"
        if use_gpu:
            dispatch.set_backend(backend); model.to("cuda"); xa, ya = cp.asarray(xtr), cp.asarray(ytr); xt, yt = cp.asarray(xte), cp.asarray(yte)
        else: xa, ya, xt, yt = xtr, ytr, xte, yte
        opt = Adam(list(model.params()), lr=args.lr)
        order = np.arange(len(xtr))
        train_loss, test_acc, epoch_ms = [], [], []
        try:
            for ep in range(args.epochs):
                rng_ep = np.random.default_rng(9000 + ep); rng_ep.shuffle(order)
                t0 = time.perf_counter(); losses = []
                for bi in range(batches):
                    ids = order[bi * args.batch_size:(bi + 1) * args.batch_size]
                    model.cleargrads()
                    xb = Tensor(xa[ids], device="cuda" if use_gpu else "cpu"); tb = Tensor(ya[ids], device="cuda" if use_gpu else "cpu")
                    loss = CrossEntropyLoss()(model(xb), tb); loss.backward(); opt.step()
                    losses.append(float(cp.asnumpy(loss.data)) if use_gpu else float(loss.data))
                if use_gpu: cp.cuda.Stream.null.synchronize()
                epoch_ms.append((time.perf_counter() - t0) * 1000.0)
                # 测试集只做推理，不建立反向图。
                correct = 0
                for j in range(0, len(xte), args.batch_size):
                    xb = Tensor(xt[j:j + args.batch_size], device="cuda" if use_gpu else "cpu")
                    logits = model(xb); pred = cp.asnumpy(cp.argmax(logits.data, axis=1)) if use_gpu else np.argmax(logits.data, axis=1)
                    correct += int(np.sum(pred == yte[j:j + args.batch_size]))
                train_loss.append(float(np.mean(losses))); test_acc.append(correct / len(xte))
            results[backend] = {"status": "ok", "train_loss": train_loss, "test_accuracy": test_acc, "epoch_ms": epoch_ms, "total_ms": float(np.sum(epoch_ms))}
        except Exception as exc:
            results[backend] = {"status": "unavailable", "error": repr(exc)}
    # 浮点归约和并行执行顺序不同会使长训练轨迹逐步分叉；报告 loss 最大偏差，
    # 但以测试准确率差作为“功能等价”的主判据，而不是要求逐 epoch loss 完全相等。
    loss_deltas, acc_deltas = {}, {}
    if results.get("numpy", {}).get("status") == "ok":
        for key in ("cupy", "rawmodule"):
            if results.get(key, {}).get("status") == "ok":
                loss_deltas[key] = float(np.max(np.abs(np.asarray(results[key]["train_loss"]) - np.asarray(results["numpy"]["train_loss"]))))
                acc_deltas[key] = float(np.max(np.abs(np.asarray(results[key]["test_accuracy"]) - np.asarray(results["numpy"]["test_accuracy"]))))
    if results.get("numpy", {}).get("total_ms") and results.get("cupy", {}).get("total_ms"):
        results["cupy"]["speedup_vs_numpy"] = results["numpy"]["total_ms"] / results["cupy"]["total_ms"]
    if results.get("numpy", {}).get("total_ms") and results.get("rawmodule", {}).get("total_ms"):
        results["rawmodule"]["speedup_vs_numpy"] = results["numpy"]["total_ms"] / results["rawmodule"]["total_ms"]
    if results.get("cupy", {}).get("total_ms") and results.get("rawmodule", {}).get("total_ms"):
        results["rawmodule"]["speedup_vs_cupy"] = results["cupy"]["total_ms"] / results["rawmodule"]["total_ms"]
    report = {"schema_version": 1, "dataset": str(args.data), "train_samples": len(xtr), "test_samples": len(xte), "epochs": args.epochs, "batch_size": args.batch_size, "results": results, "trajectory_delta": {"max_loss_vs_numpy": loss_deltas, "max_accuracy_vs_numpy": acc_deltas}}
    ok = [r for r in results.values() if r.get("status") == "ok"]
    report["acceptance"] = {"all_completed": len(ok) == 3, "functional_equivalence": all(v <= 0.01 for v in acc_deltas.values()), "rawmodule_faster_than_cupy": results.get("rawmodule", {}).get("total_ms", 1e99) < results.get("cupy", {}).get("total_ms", -1)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); plot = Path(args.plot) if args.plot else out.with_suffix(".png")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
        for key, label, color in (("numpy", "CPU NumPy", "#7f8c8d"), ("cupy", "CuPy", "#3498db"), ("rawmodule", "CUDA C RawModule", "#e67e22")):
            r = results.get(key, {})
            if r.get("status") == "ok":
                ax1.plot(range(1, args.epochs + 1), r["train_loss"], marker="o", label=label, color=color)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("MNIST training loss"); ax1.grid(alpha=.25); ax1.legend()
        for key, label, color in (("numpy", "CPU NumPy", "#7f8c8d"), ("cupy", "CuPy", "#3498db"), ("rawmodule", "CUDA C RawModule", "#e67e22")):
            r = results.get(key, {})
            if r.get("status") == "ok": ax2.plot(range(1, args.epochs + 1), r["test_accuracy"], marker="o", label=label, color=color)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test accuracy"); ax2.set_title("MNIST test accuracy"); ax2.grid(alpha=.25); ax2.legend()
        names, vals = [], []
        for key, label in (("numpy", "CPU NumPy"), ("cupy", "CuPy"), ("rawmodule", "CUDA C RawModule")):
            if results.get(key, {}).get("status") == "ok": names.append(label); vals.append(results[key]["total_ms"] / 1000)
        bars = ax3.bar(names, vals, color=["#7f8c8d", "#3498db", "#e67e22"][:len(vals)]); ax3.set_ylabel("Total training time (s)"); ax3.set_title("MNIST LeNet training time")
        for bar, v in zip(bars, vals): ax3.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}s", ha="center", va="bottom")
        fig.tight_layout(); plot.parent.mkdir(parents=True, exist_ok=True); fig.savefig(plot); plt.close(fig); report["plot"] = str(plot)
    except Exception as exc: report["plot_error"] = repr(exc)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    # 退出码只反映功能验收；性能是否超过 CuPy 由独立字段报告，不把“尚需优化”
    # 误写成训练失败。
    return 0 if report["acceptance"]["all_completed"] and report["acceptance"]["functional_equivalence"] else 1

if __name__ == "__main__": raise SystemExit(main())
