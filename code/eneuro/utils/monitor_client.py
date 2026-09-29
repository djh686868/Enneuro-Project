"""
MonitorClient — 向 EnNeuro Web 监控器推送训练指标。

用法（在训练脚本中加 3 行）：

    from eneuro.utils.monitor_client import MonitorClient
    monitor = MonitorClient(run_name="my_experiment")
    trainer = Trainer(model, loss_fn, optimizer,
                      on_epoch_end=monitor.on_epoch_end,
                      on_batch_end=monitor.on_batch_end)

然后在浏览器打开 http://localhost:8000/app → 「外部监控」即可实时查看。
"""

import uuid
import time
import threading
import queue as _queue


class MonitorClient:
    def __init__(self, host: str = "localhost", port: int = 8000,
                 run_name: str = None):
        self.run_id   = str(uuid.uuid4())[:8]
        self.run_name = run_name or f"run_{self.run_id}"
        self._url     = f"http://{host}:{port}/api/monitor/push"
        self._q: _queue.Queue = _queue.Queue()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[MonitorClient] run_id={self.run_id}  "
              f"监控地址: http://{host}:{port}/app  (切换到「外部监控」Tab)")

    # ── Trainer 回调（可直接传给 on_epoch_end / on_batch_end）──────────────

    def on_epoch_end(self, metrics: dict):
        self._push({"type": "epoch", **metrics})

    def on_batch_end(self, metrics: dict):
        self._push({"type": "batch", **metrics})

    def on_train_end(self, metrics: dict = None):
        self._push({"type": "done", **(metrics or {})})
        # 确保 done 事件在队列清空后再退出
        import time; time.sleep(0.5)

    # ── 手动推送 ────────────────────────────────────────────────────────────

    def push(self, **kwargs):
        self._push(kwargs)

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _push(self, payload: dict):
        payload["run_id"]   = self.run_id
        payload["run_name"] = self.run_name
        payload["ts"]       = time.time()
        self._q.put_nowait(payload)

    def _worker(self):
        try:
            import urllib.request, json
        except ImportError:
            return
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=1)
            except _queue.Empty:
                continue
            try:
                data = json.dumps(payload).encode()
                req  = urllib.request.Request(
                    self._url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                pass  # web server 未启动时静默忽略

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)
