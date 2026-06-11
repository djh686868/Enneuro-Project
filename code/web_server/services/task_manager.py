"""
训练任务管理器。
每个训练任务在独立线程中运行（避免 Windows 多进程 spawn 限制），
通过 queue.Queue 将指标推送回主线程。
"""
import uuid
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))


@dataclass
class TrainingTask:
    task_id: str
    thread: Optional[threading.Thread] = None
    q: Optional[queue.Queue] = None
    status: str = "pending"
    metrics_history: list = field(default_factory=list)
    trainer_ref: Optional[object] = None


_tasks: dict[str, TrainingTask] = {}


def _training_worker(config: dict, q: queue.Queue, task: TrainingTask):
    """线程入口：实例化模型、加载数据、运行训练。"""
    try:
        from eneuro.train.trainer import Trainer
        from eneuro.nn.loss import SoftmaxWithLoss
        from eneuro.nn.optim import Adam, SGD, MomentumSGD
        from web_server.services.model_registry import build_model_from_config
        from web_server.services.dataset_manager import load_dataset_for_training

        task.status = "running"
        q.put({"type": "status", "status": "running"})

        _, model = build_model_from_config(config["model_config"])

        train_loader, val_loader = load_dataset_for_training(
            dataset_id=config["dataset_id"],
            batch_size=config["batch_size"],
            val_split=config["val_split"],
        )

        optim_map = {"adam": Adam, "sgd": SGD, "momentum_sgd": MomentumSGD}
        optimizer_cls = optim_map.get(config["optimizer"], Adam)
        optimizer = optimizer_cls(model.params(), lr=config["lr"])
        loss_fn = SoftmaxWithLoss()

        _train_metrics = {"train_loss": 0.0, "train_acc": 0.0}

        def on_epoch_end(metrics: dict):
            q.put({"type": "epoch", **metrics})
            task.metrics_history.append(metrics)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            on_epoch_end=on_epoch_end,
        )
        task.trainer_ref = trainer

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            device=config["device"],
            verbose=False,
        )
        task.status = "finished"
        q.put({"type": "done", "status": "finished"})

    except Exception as e:
        import traceback
        task.status = "error"
        q.put({"type": "error", "message": str(e), "traceback": traceback.format_exc()})


def start_training(training_config: dict, model_config: dict) -> str:
    task_id = str(uuid.uuid4())[:8]
    q = queue.Queue()
    full_config = {**training_config, "model_config": model_config}

    task = TrainingTask(task_id=task_id, q=q)
    _tasks[task_id] = task

    t = threading.Thread(
        target=_training_worker,
        args=(full_config, q, task),
        daemon=True,
    )
    task.thread = t
    t.start()
    return task_id


def stop_training(task_id: str) -> bool:
    task = _tasks.get(task_id)
    if task is None:
        return False
    if task.trainer_ref is not None:
        task.trainer_ref.request_stop()
    task.status = "stopped"
    task.q.put({"type": "done", "status": "stopped"})
    return True


def get_task(task_id: str) -> Optional[TrainingTask]:
    return _tasks.get(task_id)


def drain_queue(task_id: str) -> list[dict]:
    """读取队列中所有当前可用消息（非阻塞）。"""
    task = _tasks.get(task_id)
    if task is None:
        return []
    messages = []
    while True:
        try:
            msg = task.q.get_nowait()
            messages.append(msg)
            if msg["type"] in ("done", "error"):
                if msg["type"] != "done" or task.status not in ("stopped",):
                    task.status = msg.get("status", msg["type"])
        except queue.Empty:
            break
    return messages
