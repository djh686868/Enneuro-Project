import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from web_server.schemas import TrainingStartRequest
from web_server.services.task_manager import (
    start_training, stop_training, get_task, drain_queue,
)
from web_server.services.model_registry import _model_store, _model_configs

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start")
def start_training_task(req: TrainingStartRequest):
    if req.model_id not in _model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model_config = _model_configs.get(req.model_id, {})
    if not model_config:
        raise HTTPException(status_code=400, detail="Model config not found; rebuild the model")

    training_config = {
        "dataset_id": req.dataset_id,
        "epochs":     req.epochs,
        "batch_size": req.batch_size,
        "lr":         req.lr,
        "optimizer":  req.optimizer,
        "device":     req.device,
        "val_split":  req.val_split,
    }

    task_id = start_training(training_config, model_config)
    return {"task_id": task_id}


@router.get("/{task_id}/stream")
async def stream_training(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        finished = False
        while not finished:
            messages = drain_queue(task_id)
            for msg in messages:
                event_type = msg.pop("type")
                yield f"event: {event_type}\ndata: {json.dumps(msg)}\n\n"
                if event_type in ("done", "error"):
                    finished = True
            if not finished:
                await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{task_id}/stop")
def stop_task(task_id: str):
    ok = stop_training(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": ok}


@router.get("/{task_id}/status")
def task_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id":  task_id,
        "status":   task.status,
        "num_epochs_done": len(task.metrics_history),
    }
