import time
import queue
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

# run_id -> { meta, queue_list, history }
_runs: dict[str, dict] = {}


def _get_or_create(run_id: str, run_name: str = "") -> dict:
    if run_id not in _runs:
        _runs[run_id] = {
            "run_id":   run_id,
            "run_name": run_name or run_id,
            "start_ts": time.time(),
            "last_ts":  time.time(),
            "queues":   [],      # asyncio.Queue per SSE subscriber
            "history":  [],      # all pushed events (for late joiners)
        }
    return _runs[run_id]


@router.post("/push")
def push_metrics(body: dict):
    run_id   = body.get("run_id", "default")
    run_name = body.get("run_name", run_id)
    run = _get_or_create(run_id, run_name)
    run["last_ts"] = time.time()

    event = {k: v for k, v in body.items() if k not in ("run_id", "run_name")}
    run["history"].append(event)
    # keep history bounded
    if len(run["history"]) > 5000:
        run["history"] = run["history"][-5000:]

    for q in run["queues"]:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

    return {"ok": True}


@router.get("/runs")
def list_runs():
    return [
        {
            "run_id":   r["run_id"],
            "run_name": r["run_name"],
            "start_ts": r["start_ts"],
            "last_ts":  r["last_ts"],
            "events":   len(r["history"]),
        }
        for r in _runs.values()
    ]


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    _runs.pop(run_id, None)
    return {"ok": True}


@router.get("/stream/{run_id}")
async def stream(run_id: str):
    run = _get_or_create(run_id)
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    run["queues"].append(q)

    async def generate():
        # 先把历史数据发给新连接的客户端
        import json
        for ev in run["history"]:
            yield f"data: {json.dumps(ev)}\n\n"
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            run["queues"].remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
