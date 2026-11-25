import atexit

from fastapi import APIRouter

from eneuro.profiling.collectors import SystemCollector

router = APIRouter(prefix="/api/system", tags=["system"])

_collector = SystemCollector()
atexit.register(_collector.close)


def _gib(value):
    return round(value / 1024 ** 3, 2) if value is not None else None


@router.get("/stats")
def get_stats():
    snapshot = _collector.sample()
    host = snapshot["host"]
    cpu_percent = host["cpu_percent"]
    if cpu_percent is None and snapshot["sources"]["host"] == "psutil":
        cpu_percent = 0.0
    result = {
        "cpu_percent": round(cpu_percent, 1) if cpu_percent is not None else None,
        "mem_used_gb": _gib(host["memory_used_bytes"]),
        "mem_total_gb": _gib(host["memory_total_bytes"]),
        "mem_percent": host["memory_percent"],
        "gpu": None,
    }

    for gpu in snapshot["gpus"]:
        if gpu["index"] == 0 and gpu["utilization_percent"] is not None and gpu["memory_percent"] is not None:
            result["gpu"] = {
                "gpu_percent": gpu["utilization_percent"],
                "mem_used_gb": _gib(gpu["memory_used_bytes"]),
                "mem_total_gb": _gib(gpu["memory_total_bytes"]),
                "mem_percent": round(gpu["memory_percent"], 1),
            }
            break

    return result
