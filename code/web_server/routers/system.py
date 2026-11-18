import psutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])

_nvml_ok = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    pass


@router.get("/stats")
def get_stats():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)

    result = {
        "cpu_percent":  cpu,
        "mem_used_gb":  round(mem.used  / 1024 ** 3, 2),
        "mem_total_gb": round(mem.total / 1024 ** 3, 2),
        "mem_percent":  mem.percent,
        "gpu": None,
    }

    if _nvml_ok:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result["gpu"] = {
                "gpu_percent":  util.gpu,
                "mem_used_gb":  round(info.used  / 1024 ** 3, 2),
                "mem_total_gb": round(info.total / 1024 ** 3, 2),
                "mem_percent":  round(info.used / info.total * 100, 1),
            }
        except Exception:
            pass

    return result
