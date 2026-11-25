"""CUDA logical device metadata and framework allocator measurements."""

from importlib import import_module


def cuda_device_name(device_id=0):
    """Return the name of the selected CuPy-visible CUDA device."""
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties["name"] if "name" in properties else properties[b"name"]
    return name.decode(errors="replace") if isinstance(name, bytes) else str(name)


def cuda_allocator_memory(framework, *, device=None):
    """Return allocated, reserved and peak allocated bytes for one CUDA allocator.

    Framework is a package name: torch, paddle, or tensorflow. A default
    device uses the framework's current device, or its first logical GPU
    for TensorFlow. Missing providers and devices return None.
    """
    if framework not in {"torch", "paddle", "tensorflow"}:
        raise ValueError("framework must be torch, paddle, or tensorflow")

    try:
        backend = import_module(framework)
        if framework == "torch":
            if not backend.cuda.is_available():
                return None
            stats = backend.cuda.memory_stats(device=device)
            allocated = stats.get("allocated_bytes.all.current", 0)
            reserved = stats.get("reserved_bytes.all.current", 0)
            peak = stats.get("allocated_bytes.all.peak", 0)
        elif framework == "paddle":
            if not backend.device.is_compiled_with_cuda():
                return None
            cuda = backend.device.cuda
            allocated = cuda.memory_allocated(device)
            reserved = cuda.memory_reserved(device)
            peak_fn = getattr(cuda, "max_memory_allocated", None)
            peak = peak_fn(device) if peak_fn is not None else None
        else:
            get_memory_info = backend.config.experimental.get_memory_info
            if not backend.config.list_physical_devices("GPU"):
                return None
            if device is None:
                device = backend.config.list_logical_devices("GPU")[0].name
            elif isinstance(device, int):
                device = f"GPU:{device}"
            stats = get_memory_info(device)
            allocated = stats["current"]
            reserved = None
            peak = stats.get("peak")
    except Exception:
        return None

    return {"allocated_bytes": allocated, "reserved_bytes": reserved, "peak_allocated_bytes": peak}
