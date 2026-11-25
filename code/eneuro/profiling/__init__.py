"""Optional training instrumentation and system telemetry."""

from .collectors import SystemCollector, cuda_device_memory, process_snapshot, system_snapshot

__all__ = [
    "ProfileConfig", "ProfileSession", "SystemCollector", "cuda_device_memory",
    "current_profile", "process_snapshot", "system_snapshot",
]


def __getattr__(name):
    if name in {"ProfileConfig", "ProfileSession", "current_profile"}:
        from . import session

        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
