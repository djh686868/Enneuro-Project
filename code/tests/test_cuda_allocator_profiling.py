from types import SimpleNamespace

import pytest

from eneuro.profiling import cuda


def test_torch_stats_use_one_allocator_snapshot(monkeypatch):
    def stats(device):
        assert device == 2
        return {
            "allocated_bytes.all.current": 75,
            "reserved_bytes.all.current": 128,
            "allocated_bytes.all.peak": 100,
        }

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, memory_stats=stats))
    monkeypatch.setattr(cuda, "import_module", {"torch": torch}.__getitem__)
    assert cuda.cuda_allocator_memory("torch", device=2) == {
        "allocated_bytes": 75,
        "reserved_bytes": 128,
        "peak_allocated_bytes": 100,
    }


def test_paddle_and_tensorflow_report_supported_allocator_fields(monkeypatch):
    paddle_cuda = SimpleNamespace(memory_allocated=lambda device: 45, memory_reserved=lambda device: 90)
    paddle = SimpleNamespace(device=SimpleNamespace(is_compiled_with_cuda=lambda: True, cuda=paddle_cuda))
    gpu = SimpleNamespace(name="/device:GPU:0")
    stats = {gpu.name: {"current": 65, "peak": 95}, "GPU:1": {"current": 4, "peak": 9}}
    tensorflow = SimpleNamespace(config=SimpleNamespace(
        list_physical_devices=lambda kind: [gpu],
        list_logical_devices=lambda kind: [gpu],
        experimental=SimpleNamespace(get_memory_info=stats.__getitem__),
    ))
    modules = {"paddle": paddle, "tensorflow": tensorflow}
    monkeypatch.setattr(cuda, "import_module", modules.__getitem__)

    assert cuda.cuda_allocator_memory("paddle") == {
        "allocated_bytes": 45,
        "reserved_bytes": 90,
        "peak_allocated_bytes": None,
    }
    assert cuda.cuda_allocator_memory("tensorflow") == {
        "allocated_bytes": 65,
        "reserved_bytes": None,
        "peak_allocated_bytes": 95,
    }
    assert cuda.cuda_allocator_memory("tensorflow", device=1)["allocated_bytes"] == 4


def test_unavailable_allocator_and_invalid_provider(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(cuda, "import_module", lambda name: torch)
    assert cuda.cuda_allocator_memory("torch") is None
    with pytest.raises(ValueError, match="framework"):
        cuda.cuda_allocator_memory("other")
