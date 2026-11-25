import sys
from types import SimpleNamespace

from eneuro.profiling.cuda import cuda_device_name
from eneuro.profiling.perf import perf_stat
from eneuro.profiling.timing import CudaEventTimer


def test_cuda_timer_records_both_events_on_the_original_stream(monkeypatch):
    stream = object()
    device = object()
    records = []
    synchronized = []

    class Event:
        done = False

        def record(self, event_stream):
            records.append((self, event_stream))

        def synchronize(self):
            synchronized.append(self)
            self.done = True

    def get_elapsed_time(start, stop):
        assert (start, stream) == records[0]
        assert (stop, stream) == records[1]
        return 1.5

    cuda = SimpleNamespace(Device=lambda: device, Event=Event,
                           get_current_stream=lambda: stream,
                           get_elapsed_time=get_elapsed_time)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=cuda))

    timer = CudaEventTimer()
    timer.stop()
    assert timer.device is device
    assert timer.elapsed_ms(wait=False) is None
    assert synchronized == []
    assert timer.elapsed_ms() == 1.5
    assert timer.elapsed_ms(wait=False) == 1.5
    assert synchronized == [records[1][0]]


def test_cuda_name_uses_the_requested_logical_ordinal(monkeypatch):
    def properties(device_id):
        assert device_id == 2
        return {b"name": b"Mock GPU"}

    runtime = SimpleNamespace(getDeviceProperties=properties)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))

    assert cuda_device_name(2) == "Mock GPU"


def test_perf_stat_returns_counters_and_workload_output(monkeypatch):
    def run(argv, *, capture_output, text, check):
        assert argv == ["perf", "stat", "-x,", "-e", "cache-references,cache-misses", "sample", "--once"]
        assert capture_output and text and not check
        return SimpleNamespace(
            returncode=0, stdout="workload output",
            stderr="37,,cache-references,100,100.00\n"
                   "<not supported>,,cycles,0,0.00\n"
                   "2,,cache-misses,100,100.00\n",
        )

    monkeypatch.setattr("eneuro.profiling.perf.shutil.which", lambda name: name)
    monkeypatch.setattr("eneuro.profiling.perf.subprocess.run", run)

    assert perf_stat(["sample", "--once"], ["cache-references", "cache-misses"]) == (
        {"cache-references": 37, "cache-misses": 2}, "workload output",
    )
