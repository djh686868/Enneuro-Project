"""CUDA event timing shared by training profiles and GPU benchmarks."""


class CudaEventTimer:
    """Measure elapsed milliseconds on the CUDA stream current at construction.

    Construction records the start event. Call stop and elapsed_ms on the
    recorded device, using the device context if the active device may change.
    """

    def __init__(self):
        import cupy as cp

        self._cuda = cp.cuda
        self.device = self._cuda.Device()
        self._stream = self._cuda.get_current_stream()
        self._start = self._cuda.Event()
        self._end = self._cuda.Event()
        self._start.record(self._stream)

    def stop(self):
        self._end.record(self._stream)

    def elapsed_ms(self, *, wait=True):
        """Wait for the stop event or return None while it is still pending."""
        if wait:
            self._end.synchronize()
        elif not self._end.done:
            return None
        return float(self._cuda.get_elapsed_time(self._start, self._end))
