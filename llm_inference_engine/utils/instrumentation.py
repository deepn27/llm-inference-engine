from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import torch


class Instrumentation:

    def __init__(self, enabled: bool = False, nvtx: bool = False):
        self.enabled = enabled
        self.nvtx = nvtx
        self.cpu_ms = defaultdict(float)
        self.gpu_ms = defaultdict(float)
        self._pending_gpu_events = []

    def reset(self):
        self.cpu_ms.clear()
        self.gpu_ms.clear()
        self._pending_gpu_events.clear()

    @contextmanager
    def cpu_range(self, name: str):
        if self.nvtx:
            torch.cuda.nvtx.range_push(name)
        start = perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                self.cpu_ms[name] += (perf_counter() - start) * 1000
            if self.nvtx:
                torch.cuda.nvtx.range_pop()

    @contextmanager
    def gpu_range(self, name: str):
        if self.nvtx:
            torch.cuda.nvtx.range_push(name)
        start = end = None
        if self.enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if self.enabled:
                end.record()
                self._pending_gpu_events.append((name, start, end))
            if self.nvtx:
                torch.cuda.nvtx.range_pop()

    def flush_gpu(self):
        if not self._pending_gpu_events:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending_gpu_events:
            self.gpu_ms[name] += start.elapsed_time(end)
        self._pending_gpu_events.clear()

    def snapshot(self):
        return {
            "cpu_ms": dict(self.cpu_ms),
            "gpu_ms": dict(self.gpu_ms),
        }