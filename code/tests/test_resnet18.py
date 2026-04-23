# -*- coding: utf-8 -*-
import sys
import os
import time
import ctypes
import tracemalloc
from pathlib import Path

import numpy as np

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


if os.name == 'nt':
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
            ('PrivateUsage', ctypes.c_size_t),
        ]


def _get_cpu_time_seconds_fallback():
    if os.name != 'nt':
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    ok = kernel32.GetProcessTimes(
        handle,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    )
    if not ok:
        return None

    def _to_100ns(ft):
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    return (_to_100ns(kernel) + _to_100ns(user)) / 1e7


def _get_memory_info_fallback():
    if os.name != 'nt':
        return (None, None)
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return (None, None)
    return (int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))

# 确保能导入 code 下的包
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import as_Tensor, Config  # noqa: E402
from eneuro.base import functions as F      # noqa: E402
from eneuro.nn.module import (
    Module,
    Conv2d,
    BatchNorm,
    Linear,
    ResidualBlock,
    Sequential,
)  # noqa: E402


class ResNet18(Module):
    """基于项目中 `ResidualBlock` 的 ResNet-18 风格微型实现（用于测试）。"""

    def __init__(self, num_classes=1000):
        super().__init__()
        # 初始通道数基准
        self.in_channels = 64

        # stem: 7x7 conv, stride=2, pad=3
        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=3)
        self.bn1 = BatchNorm(64)

        # 4 个 stage，每个 stage 包含若干个 ResidualBlock
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        # 分类头
        self.fc = Linear(num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        # 首个 block 可能需要下采样/升维
        downsample = (stride != 1) or (self.in_channels != out_channels)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels

        for _ in range(blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1, downsample=False))

        return Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.pooling(x, kernel_size=3, stride=2, pad=1)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global pool -> fc
        x = F.global_average_pooling(x)
        x = F.flatten(x)
        x = self.fc(x)
        return x


def count_learnable_params(model):
    total = 0
    for p in model.params():
        if getattr(p, 'requires_grad', True) and getattr(p, 'data', None) is not None:
            total += int(np.prod(p.data.shape))
    return total


def _memory_mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def collect_process_metrics(start_wall, start_cpu_time, start_rss, fallback_proc_time_start=None, fallback_trace_started=False):
    wall = time.perf_counter() - start_wall
    result = {
        'wall_s': wall,
        'cpu_avg_percent': None,
        'mem_peak_mb': None,
    }

    if psutil is None:
        end_cpu_time = _get_cpu_time_seconds_fallback()
        if start_cpu_time is not None and end_cpu_time is not None and wall > 0:
            logical_cores = os.cpu_count() or 1
            result['cpu_avg_percent'] = ((end_cpu_time - start_cpu_time) / wall) * 100.0 / logical_cores
        elif fallback_proc_time_start is not None and wall > 0:
            logical_cores = os.cpu_count() or 1
            cpu_sec = time.process_time() - fallback_proc_time_start
            result['cpu_avg_percent'] = (cpu_sec / wall) * 100.0 / logical_cores

        rss_now, peak_rss = _get_memory_info_fallback()
        if peak_rss is not None:
            result['mem_peak_mb'] = _memory_mb(peak_rss)
        elif start_rss is not None and rss_now is not None:
            result['mem_peak_mb'] = _memory_mb(max(start_rss, rss_now))
        elif fallback_trace_started and tracemalloc.is_tracing():
            _, peak_bytes = tracemalloc.get_traced_memory()
            result['mem_peak_mb'] = _memory_mb(peak_bytes)
            tracemalloc.stop()

        return result

    proc = psutil.Process(os.getpid())
    end_cpu = proc.cpu_times()
    cpu_time_delta = (end_cpu.user + end_cpu.system) - start_cpu_time
    logical_cores = psutil.cpu_count(logical=True) or 1
    if wall > 0:
        result['cpu_avg_percent'] = (cpu_time_delta / wall) * 100.0 / logical_cores

    mem_info = proc.memory_info()
    peak_bytes = getattr(mem_info, 'peak_wset', None)
    if peak_bytes is None:
        peak_bytes = max(mem_info.rss, start_rss)
    result['mem_peak_mb'] = _memory_mb(peak_bytes)
    return result


def test_resnet18_forward_backward():
    """简单的前向与反向检查，确保形状与梯度通路正常。"""
    np.random.seed(0)
    model = ResNet18(num_classes=10)
    learnable_params = count_learnable_params(model)

    proc = psutil.Process(os.getpid()) if psutil is not None else None
    fallback_proc_time_start = time.process_time()
    fallback_trace_started = False
    if proc is not None:
        start_cpu = proc.cpu_times()
        start_cpu_time = start_cpu.user + start_cpu.system
        start_rss = proc.memory_info().rss
    else:
        start_cpu_time = _get_cpu_time_seconds_fallback()
        start_rss, _ = _get_memory_info_fallback()
        if start_rss is None:
            tracemalloc.start()
            fallback_trace_started = True
    start_wall = time.perf_counter()

    # 随机输入 (batch=2, 3, 224, 224)
    x = np.random.randn(2, 3, 224, 224).astype(np.float32)
    xt = as_Tensor(x)

    # 前向
    y = model(xt)
    assert y.shape == (2, 10), f"ResNet18 输出形状错误: got={y.shape}"

    # 反向（简单平方和）并检查部分参数是否有梯度
    loss = y * y
    model.cleargrads()
    loss.backward()

    # 检查少量关键层是否有梯度
    assert model.conv1.W.grad is not None, "conv1.W 未获得梯度"
    # 检查某个 stage 的第一个 block 的 conv1 是否有梯度
    first_block = model.layer2.layers[0]
    assert first_block.conv1.W.grad is not None, "layer2 first block conv1 未获得梯度"

    print("[PASS] test_resnet18_forward_backward")
    metrics = collect_process_metrics(
        start_wall,
        start_cpu_time,
        start_rss,
        fallback_proc_time_start=fallback_proc_time_start,
        fallback_trace_started=fallback_trace_started,
    )

    print("[METRIC] wall_time_s={:.4f}".format(metrics['wall_s']))
    print("[METRIC] learnable_params={}".format(learnable_params))
    if metrics['mem_peak_mb'] is not None:
        print("[METRIC] peak_memory_mb={:.2f}".format(metrics['mem_peak_mb']))
    else:
        print("[METRIC] peak_memory_mb=N/A")
    if metrics['cpu_avg_percent'] is not None:
        print("[METRIC] cpu_avg_percent={:.2f}".format(metrics['cpu_avg_percent']))
    else:
        print("[METRIC] cpu_avg_percent=N/A")


if __name__ == '__main__':
    test_resnet18_forward_backward()
