from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class ProfileResult:
    runtime_s: float
    memory_mb: float


@contextmanager
def profile_block():
    proc = psutil.Process()
    start_mem = proc.memory_info().rss
    start = time.perf_counter()
    yield lambda: ProfileResult(time.perf_counter() - start, (proc.memory_info().rss - start_mem) / (1024**2))
