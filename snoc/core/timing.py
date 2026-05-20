"""Timing utilities: decorator, context manager, and torch device detection."""

from __future__ import annotations

import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"'{func.__name__}' took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper


class TimerContext:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.end     = time.perf_counter()
        self.elapsed = self.end - self.start


def get_device(always_cpu: bool = False):
    """Return best available torch device (MPS > CUDA > CPU). Returns None if torch absent."""
    try:
        import torch
        if always_cpu:
            return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None
