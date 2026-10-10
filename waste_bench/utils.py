"""Logging, seeding, timing and GPU-utilization helpers."""

import logging
import random
import threading

import numpy as np
import torch

logger = logging.getLogger("benchmark_backbones")


def setup_logging(log_file=None):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed):
    """Seeds python/numpy/torch so a run is repeatable; shuffling and the
    per-sample augmentation both draw from these generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    """Renders a duration in seconds as e.g. '1h 12m 04s' (omitting leading
    zero units), so long training runs read as hours/minutes instead of a
    raw, hard-to-parse second count."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class GpuUtilSampler:
    """Polls GPU utilization % on a background thread while a block runs.
    Silently produces no samples (avg/max report as None) if no NVML/GPU is
    available, so this is always safe to use even on CPU-only machines."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._backend = None
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._pynvml = pynvml
                self._backend = "pynvml"
            except Exception:
                self._backend = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                if self._backend == "pynvml":
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self.samples.append(util.gpu)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._backend is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False

    def summary(self):
        if not self.samples:
            return {"gpu_util_avg_pct": None, "gpu_util_max_pct": None}
        return {"gpu_util_avg_pct": float(np.mean(self.samples)),
                "gpu_util_max_pct": float(np.max(self.samples))}
