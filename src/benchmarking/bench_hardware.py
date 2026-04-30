"""Background hardware sampler — CPU%, memory (RSS), and active thread count.

Runs a daemon thread that polls ``psutil`` at a fixed interval while a strategy
is executing.  Call ``start()`` before the strategy and ``stop()`` immediately
after; ``summary()`` returns the collected samples plus aggregate statistics.

Usage::

    sampler = HardwareSampler(interval_s=0.5)
    sampler.start()
    # ... run strategy ...
    sampler.stop()
    data = sampler.summary()   # dict ready to embed in a benchmark record
"""

from __future__ import annotations

import threading
import time
from typing import Any

import psutil


class HardwareSampler:
    """Polls hardware metrics in a daemon thread while a benchmark strategy runs.

    Args:
        interval_s: Seconds between each sample. Default 0.5 s.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval = interval_s
        self._cpu:  list[float] = []
        self._mem:  list[float] = []   # RSS in MB
        self._thr:  list[int]   = []   # active Python thread count
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock    = threading.Lock()
        self._process = psutil.Process()

    # Lifecycle

    def start(self) -> None:
        """Begin sampling. Clears any data from a previous run."""
        with self._lock:
            self._cpu.clear()
            self._mem.clear()
            self._thr.clear()
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="bench-hw-sampler"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the sampler thread to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # Sampling loop

    def _loop(self) -> None:
        # First call to cpu_percent always returns 0.0 — prime it before the loop.
        psutil.cpu_percent(interval=None)
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=None)          # system-wide %
                mem = self._process.memory_info().rss / (1024 * 1024)  # MB
                thr = threading.active_count()
                with self._lock:
                    self._cpu.append(round(cpu, 1))
                    self._mem.append(round(mem, 1))
                    self._thr.append(thr)
            except Exception:
                pass  # never crash the sampler thread
            time.sleep(self._interval)

    # Results

    def summary(self) -> dict[str, Any]:
        """Return a dict with raw samples and derived aggregates.

        Returns an empty-samples dict if no data was collected (e.g. strategy
        ran faster than one interval).
        """
        with self._lock:
            cpu = list(self._cpu)
            mem = list(self._mem)
            thr = list(self._thr)

        base: dict[str, Any] = {
            "sample_interval_s":   self._interval,
            "cpu_pct_samples":     cpu,
            "mem_rss_mb_samples":  mem,
            "thread_count_samples": thr,
        }

        if not cpu:
            return base

        base.update({
            "cpu_pct_mean":       round(sum(cpu) / len(cpu), 1),
            "cpu_pct_max":        round(max(cpu), 1),
            "mem_rss_mb_mean":    round(sum(mem) / len(mem), 1),
            "mem_rss_mb_max":     round(max(mem), 1),
            "thread_count_mean":  round(sum(thr) / len(thr), 1),
            "thread_count_max":   max(thr),
        })
        return base
