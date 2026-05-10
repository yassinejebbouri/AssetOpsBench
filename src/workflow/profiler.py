"""Hardware profiler for measuring resource usage of individual tool calls."""

from __future__ import annotations

import os
import threading
import time

import psutil


class HardwareProfiler:
    """Context manager that records CPU, RAM, disk I/O, and wall time for a single call.

    Usage:
        with HardwareProfiler(server="IoTAgent", tool="history", scenario_id="iot_001") as prof:
            result = do_work()
        print(prof.to_dict())
    """

    def __init__(
        self,
        server: str,
        tool: str,
        scenario_id: str = "",
        orchestration: str = "mcp",
        run_id: int = 0,
    ):
        self.server = server
        self.tool = tool
        self.scenario_id = scenario_id
        self.orchestration = orchestration
        self.run_id = run_id

        # all set in __exit__
        self.wall_time_s = 0.0        # total elapsed time for the call
        self.cpu_percent_peak = 0.0   # highest CPU burst recorded during the call
        self.ram_mb_start = 0.0       # process RAM before the call starts
        self.ram_mb_peak = 0.0        # maximum RAM during the call
        self.ram_mb_end = 0.0         # process RAM after the call completes
        self.io_read_bytes = 0        # bytes read from disk during the call

        self._sampling = False
        self._samples_cpu: list[float] = []
        self._samples_ram: list[float] = []

    def _sample_loop(self) -> None:
        # background thread that polls process metrics every 100ms
        proc = psutil.Process(os.getpid())
        proc.cpu_percent()  # first call always returns 0.0, discard it
        while self._sampling:
            self._samples_cpu.append(proc.cpu_percent())
            self._samples_ram.append(proc.memory_info().rss / 1024 / 1024)
            time.sleep(0.1)

    def __enter__(self) -> "HardwareProfiler":
        proc = psutil.Process(os.getpid())
        self.ram_mb_start = proc.memory_info().rss / 1024 / 1024
        try:
            self._io_start = proc.io_counters().read_bytes
        except (AttributeError, psutil.AccessDenied, NotImplementedError):
            # io_counters not available on all platforms (e.g. macOS with SIP)
            self._io_start = 0
        self._samples_cpu = []
        self._samples_ram = []
        self._t_start = time.perf_counter()
        self._sampling = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._sampling = False
        self._thread.join(timeout=1.0)
        proc = psutil.Process(os.getpid())
        self.wall_time_s = time.perf_counter() - self._t_start
        self.ram_mb_end = proc.memory_info().rss / 1024 / 1024
        self.ram_mb_peak = max(self._samples_ram) if self._samples_ram else self.ram_mb_end
        self.cpu_percent_peak = max(self._samples_cpu) if self._samples_cpu else 0.0
        try:
            self.io_read_bytes = proc.io_counters().read_bytes - self._io_start
        except (AttributeError, psutil.AccessDenied, NotImplementedError):
            self.io_read_bytes = 0

    def to_dict(self) -> dict:
        return {
            "server": self.server,
            "tool": self.tool,
            "scenario_id": self.scenario_id,
            "orchestration": self.orchestration,
            "run_id": self.run_id,
            "wall_time_s": round(self.wall_time_s, 4),
            "cpu_percent_peak": round(self.cpu_percent_peak, 2),
            "ram_mb_start": round(self.ram_mb_start, 2),
            "ram_mb_peak": round(self.ram_mb_peak, 2),
            "ram_mb_end": round(self.ram_mb_end, 2),
            "io_read_bytes": self.io_read_bytes,
        }

    def log_to_wandb(self) -> None:
        """Log this call's metrics to the active wandb run if one is initialized."""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(self.to_dict())
        except ImportError:
            pass
