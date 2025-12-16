"""
Metrics Collection and Resource Monitoring for Experiment 2
Tracks CPU, memory, I/O, and custom metrics during execution
"""

import psutil
import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class ResourceSnapshot:
    """Single snapshot of resource utilization"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    page_faults: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """
    Monitor system resources during execution
    Runs in background thread and collects snapshots
    """

    def __init__(
        self,
        interval: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize resource monitor

        Args:
            interval: Sampling interval in seconds
            logger: Logger instance
        """
        self.interval = interval
        self.logger = logger or logging.getLogger(__name__)
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()

        # Initialize disk counters
        try:
            self.disk_io_start = psutil.disk_io_counters()
        except Exception as e:
            self.logger.warning(f"Could not initialize disk I/O counters: {e}")
            self.disk_io_start = None

    def start(self):
        """Start monitoring in background thread"""
        if self.monitoring:
            self.logger.warning("Monitor already running")
            return

        self.logger.info(f"Starting resource monitor (interval={self.interval}s)")
        self.monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring"""
        if not self.monitoring:
            return

        self.logger.info("Stopping resource monitor")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        self.logger.info(f"Collected {len(self.snapshots)} resource snapshots")

    def _monitor_loop(self):
        """Main monitoring loop (runs in thread)"""
        prev_disk_io = self.disk_io_start
        prev_time = time.time()

        while self.monitoring:
            try:
                current_time = time.time()
                elapsed = current_time - prev_time

                # CPU utilization
                cpu_percent = self.process.cpu_percent(interval=None)

                # Memory utilization
                mem_info = self.process.memory_info()
                memory_mb = mem_info.rss / (1024 ** 2)
                memory_percent = self.process.memory_percent()

                # Disk I/O
                disk_read_mb = 0.0
                disk_write_mb = 0.0
                disk_read_mb_s = 0.0
                disk_write_mb_s = 0.0

                try:
                    curr_disk_io = psutil.disk_io_counters()
                    if curr_disk_io and prev_disk_io:
                        disk_read_mb = curr_disk_io.read_bytes / (1024 ** 2)
                        disk_write_mb = curr_disk_io.write_bytes / (1024 ** 2)

                        if elapsed > 0:
                            disk_read_mb_s = (curr_disk_io.read_bytes - prev_disk_io.read_bytes) / (1024 ** 2) / elapsed
                            disk_write_mb_s = (curr_disk_io.write_bytes - prev_disk_io.write_bytes) / (1024 ** 2) / elapsed

                        prev_disk_io = curr_disk_io
                except Exception:
                    pass

                # Page faults (Windows: num_page_faults, Linux: pfaults)
                page_faults = 0
                try:
                    if hasattr(mem_info, 'num_page_faults'):
                        page_faults = mem_info.num_page_faults
                    elif hasattr(mem_info, 'pfaults'):
                        page_faults = mem_info.pfaults
                except Exception:
                    pass

                snapshot = ResourceSnapshot(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_read_mb=disk_read_mb,
                    disk_write_mb=disk_write_mb,
                    disk_read_mb_s=disk_read_mb_s,
                    disk_write_mb_s=disk_write_mb_s,
                    page_faults=page_faults
                )

                self.snapshots.append(snapshot)
                prev_time = current_time

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.interval)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from collected snapshots"""
        if not self.snapshots:
            return {}

        cpu_vals = [s.cpu_percent for s in self.snapshots]
        mem_vals = [s.memory_mb for s in self.snapshots]
        read_vals = [s.disk_read_mb_s for s in self.snapshots]
        write_vals = [s.disk_write_mb_s for s in self.snapshots]

        summary = {
            'cpu_mean': sum(cpu_vals) / len(cpu_vals),
            'cpu_max': max(cpu_vals),
            'cpu_min': min(cpu_vals),
            'memory_mean_mb': sum(mem_vals) / len(mem_vals),
            'memory_peak_mb': max(mem_vals),
            'memory_min_mb': min(mem_vals),
            'disk_read_mean_mb_s': sum(read_vals) / len(read_vals),
            'disk_read_max_mb_s': max(read_vals),
            'disk_write_mean_mb_s': sum(write_vals) / len(write_vals),
            'disk_write_max_mb_s': max(write_vals),
            'num_snapshots': len(self.snapshots)
        }

        if self.snapshots:
            summary['page_faults_total'] = self.snapshots[-1].page_faults

        return summary

    def save_snapshots(self, output_path: Path):
        """Save all snapshots to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'snapshots': [s.to_dict() for s in self.snapshots],
            'summary': self.get_summary(),
            'interval': self.interval
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved {len(self.snapshots)} snapshots to {output_path}")


class MetricsCollector:
    """
    Collect custom metrics during experiment execution
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize metrics collector"""
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def add_metric(self, name: str, value: Any):
        """Add a single metric"""
        self.metrics[name] = value
        self.logger.debug(f"Metric recorded: {name} = {value}")

    def add_metrics(self, metrics_dict: Dict[str, Any]):
        """Add multiple metrics at once"""
        self.metrics.update(metrics_dict)
        for name, value in metrics_dict.items():
            self.logger.debug(f"Metric recorded: {name} = {value}")

    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
        self.logger.debug(f"Timer started: {name}")

    def stop_timer(self, name: str) -> float:
        """Stop timing an operation and record duration"""
        if name not in self.start_times:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0

        duration = time.time() - self.start_times[name]

        if name not in self.timings:
            self.timings[name] = []

        self.timings[name].append(duration)
        del self.start_times[name]

        self.logger.debug(f"Timer stopped: {name} = {duration:.4f}s")
        return duration

    def get_timing_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a timer"""
        if name not in self.timings or not self.timings[name]:
            return {}

        values = self.timings[name]
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'total': sum(values),
            'count': len(values)
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        result = dict(self.metrics)

        # Add timing summaries
        for name in self.timings:
            summary = self.get_timing_summary(name)
            for key, value in summary.items():
                result[f"{name}_{key}"] = value

        return result

    def save_metrics(self, output_path: Path):
        """Save all metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metrics': self._convert_to_json_serializable(self.metrics),
            'timings': self._convert_to_json_serializable(self.timings),
            'summary': self._convert_to_json_serializable(self.get_all_metrics()),
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved metrics to {output_path}")

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


if __name__ == '__main__':
    # Test resource monitor
    print("Testing ResourceMonitor...")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    monitor = ResourceMonitor(interval=0.5, logger=logger)
    monitor.start()

    # Simulate some work
    time.sleep(3)
    data = [i ** 2 for i in range(1000000)]

    monitor.stop()

    summary = monitor.get_summary()
    print("\nResource Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    # Test metrics collector
    print("\nTesting MetricsCollector...")

    collector = MetricsCollector(logger=logger)
    collector.add_metric('test_value', 42)
    collector.start_timer('operation')
    time.sleep(1)
    collector.stop_timer('operation')

    print("\nMetrics Summary:")
    for key, value in collector.get_all_metrics().items():
        print(f"  {key}: {value}")

    print("\nMetrics test completed successfully!")
