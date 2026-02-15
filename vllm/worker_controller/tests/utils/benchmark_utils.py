from __future__ import annotations

from datetime import datetime
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests


def print_banner(title: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def to_optional_float(value: Any) -> float | None:
    return float(value) if is_finite_number(value) else None


def average_finite(values, default: float | None = None) -> float | None:
    numeric = [float(v) for v in values if is_finite_number(v)]
    if not numeric:
        return default
    return sum(numeric) / len(numeric)


def format_seconds(value: Any, precision: int = 2) -> str:
    as_float = to_optional_float(value)
    if as_float is None:
        return "N/A"
    return f"{as_float:.{precision}f}s"


def results_for_model(items: list[dict], model_name: str) -> list[dict]:
    return [result for result in items if result.get("model") == model_name]


def avg_metric(items: list[dict], key: str, default: float = 0.0) -> float:
    avg = average_finite([item.get(key) for item in items], default=default)
    return float(avg) if isinstance(avg, (int, float)) else default


def avg_nested_metric(items: list[dict], parent_key: str, key: str) -> float:
    values = []
    for item in items:
        parent = item.get(parent_key) or {}
        values.append(parent.get(key))
    avg = average_finite(values, default=0.0)
    return float(avg) if isinstance(avg, (int, float)) else 0.0


def avg_optional_metric(items: list[dict], key: str) -> float | None:
    return average_finite([item.get(key) for item in items], default=None)


def _extract_log_timestamp_seconds(line: str) -> float | None:
    match = re.search(r"\b\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})\b", line)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))
    return float(hour * 3600 + minute * 60 + second)


def parse_standard_startup_milestones(log_lines: list[str]) -> dict[str, float]:
    """Parse startup milestones from standard vLLM logs.

    Accepts either plain log lines or (received_ts, line) tuples.
    """
    event_patterns = {
        "api_process_started_log": "vLLM API server version",
        "engine_process_init_log": "Initializing a V1 LLM engine",
        "distributed_init_log": "world_size=",
        "model_load_start_log": "Starting to load model",
        "weights_loaded_log": "Loading weights took",
        "model_loaded_log": "Model loading took",
        "kv_cache_init_done_log": "init engine (profile, create kv cache, warmup model) took",
        "api_routes_ready_log": "Starting vLLM API server",
        "uvicorn_started_log": "Started server process",
        "app_startup_complete_log": "Application startup complete",
        "health_ok_log": "GET /health",
    }

    event_clock: dict[str, float] = {}
    event_received: dict[str, float] = {}
    first_received_ts: float | None = None
    for entry in log_lines:
        if isinstance(entry, tuple) and len(entry) == 2:
            received_ts = float(entry[0])
            line = str(entry[1])
            if first_received_ts is None:
                first_received_ts = received_ts
        else:
            received_ts = None
            line = str(entry)

        for event_name, pattern in event_patterns.items():
            if event_name in event_clock:
                continue
            if pattern in line:
                ts = _extract_log_timestamp_seconds(line)
                if ts is not None:
                    event_clock[event_name] = ts
                if received_ts is not None:
                    event_received[event_name] = received_ts

    milestones: dict[str, float] = {}
    ordered = [
        "api_process_started_log",
        "engine_process_init_log",
        "distributed_init_log",
        "model_load_start_log",
        "weights_loaded_log",
        "model_loaded_log",
        "kv_cache_init_done_log",
        "api_routes_ready_log",
        "health_ok_log",
    ]

    for prev_name, next_name in zip(ordered, ordered[1:]):
        if prev_name in event_clock and next_name in event_clock:
            delta = event_clock[next_name] - event_clock[prev_name]
            if delta >= 0:
                milestones[f"{prev_name}__to__{next_name}_s"] = delta

    if first_received_ts is not None:
        milestones["first_log_received_ts"] = first_received_ts
    for event_name, received_ts in event_received.items():
        milestones[f"{event_name}_received_ts"] = received_ts

    return milestones


def parse_standard_startup_breakdown(log_lines: list[str]) -> dict:
    """Parse startup breakdown timings from standard vLLM logs."""
    parsed = {
        "std_worker_setup_time": 0.0,
        "std_mq_setup_time": 0.0,
        "std_model_load_time": 0.0,
        "std_engine_init_time": 0.0,
        "std_weight_load_time": 0.0,
        "model_weight_load_s": 0.0,
        "model_load_total_s": 0.0,
        "memory_profile_s": 0.0,
        "engine_init_profile_kv_cache_warmup_s": 0.0,
    }

    startup_re = re.compile(
        r"MultiprocExecutor startup breakdown: worker_ready=([0-9.]+)s, mq_ready=([0-9.]+)s, total=([0-9.]+)s"
    )
    weight_re = re.compile(r"Loading weights took\s+([0-9.]+)\s+seconds")
    model_load_re = re.compile(
        r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds"
    )
    profile_re = re.compile(r"profiling memory takes:\s*([0-9.]+)s")
    init_re = re.compile(
        r"init engine \(profile, create kv cache, warmup model\) took\s+([0-9.]+)\s+seconds"
    )

    for entry in log_lines:
        if isinstance(entry, tuple) and len(entry) == 2:
            line = str(entry[1])
        else:
            line = str(entry)

        m = startup_re.search(line)
        if m:
            parsed["std_worker_setup_time"] = float(m.group(1))
            parsed["std_mq_setup_time"] = float(m.group(2))
            continue

        m = weight_re.search(line)
        if m:
            parsed["std_weight_load_time"] = float(m.group(1))
            parsed["model_weight_load_s"] = float(m.group(1))
            continue

        m = model_load_re.search(line)
        if m:
            parsed["model_load_total_s"] = float(m.group(1))
            continue

        m = profile_re.search(line)
        if m:
            parsed["memory_profile_s"] = float(m.group(1))
            continue

        m = init_re.search(line)
        if m:
            parsed["std_engine_init_time"] = float(m.group(1))
            parsed["engine_init_profile_kv_cache_warmup_s"] = float(m.group(1))
            continue

    if parsed["model_load_total_s"] > 0:
        parsed["std_model_load_time"] = parsed["model_load_total_s"]
    else:
        parsed["std_model_load_time"] = max(
            0.0,
            parsed["std_engine_init_time"] - parsed["std_weight_load_time"],
        )

    return parsed


def _extract_event_receive_times(
    log_lines: list,
    event_patterns: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    event_times: dict[str, float] = {}
    for entry in log_lines:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            continue
        received_ts = float(entry[0])
        line = str(entry[1])
        for event_name, patterns in event_patterns.items():
            if event_name in event_times:
                continue
            if any(pattern in line for pattern in patterns):
                event_times[event_name] = received_ts
    return event_times


def _delta_or_none(
    event_times: dict[str, float],
    start_event: str,
    end_event: str,
) -> float | None:
    start_ts = event_times.get(start_event)
    end_ts = event_times.get(end_event)
    if not isinstance(start_ts, (int, float)) or not isinstance(end_ts, (int, float)):
        return None
    return max(0.0, float(end_ts) - float(start_ts))


def compute_standard_flow_timing(
    log_lines: list,
    spawn_call_time_s: float | None,
    server_ready_time_s: float | None,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Compute a non-overlapping stage timing view for standard vLLM startup."""
    event_patterns = {
        "first_log": ("",),
        "api_process_started": ("vLLM API server version",),
        "engine_init": ("Initializing a V1 LLM engine",),
        "distributed_init": ("world_size=",),
        "model_load_start": ("Starting to load model",),
        "model_loaded": ("Model loading took",),
        "kv_ready": (
            "init engine (profile, create kv cache, warmup model) took",
        ),
        "api_routes_ready": ("Starting vLLM API server",),
        "health_ok": ("GET /health",),
    }

    event_times = _extract_event_receive_times(log_lines, event_patterns)

    first_log_ts = None
    if log_lines and isinstance(log_lines[0], tuple) and len(log_lines[0]) == 2:
        first_log_ts = float(log_lines[0][0])
        event_times.setdefault("first_log", first_log_ts)

    stages: dict[str, float | None] = {
        "spawn_call_time_s": float(spawn_call_time_s)
        if isinstance(spawn_call_time_s, (int, float))
        else None,
        "spawn_to_first_log_s": _delta_or_none(event_times, "first_log", "api_process_started"),
        "api_bootstrap_to_engine_init_s": _delta_or_none(
            event_times,
            "api_process_started",
            "engine_init",
        ),
        "engine_init_to_distributed_init_s": _delta_or_none(
            event_times,
            "engine_init",
            "distributed_init",
        ),
        "distributed_init_to_model_load_start_s": _delta_or_none(
            event_times,
            "distributed_init",
            "model_load_start",
        ),
        "model_load_phase_s": _delta_or_none(
            event_times,
            "model_load_start",
            "model_loaded",
        ),
        "post_model_to_kv_ready_s": _delta_or_none(
            event_times,
            "model_loaded",
            "kv_ready",
        ),
        "post_kv_to_api_routes_ready_s": _delta_or_none(
            event_times,
            "kv_ready",
            "api_routes_ready",
        ),
        "api_routes_to_health_ready_s": _delta_or_none(
            event_times,
            "api_routes_ready",
            "health_ok",
        ),
        "spawn_to_health_ready_probe_s": float(server_ready_time_s)
        if isinstance(server_ready_time_s, (int, float))
        else None,
    }

    event_offsets: dict[str, float | None] = {}
    for event_name, ts in event_times.items():
        if isinstance(first_log_ts, (int, float)):
            event_offsets[f"{event_name}_after_first_log_s"] = max(
                0.0,
                float(ts) - float(first_log_ts),
            )
        else:
            event_offsets[f"{event_name}_after_first_log_s"] = None

    return stages, event_offsets


def compute_worker_controller_flow_timing(
    create_time_s: float | None,
    api_ready_time_s: float | None,
    first_inference_time_s: float | None,
    create_timings: dict | None,
    model_load_summary: dict | None,
    startup_timing: dict | None,
) -> dict[str, float | None]:
    """Compute a non-overlapping stage timing view for Worker Controller startup."""
    attach_to_workers_s = None
    api_process_spawn_s = None
    if isinstance(create_timings, dict):
        attach_parts = [
            create_timings.get("resource_allocation_time"),
            create_timings.get("ipc_queue_setup_time"),
            create_timings.get("proxy_register_time"),
        ]
        attach_numeric = [float(v) for v in attach_parts if isinstance(v, (int, float))]
        if attach_numeric:
            attach_to_workers_s = sum(attach_numeric)
        spawn_value = create_timings.get("api_process_spawn_time")
        if isinstance(spawn_value, (int, float)):
            api_process_spawn_s = float(spawn_value)

    engine_init_profile_kv_warmup_s = None
    remote_load_model_rpc_s = None
    if isinstance(model_load_summary, dict):
        init_v = model_load_summary.get("init_engine_time_seconds")
        rpc_v = model_load_summary.get("remote_executor_load_model_rpc_time")
        if isinstance(init_v, (int, float)):
            engine_init_profile_kv_warmup_s = float(init_v)
        if isinstance(rpc_v, (int, float)):
            remote_load_model_rpc_s = float(rpc_v)

    api_routes_to_health_ready_s = None
    if isinstance(startup_timing, dict):
        routes_v = startup_timing.get("api_routes_to_first_health_s")
        if isinstance(routes_v, (int, float)):
            api_routes_to_health_ready_s = float(routes_v)

    startup_ready = None
    if isinstance(create_time_s, (int, float)) and isinstance(api_ready_time_s, (int, float)):
        startup_ready = float(create_time_s) + float(api_ready_time_s)

    return {
        "controller_create_http_roundtrip_s": float(create_time_s)
        if isinstance(create_time_s, (int, float))
        else None,
        "controller_attach_to_workers_s": attach_to_workers_s,
        "controller_spawn_api_process_s": api_process_spawn_s,
        "engine_health_wait_after_create_s": float(api_ready_time_s)
        if isinstance(api_ready_time_s, (int, float))
        else None,
        "engine_api_routes_to_health_ready_s": api_routes_to_health_ready_s,
        "engine_init_profile_kv_cache_warmup_s": engine_init_profile_kv_warmup_s,
        "engine_remote_load_model_rpc_s": remote_load_model_rpc_s,
        "spawn_to_health_ready_probe_s": startup_ready,
        "first_inference_s": float(first_inference_time_s)
        if isinstance(first_inference_time_s, (int, float))
        else None,
    }


def print_flow_stage_report(title: str, stages: dict[str, float | None]) -> None:
    print(f"{title} flow stage timing:")
    for key, value in stages.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {float(value):.3f}")
        else:
            print(f"  {key}: N/A")


def print_flow_event_offsets(title: str, event_offsets: dict[str, float | None]) -> None:
    print(f"{title} event offsets:")
    for key, value in event_offsets.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {float(value):.3f}")
        else:
            print(f"  {key}: N/A")


class _FilteredOutput:
    """Drop noisy low-value lines from stdout/stderr."""

    def __init__(self, wrapped, drop_patterns: tuple[str, ...]):
        self._wrapped = wrapped
        self._drop_patterns = drop_patterns
        self._buffer = ""

    def write(self, data: str):
        if not isinstance(data, str):
            data = str(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if any(pattern in line for pattern in self._drop_patterns):
                continue
            self._wrapped.write(line + "\n")
        return len(data)

    def flush(self):
        if self._buffer:
            if not any(pattern in self._buffer for pattern in self._drop_patterns):
                self._wrapped.write(self._buffer)
            self._buffer = ""
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def install_output_filter(noisy_output_patterns: tuple[str, ...]) -> None:
    if not isinstance(sys.stdout, _FilteredOutput):
        sys.stdout = _FilteredOutput(sys.stdout, noisy_output_patterns)
    if not isinstance(sys.stderr, _FilteredOutput):
        sys.stderr = _FilteredOutput(sys.stderr, noisy_output_patterns)


def wc_log(stage: str, message: str) -> None:
    print(f"[WC][{stage}] {message}")


def prewarm_model_files(
    model_name: str,
    prewarm_download_if_missing: bool,
    model_shard_extensions: tuple[str, ...],
) -> dict:
    """Read model shard files once to warm Linux page cache."""
    prewarm_start = time.time()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return {
            "model": model_name,
            "ok": False,
            "error": f"huggingface_hub import failed: {exc}",
            "seconds": 0.0,
            "files": 0,
            "bytes": 0,
        }

    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=not prewarm_download_if_missing,
            allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json", "*.model"],
        )
    except Exception as exc:
        return {
            "model": model_name,
            "ok": False,
            "error": f"snapshot_download failed: {exc}",
            "seconds": time.time() - prewarm_start,
            "files": 0,
            "bytes": 0,
        }

    shard_files = []
    for path in Path(snapshot_path).rglob("*"):
        if path.is_file() and path.suffix.lower() in model_shard_extensions:
            shard_files.append(path)

    total_bytes = 0
    chunk_size = 8 * 1024 * 1024
    for shard in shard_files:
        with shard.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)

    return {
        "model": model_name,
        "ok": True,
        "snapshot_path": snapshot_path,
        "seconds": time.time() - prewarm_start,
        "files": len(shard_files),
        "bytes": total_bytes,
    }


def format_create_timings(create_timings: dict | None) -> str:
    if not create_timings:
        return "n/a"
    ordered_keys = [
        "resource_allocation_time",
        "ipc_queue_setup_time",
        "proxy_register_time",
        "api_process_spawn_time",
        "create_call_total_time",
    ]
    parts = []
    for key in ordered_keys:
        value = create_timings.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key.replace('_time', '')}={value:.3f}s")
    return ", ".join(parts) if parts else "n/a"


def start_subprocess_log_collector(
    proc,
    log_lines: list,
    prefix: str = "",
    capture_receive_ts: bool = False,
):
    """Collect subprocess logs asynchronously into log_lines."""

    def _reader():
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            if capture_receive_ts:
                log_lines.append((time.time(), line))
            else:
                log_lines.append(line)
            if len(log_lines) > 5000:
                del log_lines[:1000]
            if (
                "MultiprocExecutor startup breakdown" in line
                or "Loading weights took" in line
                or "init engine (profile, create kv cache, warmup model) took" in line
            ):
                print(f"{prefix}{line}")

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def print_standard_startup_overhead(
    server_ready_time: float | None,
    std_startup_breakdown: dict,
    std_milestones: dict[str, float],
) -> None:
    """Print startup decomposition aligned with api_layer timing output."""

    def _print_metric(name: str, value: float | None):
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.3f}")
        else:
            print(f"  {name}: N/A")

    def _milestone_after_spawn(key: str) -> float | None:
        first = std_milestones.get("first_log_received_ts")
        received = std_milestones.get(f"{key}_received_ts")
        if isinstance(first, (int, float)) and isinstance(received, (int, float)):
            return max(0.0, float(received) - float(first))
        return None

    relative_milestones: dict[str, float] = {}
    for key in [
        "api_process_started_log",
        "engine_process_init_log",
        "distributed_init_log",
        "model_load_start_log",
        "weights_loaded_log",
        "model_loaded_log",
        "kv_cache_init_done_log",
        "api_routes_ready_log",
        "uvicorn_started_log",
        "app_startup_complete_log",
        "health_ok_log",
    ]:
        rel = _milestone_after_spawn(key)
        if isinstance(rel, (int, float)):
            relative_milestones[f"{key}_after_spawn_s"] = rel

    first_rel = 0.0 if "first_log_received_ts" in std_milestones else None
    if isinstance(first_rel, (int, float)):
        relative_milestones["first_log_after_spawn_s"] = first_rel
    if isinstance(server_ready_time, (int, float)):
        relative_milestones["ready_after_spawn_s"] = float(server_ready_time)

    model_weight_load = std_startup_breakdown.get("model_weight_load_s")
    model_load_total = std_startup_breakdown.get("model_load_total_s")
    memory_profile = std_startup_breakdown.get("memory_profile_s")
    engine_init_total = std_startup_breakdown.get("engine_init_profile_kv_cache_warmup_s")

    if not isinstance(model_weight_load, (int, float)) or model_weight_load <= 0:
        model_weight_load = std_startup_breakdown.get("std_weight_load_time")
    if not isinstance(model_load_total, (int, float)) or model_load_total <= 0:
        model_load_total = std_startup_breakdown.get("std_model_load_time")
    if not isinstance(engine_init_total, (int, float)) or engine_init_total <= 0:
        engine_init_total = std_startup_breakdown.get("std_engine_init_time")

    kv_cache_plus_warmup_est = None
    if isinstance(engine_init_total, (int, float)) and isinstance(memory_profile, (int, float)):
        kv_cache_plus_warmup_est = max(0.0, float(engine_init_total) - float(memory_profile))

    remaining_startup_overhead = None
    if isinstance(server_ready_time, (int, float)):
        known_sum = 0.0
        if isinstance(model_load_total, (int, float)):
            known_sum += float(model_load_total)
        if isinstance(engine_init_total, (int, float)):
            known_sum += float(engine_init_total)
        if known_sum > 0:
            remaining_startup_overhead = max(0.0, float(server_ready_time) - known_sum)

    api_process_bootstrap_s = relative_milestones.get("api_process_started_log_after_spawn_s")
    engine_init_seen_s = relative_milestones.get("engine_process_init_log_after_spawn_s")
    distributed_init_seen_s = relative_milestones.get("distributed_init_log_after_spawn_s")
    model_load_start_seen_s = relative_milestones.get("model_load_start_log_after_spawn_s")
    model_load_done_seen_s = relative_milestones.get("model_loaded_log_after_spawn_s")
    kv_ready_seen_s = relative_milestones.get("kv_cache_init_done_log_after_spawn_s")
    api_routes_ready_seen_s = relative_milestones.get("api_routes_ready_log_after_spawn_s")
    health_ready_seen_s = relative_milestones.get("health_ok_log_after_spawn_s")

    api_bootstrap_until_engine_init_visible_s = None
    if isinstance(api_process_bootstrap_s, (int, float)) and isinstance(engine_init_seen_s, (int, float)):
        api_bootstrap_until_engine_init_visible_s = max(0.0, engine_init_seen_s - api_process_bootstrap_s)

    worker_process_startup_est_s = None
    if isinstance(engine_init_seen_s, (int, float)) and isinstance(distributed_init_seen_s, (int, float)):
        worker_process_startup_est_s = max(0.0, distributed_init_seen_s - engine_init_seen_s)

    engine_pre_model_overhead_s = None
    if isinstance(distributed_init_seen_s, (int, float)) and isinstance(model_load_start_seen_s, (int, float)):
        engine_pre_model_overhead_s = max(0.0, model_load_start_seen_s - distributed_init_seen_s)

    engine_creation_phase_s = None
    if isinstance(engine_init_seen_s, (int, float)) and isinstance(kv_ready_seen_s, (int, float)):
        engine_creation_phase_s = max(0.0, kv_ready_seen_s - engine_init_seen_s)

    model_loading_phase_s = None
    if isinstance(model_load_done_seen_s, (int, float)) and isinstance(model_load_start_seen_s, (int, float)):
        model_loading_phase_s = max(0.0, model_load_done_seen_s - model_load_start_seen_s)

    kv_cache_phase_after_model_load_s = std_milestones.get(
        "model_loaded_log__to__kv_cache_init_done_log_s"
    )

    post_engine_to_routes_s = None
    if isinstance(kv_ready_seen_s, (int, float)) and isinstance(api_routes_ready_seen_s, (int, float)):
        post_engine_to_routes_s = max(0.0, api_routes_ready_seen_s - kv_ready_seen_s)

    routes_to_health_s = None
    if isinstance(api_routes_ready_seen_s, (int, float)) and isinstance(health_ready_seen_s, (int, float)):
        routes_to_health_s = max(0.0, health_ready_seen_s - api_routes_ready_seen_s)

    post_engine_api_readiness_s = None
    if isinstance(kv_ready_seen_s, (int, float)) and isinstance(health_ready_seen_s, (int, float)):
        post_engine_api_readiness_s = max(0.0, health_ready_seen_s - kv_ready_seen_s)

    print("Startup timing breakdown:")
    _print_metric("total_api_ready_s", server_ready_time)
    _print_metric("model_weight_load_s", model_weight_load)
    _print_metric("model_load_total_s", model_load_total)
    _print_metric("memory_profile_s", memory_profile)
    _print_metric("engine_init_profile_kv_cache_warmup_s", engine_init_total)
    _print_metric("kv_cache_plus_warmup_estimated_s", kv_cache_plus_warmup_est)
    _print_metric("remaining_startup_overhead_s", remaining_startup_overhead)

    print("Startup overhead decomposition:")
    _print_metric("api_process_bootstrap_to_first_api_log_s", api_process_bootstrap_s)
    _print_metric("api_bootstrap_until_engine_init_visible_s", api_bootstrap_until_engine_init_visible_s)
    _print_metric("  worker_process_startup_estimated_s (included above)", worker_process_startup_est_s)
    _print_metric("  engine_pre_model_overhead_estimated_s (included above)", engine_pre_model_overhead_s)
    _print_metric("engine_creation_phase_s", engine_creation_phase_s)
    _print_metric("  model_loading_phase_s (included above)", model_loading_phase_s)
    _print_metric("  kv_cache_phase_after_model_load_s (included above)", kv_cache_phase_after_model_load_s)
    _print_metric("post_engine_api_readiness_s", post_engine_api_readiness_s)
    _print_metric("  post_engine_to_api_routes_ready_s (included above)", post_engine_to_routes_s)
    _print_metric("  api_routes_to_health_ready_s (included above)", routes_to_health_s)

    print("Startup milestones (relative timings):")
    priority_order = [
        "first_log_after_spawn_s",
        "api_process_started_log_after_spawn_s",
        "engine_process_init_log_after_spawn_s",
        "distributed_init_log_after_spawn_s",
        "model_load_start_log_after_spawn_s",
        "weights_loaded_log_after_spawn_s",
        "model_loaded_log_after_spawn_s",
        "kv_cache_init_done_log_after_spawn_s",
        "api_routes_ready_log_after_spawn_s",
        "uvicorn_started_log_after_spawn_s",
        "app_startup_complete_log_after_spawn_s",
        "health_ok_log_after_spawn_s",
        "ready_after_spawn_s",
    ]
    for key in priority_order:
        if key in relative_milestones:
            _print_metric(key, relative_milestones[key])

    print("Startup milestones (log timestamp segments):")
    segment_keys = [
        "api_process_started_log__to__engine_process_init_log_s",
        "engine_process_init_log__to__distributed_init_log_s",
        "distributed_init_log__to__model_load_start_log_s",
        "model_load_start_log__to__weights_loaded_log_s",
        "weights_loaded_log__to__model_loaded_log_s",
        "model_loaded_log__to__kv_cache_init_done_log_s",
        "kv_cache_init_done_log__to__api_routes_ready_log_s",
    ]
    for key in segment_keys:
        if key in std_milestones:
            _print_metric(key, std_milestones[key])


def print_worker_controller_startup_overhead(
    create_time: float | None,
    api_ready_time: float | None,
    create_timings: dict | None,
    model_load_summary: dict | None,
    model_load_timings: list[dict] | None,
    startup_timing: dict | None = None,
    routes_to_health: float | None = None,
) -> None:
    """Print hierarchical startup decomposition for Worker Controller."""

    def _print_metric(name: str, value: float | None):
        if isinstance(value, (int, float)):
            print(f"    {name}: {value:.3f}s")
        else:
            print(f"    {name}: N/A")

    total_api_ready = None
    if isinstance(create_time, (int, float)) and isinstance(api_ready_time, (int, float)):
        total_api_ready = create_time + api_ready_time

    init_engine_time = None
    remote_executor_load_rpc = None
    if isinstance(model_load_summary, dict):
        maybe_init = model_load_summary.get("init_engine_time_seconds")
        if isinstance(maybe_init, (int, float)):
            init_engine_time = float(maybe_init)

        maybe_rpc = model_load_summary.get("remote_executor_load_model_rpc_time")
        if isinstance(maybe_rpc, (int, float)):
            remote_executor_load_rpc = float(maybe_rpc)

    worker_effective_load_avg = None
    if isinstance(model_load_timings, list) and model_load_timings:
        worker_effective_loads: list[float] = []
        for wt in model_load_timings:
            if not isinstance(wt, dict):
                continue
            effective = wt.get(
                "effective_model_load_time",
                (wt.get("model_runner_init_time", 0) or 0)
                + (wt.get("weight_load_time", 0) or 0),
            )
            if isinstance(effective, (int, float)):
                worker_effective_loads.append(float(effective))
        if worker_effective_loads:
            worker_effective_load_avg = sum(worker_effective_loads) / len(worker_effective_loads)

    kv_cache_phase_after_model_load = None
    if isinstance(init_engine_time, (int, float)) and isinstance(worker_effective_load_avg, (int, float)):
        kv_cache_phase_after_model_load = max(0.0, init_engine_time - worker_effective_load_avg)

    remaining_startup_overhead = None
    if isinstance(total_api_ready, (int, float)) and isinstance(init_engine_time, (int, float)):
        remaining_startup_overhead = max(0.0, total_api_ready - init_engine_time)

    total_ready_accounted = None
    if isinstance(create_time, (int, float)) and isinstance(api_ready_time, (int, float)):
        total_ready_accounted = create_time + api_ready_time

    total_ready_residual = None
    if isinstance(total_api_ready, (int, float)) and isinstance(total_ready_accounted, (int, float)):
        total_ready_residual = max(0.0, total_api_ready - total_ready_accounted)

    resource_allocation_time = None
    ipc_queue_setup_time = None
    proxy_register_time = None
    api_process_spawn_time = None
    if isinstance(create_timings, dict):
        for key_name, target in [
            ("resource_allocation_time", "resource_allocation_time"),
            ("ipc_queue_setup_time", "ipc_queue_setup_time"),
            ("proxy_register_time", "proxy_register_time"),
            ("api_process_spawn_time", "api_process_spawn_time"),
        ]:
            value = create_timings.get(key_name)
            if isinstance(value, (int, float)):
                if target == "resource_allocation_time":
                    resource_allocation_time = float(value)
                elif target == "ipc_queue_setup_time":
                    ipc_queue_setup_time = float(value)
                elif target == "proxy_register_time":
                    proxy_register_time = float(value)
                elif target == "api_process_spawn_time":
                    api_process_spawn_time = float(value)

    print("  Worker Controller startup timing breakdown:")
    _print_metric("total_api_ready_s (create + health)", total_api_ready)
    _print_metric("create_time", create_time)
    _print_metric("api_ready_time", api_ready_time)
    _print_metric("init_engine_time", init_engine_time)
    _print_metric("remaining_startup_overhead_s (api_ready - engine_internal_subset)", remaining_startup_overhead)

    post_engine_to_api_routes_ready = None
    if isinstance(api_ready_time, (int, float)) and isinstance(routes_to_health, (int, float)):
        post_engine_to_api_routes_ready = max(0.0, float(api_ready_time) - float(routes_to_health))

    print("  Worker Controller startup overhead decomposition:")
    _print_metric("engine_create_phase_s", create_time)
    _print_metric("  resource_allocation_time (included above)", resource_allocation_time)
    _print_metric("  ipc_queue_setup_time (included above)", ipc_queue_setup_time)
    _print_metric("  proxy_register_time (included above)", proxy_register_time)
    _print_metric("  api_process_spawn_time (included above)", api_process_spawn_time)
    _print_metric("engine_creation_internal_s", init_engine_time)
    _print_metric("  model_loading_phase_s (included above)", worker_effective_load_avg)
    _print_metric("  kv_cache_phase_after_model_load_s (included above)", kv_cache_phase_after_model_load)
    _print_metric("  remote_executor_load_model_rpc_time (included above)", remote_executor_load_rpc)
    _print_metric("post_engine_api_readiness_s", api_ready_time)
    _print_metric("  post_engine_to_api_routes_ready_s (included above)", post_engine_to_api_routes_ready)
    _print_metric("  api_routes_to_health_ready_s (included above)", routes_to_health)

    print("  Worker Controller startup phase accounting (non-overlapping):")
    _print_metric("phase_1_create_call_s", create_time)
    _print_metric("phase_2_wait_for_engine_health_s", api_ready_time)
    _print_metric("phase_accounted_total_api_ready_s", total_ready_accounted)
    _print_metric("phase_residual_unobserved_s", total_ready_residual)

    if (
        isinstance(worker_effective_load_avg, (int, float))
        and isinstance(init_engine_time, (int, float))
        and worker_effective_load_avg > init_engine_time + 0.2
    ):
        print(
            "    note: model_loading_phase_s and init_engine_time come from different "
            "instrumentation scopes and are not additive"
        )


def configure_debug_logging(noisy_output_patterns: tuple[str, ...]) -> None:
    install_output_filter(noisy_output_patterns)

    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_WC_QUIET_WORKER_STDIO"] = "0"
    os.environ["VLLM_WC_QUIET_API_STDIO"] = "0"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
    os.environ["GLOO_LOG_LEVEL"] = "ERROR"

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker_controller").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker_controller.worker_controller").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker_controller.worker_controller_server").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker_controller.executor.proxy_executor").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker_controller.worker.gpu_worker").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker_controller.entrypoint.api_server").setLevel(logging.WARNING)
    logging.getLogger("vllm.entrypoints.openai.api_server").setLevel(logging.WARNING)
    logging.getLogger("vllm.v1.engine.core").setLevel(logging.WARNING)
    logging.getLogger("vllm.model_executor").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def print_engine_status(
    base_url: str,
    engine_uuid: str,
    prefix: str = "",
    include_create_timings: bool = False,
) -> None:
    """Fetch and print current engine status from worker controller."""
    try:
        status_resp = requests.get(f"{base_url}/engines/{engine_uuid}", timeout=5)
        if status_resp.status_code == 200:
            status_json = status_resp.json()
            print(
                f"{prefix}status={status_json.get('status')} "
                f"pid={status_json.get('pid')} port={status_json.get('port')} "
                f"ranks={status_json.get('assigned_ranks')}"
            )
            create_timings = status_json.get("create_timings")
            if include_create_timings and create_timings:
                print(f"{prefix}create_timings: {format_create_timings(create_timings)}")
        else:
            print(
                f"{prefix}engine status query failed: "
                f"{status_resp.status_code} {status_resp.text}"
            )
    except Exception as exc:
        print(f"{prefix}engine status query error: {exc}")


def print_visual_comparison(std_time: float, wc_time: float) -> None:
    """Print ASCII bar chart showing the difference."""
    max_time = max(std_time, wc_time)
    bar_width = 50
    std_bar = int(bar_width * std_time / max_time)
    wc_bar = int(bar_width * wc_time / max_time)

    print("\n" + "=" * 80)
    print("VISUAL COMPARISON - Total Sequential Load Time")
    print("=" * 80)
    print()
    print(f"Standard vLLM:      {'#' * std_bar}{'-' * (bar_width - std_bar)} {std_time:.1f}s")
    print(f"Worker Controller:  {'#' * wc_bar}{'-' * (bar_width - wc_bar)} {wc_time:.1f}s")
    print()

    savings_pct = (1 - wc_time / std_time) * 100 if std_time > 0 else 0
    time_saved = std_time - wc_time

    print(f"  Time saved: {time_saved:.1f}s")
    print(f"  Speedup: {std_time / wc_time:.2f}x ({savings_pct:.0f}% faster)")


def save_time_savings_bar_chart(
    results: dict,
    output_path: str | None = None,
) -> str | None:
    """Save a side-by-side startup flow chart emphasizing time savings."""
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
        patches = importlib.import_module("matplotlib.patches")
    except Exception as exc:
        print(f"[chart] skipped: matplotlib not available ({exc})")
        return None

    std_runs = results.get("standard_vllm") or []
    wc_runs = results.get("worker_controller") or []
    if not std_runs or not wc_runs:
        print("[chart] skipped: no benchmark results available")
        return None

    std_aligned = _avg_aligned_metrics(std_runs, _derive_std_aligned_run_metrics)
    wc_aligned = _avg_aligned_metrics(wc_runs, _derive_wc_aligned_run_metrics)

    def _num(value: float | None, default: float = 0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else default

    def _fmt_stage_seconds(value: float) -> str:
        if value < 0.01:
            return f"{value:.6f}s"
        if value < 0.1:
            return f"{value:.4f}s"
        return f"{value:.2f}s"

    std_api = _num(std_aligned.get("remaining_startup_overhead_s"))
    std_engine_base = _num(std_aligned.get("engine_creation_phase_s"))
    std_worker = _num(std_aligned.get("worker_process_startup_estimated_s"))
    std_model = _num(std_aligned.get("model_loading_phase_s"))
    std_infer = _num(std_aligned.get("first_inference_s"))

    wc_api = _num(wc_aligned.get("remaining_startup_overhead_s"))
    wc_engine_base = _num(wc_aligned.get("engine_creation_phase_s"))
    wc_attach = _num(wc_aligned.get("worker_process_startup_estimated_s"))
    wc_model = _num(wc_aligned.get("model_loading_phase_s"))
    wc_infer = _num(wc_aligned.get("first_inference_s"))

    std_engine_total = std_engine_base + std_worker + std_model
    wc_engine_total = wc_engine_base + wc_attach + wc_model

    if max(std_api, std_engine_total, std_infer, wc_api, wc_engine_total, wc_infer) <= 0:
        print("[chart] skipped: no startup stage metrics available")
        return None

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path.cwd() / f"startup_savings_{ts}.png"
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    std_total = _num(std_aligned.get("total_cold_start_s"))
    wc_total = _num(wc_aligned.get("total_cold_start_s"))
    if std_total <= 0:
        std_total = std_api + std_worker + std_model + std_infer
    if wc_total <= 0:
        wc_total = wc_api + wc_attach + wc_model + wc_infer

    std_engine = std_worker + std_model
    wc_engine = wc_attach + wc_model
    engine_creation_saving = std_engine - wc_engine
    total_saving = std_total - wc_total

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 6), dpi=180)

    y_labels = ["Standard vLLM", "Worker Controller"]
    y_pos = [1, 0]
    h = 0.42

    stage_styles = [
        ("API Startup", "#F5F5F5", "", "#333333"),
        ("Worker Startup", "#DEDEDE", "///", "#333333"),
        ("Model Loading", "#C4C4C4", "\\\\", "#333333"),
        ("First Inference", "#8A8A8A", "", "#333333"),
    ]

    std_vals = [std_api, std_worker, std_model, std_infer]
    wc_vals = [wc_api, wc_attach, wc_model, wc_infer]

    def _draw_stacked_bar(y: int, vals: list[float]) -> None:
        left = 0.0
        for idx, (label, face, hatch, edge) in enumerate(stage_styles):
            value = max(0.0, float(vals[idx]))
            ax.barh(
                y,
                value,
                left=left,
                height=h,
                color=face,
                hatch=hatch,
                edgecolor=edge,
                linewidth=1.0,
                label=label if y == y_pos[0] else None,
            )
            if value >= 0.45:
                ax.text(
                    left + value / 2,
                    y,
                    _fmt_stage_seconds(value),
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            left += value

    _draw_stacked_bar(y_pos[0], std_vals)
    _draw_stacked_bar(y_pos[1], wc_vals)

    x_max = max(std_total, wc_total) * 1.18
    ax.set_xlim(0, x_max)
    tick_step = 2 if x_max <= 16 else 5
    ax.set_xticks([i for i in range(0, int(x_max) + 1, tick_step)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Seconds", fontsize=12)

    total_pad = 0.14 * max(1.0, x_max / 12)
    ax.text(std_total + total_pad, y_pos[0] + 0.03, f"Total: {std_total:.2f}s", va="center", fontsize=13)
    ax.text(wc_total + total_pad, y_pos[1] + 0.03, f"Total: {wc_total:.2f}s", va="center", fontsize=13)
    ax.text(
        std_total + total_pad,
        y_pos[0] - 0.10,
        f"Inference: {_fmt_stage_seconds(std_infer)}",
        va="center",
        fontsize=10,
        color="#333333",
    )
    ax.text(
        wc_total + total_pad,
        y_pos[1] - 0.10,
        f"Inference: {_fmt_stage_seconds(wc_infer)}",
        va="center",
        fontsize=10,
        color="#333333",
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.06),
        ncol=2,
        frameon=True,
        fontsize=11,
    )
    legend.get_frame().set_linewidth(1.0)

    std_engine_visible = std_worker + std_model
    wc_engine_visible = wc_attach + wc_model
    savings_text = (
        f"Cold-start saving: {total_saving:+.2f}s\n"
        f"Engine creation (Std): {std_engine_visible:.2f}s\n"
        f"Engine creation (WC): {wc_engine_visible:.2f}s\n"
        f"Engine creation saving: {engine_creation_saving:+.2f}s\n"
        f"Inference (Std/WC): {std_infer:.2f}s / {wc_infer:.2f}s"
    )
    ax.text(
        0.995,
        -0.10,
        savings_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#F7F7F7",
            "edgecolor": "#AAAAAA",
            "linewidth": 0.8,
        },
    )

    fig.text(
        0.5,
        0.01,
        "Figure 1: Latency Breakdown Comparing Standard vLLM and Worker Controller",
        ha="center",
        fontsize=14,
    )

    fig.suptitle(
        f"Overall cold-start saving: {total_saving:+.2f}s",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=(0.02, 0.08, 0.98, 0.92))
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    print(f"[chart] saved side-by-side startup flow chart: {output_file}")
    return str(output_file)


def _derive_std_aligned_run_metrics(run: dict) -> dict[str, float | None]:
    server_ready = run.get("server_ready_time")
    model_load_total = run.get("std_model_load_time")
    engine_init_total = run.get("std_engine_init_time")

    remaining_startup_overhead = None
    if isinstance(server_ready, (int, float)) and isinstance(model_load_total, (int, float)) and isinstance(engine_init_total, (int, float)):
        remaining_startup_overhead = max(0.0, float(server_ready) - float(model_load_total) - float(engine_init_total))

    post_engine_to_routes = run.get("kv_cache_init_done_log__to__api_routes_ready_log_s")
    routes_to_health = run.get("api_routes_ready_log__to__health_ok_log_s")
    post_engine_api_readiness = None
    if isinstance(post_engine_to_routes, (int, float)) and isinstance(routes_to_health, (int, float)):
        post_engine_api_readiness = float(post_engine_to_routes) + float(routes_to_health)

    engine_creation_phase = run.get("engine_process_init_log__to__kv_cache_init_done_log_s")
    model_loading_phase = run.get("model_load_start_log__to__model_loaded_log_s")
    kv_phase = run.get("model_loaded_log__to__kv_cache_init_done_log_s")
    if not isinstance(model_loading_phase, (int, float)) and isinstance(model_load_total, (int, float)):
        model_loading_phase = float(model_load_total)
    if not isinstance(engine_creation_phase, (int, float)) and isinstance(model_loading_phase, (int, float)) and isinstance(kv_phase, (int, float)):
        engine_creation_phase = float(model_loading_phase) + float(kv_phase)

    worker_startup_est = None
    raw_worker_setup = run.get("std_worker_setup_time")
    if isinstance(raw_worker_setup, (int, float)) and raw_worker_setup > 0:
        worker_startup_est = float(raw_worker_setup)
    else:
        engine_init_recv = run.get("engine_process_init_log_received_ts")
        dist_init_recv = run.get("distributed_init_log_received_ts")
        if isinstance(engine_init_recv, (int, float)) and isinstance(dist_init_recv, (int, float)):
            delta = float(dist_init_recv) - float(engine_init_recv)
            if delta >= 0:
                worker_startup_est = delta
            else:
                worker_startup_est = None

    return {
        "startup_ready_s": float(server_ready) if isinstance(server_ready, (int, float)) else None,
        "remaining_startup_overhead_s": remaining_startup_overhead,
        "api_bootstrap_until_engine_init_visible_s": run.get("api_process_started_log__to__engine_process_init_log_s") if isinstance(run.get("api_process_started_log__to__engine_process_init_log_s"), (int, float)) else None,
        "worker_process_startup_estimated_s": worker_startup_est,
        "engine_pre_model_overhead_estimated_s": run.get("distributed_init_log__to__model_load_start_log_s") if isinstance(run.get("distributed_init_log__to__model_load_start_log_s"), (int, float)) else None,
        "engine_creation_phase_s": float(engine_creation_phase) if isinstance(engine_creation_phase, (int, float)) else None,
        "model_loading_phase_s": float(model_loading_phase) if isinstance(model_loading_phase, (int, float)) else None,
        "kv_cache_phase_after_model_load_s": float(kv_phase) if isinstance(kv_phase, (int, float)) else None,
        "post_engine_api_readiness_s": post_engine_api_readiness,
        "post_engine_to_api_routes_ready_s": float(post_engine_to_routes) if isinstance(post_engine_to_routes, (int, float)) else None,
        "api_routes_to_health_ready_s": float(routes_to_health) if isinstance(routes_to_health, (int, float)) else None,
        "first_inference_s": float(run.get("first_inference_time")) if isinstance(run.get("first_inference_time"), (int, float)) else None,
        "total_cold_start_s": float(run.get("total_cold_start")) if isinstance(run.get("total_cold_start"), (int, float)) else None,
    }


def _derive_wc_aligned_run_metrics(run: dict) -> dict[str, float | None]:
    create_time = run.get("create_time")
    api_ready_time = run.get("api_ready_time")
    startup_ready = None
    if isinstance(create_time, (int, float)) and isinstance(api_ready_time, (int, float)):
        startup_ready = float(create_time) + float(api_ready_time)

    create_timings = run.get("create_timings") or {}
    model_load_summary = run.get("model_load_summary") or {}
    model_load_timings = run.get("model_load_timings") or []

    init_engine_time = model_load_summary.get("init_engine_time_seconds")
    init_engine_time = float(init_engine_time) if isinstance(init_engine_time, (int, float)) else None

    worker_loads: list[float] = []
    if isinstance(model_load_timings, list):
        for wt in model_load_timings:
            if not isinstance(wt, dict):
                continue
            effective = wt.get(
                "effective_model_load_time",
                (wt.get("model_runner_init_time", 0) or 0)
                + (wt.get("weight_load_time", 0) or 0),
            )
            if isinstance(effective, (int, float)):
                worker_loads.append(float(effective))
    model_loading_phase = (sum(worker_loads) / len(worker_loads)) if worker_loads else None

    kv_phase = None
    if isinstance(init_engine_time, (int, float)) and isinstance(model_loading_phase, (int, float)):
        kv_phase = max(0.0, init_engine_time - model_loading_phase)

    remaining_startup_overhead = None
    if isinstance(startup_ready, (int, float)) and isinstance(init_engine_time, (int, float)):
        remaining_startup_overhead = max(0.0, startup_ready - init_engine_time)

    post_engine_to_routes = run.get("wc_post_engine_to_api_routes_ready_s")
    if not isinstance(post_engine_to_routes, (int, float)) and isinstance(create_timings, dict):
        maybe = create_timings.get("api_process_spawn_time")
        if isinstance(maybe, (int, float)):
            post_engine_to_routes = float(maybe)

    routes_to_health = run.get("wc_api_routes_to_health_ready_s")
    if isinstance(routes_to_health, (int, float)):
        routes_to_health = float(routes_to_health)
    else:
        routes_to_health = None

    attach_to_workers = run.get("wc_attach_to_prewarmed_workers_s")
    if isinstance(attach_to_workers, (int, float)):
        worker_startup_est = float(attach_to_workers)
    else:
        worker_startup_est = None
        attach_parts: list[float] = []
        if isinstance(create_timings, dict):
            for key_name in [
                "resource_allocation_time",
                "ipc_queue_setup_time",
                "proxy_register_time",
            ]:
                value = create_timings.get(key_name)
                if isinstance(value, (int, float)):
                    attach_parts.append(float(value))
        if attach_parts:
            worker_startup_est = sum(attach_parts)

    return {
        "startup_ready_s": startup_ready,
        "remaining_startup_overhead_s": remaining_startup_overhead,
        "api_bootstrap_until_engine_init_visible_s": float(create_time) if isinstance(create_time, (int, float)) else None,
        "worker_process_startup_estimated_s": worker_startup_est,
        "engine_pre_model_overhead_estimated_s": float(create_timings.get("ipc_queue_setup_time")) if isinstance(create_timings.get("ipc_queue_setup_time"), (int, float)) else None,
        "engine_creation_phase_s": init_engine_time,
        "model_loading_phase_s": model_loading_phase,
        "kv_cache_phase_after_model_load_s": kv_phase,
        "post_engine_api_readiness_s": float(api_ready_time) if isinstance(api_ready_time, (int, float)) else None,
        "post_engine_to_api_routes_ready_s": float(post_engine_to_routes) if isinstance(post_engine_to_routes, (int, float)) else None,
        "api_routes_to_health_ready_s": routes_to_health,
        "first_inference_s": float(run.get("first_inference_time")) if isinstance(run.get("first_inference_time"), (int, float)) else None,
        "total_cold_start_s": float(run.get("total_cold_start")) if isinstance(run.get("total_cold_start"), (int, float)) else None,
    }


def _avg_aligned_metrics(runs: list[dict], derive_fn) -> dict[str, float | None]:
    per_run = [derive_fn(run) for run in runs]
    if not per_run:
        return {}
    keys = list(per_run[0].keys())
    out: dict[str, float | None] = {}
    for key in keys:
        out[key] = average_finite([metric.get(key) for metric in per_run], default=None)
    return out


def _fmt_optional_seconds(value: float | None) -> str:
    if isinstance(value, (int, float)):
        if value < 0.01:
            return f"{value:.6f}"
        return f"{value:.3f}"
    return "N/A"


def _resolve_aligned_metric_value(
    std_aligned: dict[str, float | None],
    wc_aligned: dict[str, float | None],
    metric: str,
) -> tuple[float | None, str, float | None, str]:
    std_v = std_aligned.get(metric)
    wc_v = wc_aligned.get(metric)
    std_src = "raw" if isinstance(std_v, (int, float)) else "na"
    wc_src = "raw" if isinstance(wc_v, (int, float)) else "na"

    if metric == "post_engine_api_readiness_s" and not isinstance(std_v, (int, float)):
        ready = std_aligned.get("startup_ready_s")
        api_boot = std_aligned.get("api_bootstrap_until_engine_init_visible_s")
        engine_create = std_aligned.get("engine_creation_phase_s")
        if (
            isinstance(ready, (int, float))
            and isinstance(api_boot, (int, float))
            and isinstance(engine_create, (int, float))
        ):
            std_v = max(0.0, ready - api_boot - engine_create)
            std_src = "est"

    if metric == "api_routes_to_health_ready_s" and not isinstance(std_v, (int, float)):
        post_ready = std_aligned.get("post_engine_api_readiness_s")
        post_to_routes = std_aligned.get("post_engine_to_api_routes_ready_s")
        if isinstance(post_ready, (int, float)) and isinstance(post_to_routes, (int, float)):
            std_v = max(0.0, post_ready - post_to_routes)
            std_src = "est"

    if metric == "kv_cache_phase_after_model_load_s" and isinstance(wc_v, (int, float)):
        wc_model_load = wc_aligned.get("model_loading_phase_s")
        wc_engine_create = wc_aligned.get("engine_creation_phase_s")
        if (
            wc_v <= 0.001
            and isinstance(wc_model_load, (int, float))
            and isinstance(wc_engine_create, (int, float))
            and wc_model_load > wc_engine_create + 0.2
        ):
            wc_v = None
            wc_src = "scope"

    return std_v, std_src, wc_v, wc_src


def _format_metric_flags(metric: str, std_src: str, wc_src: str) -> str:
    flags: list[str] = []
    non_comparable = {
        "model_loading_phase_s",
        "kv_cache_phase_after_model_load_s",
        "post_engine_api_readiness_s",
        "post_engine_to_api_routes_ready_s",
        "api_routes_to_health_ready_s",
    }
    if metric in non_comparable:
        flags.append("NC")
    if std_src == "est":
        flags.append("S~")
    if wc_src == "est":
        flags.append("W~")
    if std_src == "scope":
        flags.append("S!")
    if wc_src == "scope":
        flags.append("W!")
    if std_src == "na" or wc_src == "na":
        flags.append("NA")
    return ",".join(flags) if flags else "OK"


def print_aligned_startup_comparison(results: dict, models: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ALIGNED STARTUP COMPARISON (shared metric names, seconds)")
    print("=" * 80)

    metric_labels = {
        "startup_ready_s": "Startup ready",
        "remaining_startup_overhead_s": "Remaining startup overhead",
        "api_bootstrap_until_engine_init_visible_s": "API bootstrap -> engine visible",
        "worker_process_startup_estimated_s": "Worker process startup (est.)",
        "engine_pre_model_overhead_estimated_s": "Engine pre-model overhead (est.)",
        "engine_creation_phase_s": "Engine creation phase",
        "model_loading_phase_s": "Model loading phase",
        "kv_cache_phase_after_model_load_s": "KV-cache phase after model load",
        "post_engine_api_readiness_s": "Post-engine API readiness",
        "post_engine_to_api_routes_ready_s": "Post-engine -> API routes ready",
        "api_routes_to_health_ready_s": "API routes -> health ready",
        "first_inference_s": "First inference",
        "total_cold_start_s": "Total cold start",
    }

    def _fmt_with_src(value: float | None, src: str) -> str:
        if not isinstance(value, (int, float)):
            return "N/A"
        base = _fmt_optional_seconds(value)
        return f"~{base}" if src == "est" else base

    def _fmt_delta(std_v: float | None, wc_v: float | None) -> str:
        if not isinstance(std_v, (int, float)) or not isinstance(wc_v, (int, float)):
            return "N/A"
        delta = float(std_v) - float(wc_v)
        return f"{delta:+.3f}"

    ordered_metrics = [
        "startup_ready_s",
        "remaining_startup_overhead_s",
        "api_bootstrap_until_engine_init_visible_s",
        "worker_process_startup_estimated_s",
        "engine_pre_model_overhead_estimated_s",
        "engine_creation_phase_s",
        "model_loading_phase_s",
        "post_engine_api_readiness_s",
        "post_engine_to_api_routes_ready_s",
        "first_inference_s",
        "total_cold_start_s",
    ]

    primary_comparable_metrics = [
        "startup_ready_s",
        "remaining_startup_overhead_s",
        "api_bootstrap_until_engine_init_visible_s",
        "worker_process_startup_estimated_s",
        "engine_pre_model_overhead_estimated_s",
        "engine_creation_phase_s",
        "first_inference_s",
        "total_cold_start_s",
    ]

    for model_info in models:
        model_name = model_info["name"]
        std_runs = results_for_model(results["standard_vllm"], model_name)
        wc_runs = results_for_model(results["worker_controller"], model_name)
        if not std_runs or not wc_runs:
            continue

        std_aligned = _avg_aligned_metrics(std_runs, _derive_std_aligned_run_metrics)
        wc_aligned = _avg_aligned_metrics(wc_runs, _derive_wc_aligned_run_metrics)

        print(f"\n{model_name}:")
        print("  COMPARABLE METRICS")
        print(
            "  {:<32} {:>10} {:>10} {:>11}".format(
                "Metric", "Standard", "WorkerCtrl", "Δ(Std-WC)"
            )
        )
        print("  " + "-" * 68)

        comparable_rows: list[tuple[str, float, str, float, str]] = []

        for metric in ordered_metrics:
            std_v, std_src, wc_v, wc_src = _resolve_aligned_metric_value(std_aligned, wc_aligned, metric)
            if not isinstance(std_v, (int, float)) or not isinstance(wc_v, (int, float)):
                continue

            row = (metric, float(std_v), std_src, float(wc_v), wc_src)
            if metric in primary_comparable_metrics:
                comparable_rows.append(row)
            # Skip other rows in the aligned view to keep output compact.

        for metric, std_v, std_src, wc_v, wc_src in comparable_rows:
            print(
                "  {:<32} {:>10} {:>10} {:>11}".format(
                    metric_labels.get(metric, metric),
                    _fmt_with_src(std_v, std_src),
                    _fmt_with_src(wc_v, wc_src),
                    _fmt_delta(std_v, wc_v),
                )
            )
