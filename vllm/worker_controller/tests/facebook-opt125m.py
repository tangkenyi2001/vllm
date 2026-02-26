#!/usr/bin/env python3
"""
Test cold start latency comparison: Worker Controller vs Standard vLLM.

This test demonstrates the cold start latency reduction achieved by
the Worker Controller's pre-initialized worker pool.

The test compares (both using API servers for fair comparison):
1. Worker Controller: Workers are already initialized, CUDA context reused
2. Standard vLLM API Server: Full initialization each time including CUDA context

Key insight: The Worker Controller is better when you need to load/unload
multiple models sequentially because it reuses the CUDA context and
distributed setup, while standard vLLM must reinitialize everything each time.

Models tested:
- facebook/opt-125m
"""

import os
import atexit
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any

import requests
import uvicorn

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no display server needed)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

import vllm.worker_controller.worker_controller_server as wc_server

from vllm.worker_controller.worker_controller import WorkerController
from vllm.worker_controller.worker_controller_server import app
from vllm.worker_controller.tests.utils.benchmark_utils import (
    average_finite,
    is_finite_number,
    prewarm_model_files,
    print_banner,
    results_for_model,
    avg_metric,
    start_subprocess_log_collector,
)

BASE_URL = "http://localhost:21000"

# Global reference to server thread
_server_thread = None
_server = None

# Test configuration
MODELS = [
    {
        "name": "facebook/opt-125m (run-1)",
        "load_name": "facebook/opt-125m",
        "uuid": "opt-125m-a",
    },
    {
        "name": "facebook/opt-125m (run-2)",
        "load_name": "facebook/opt-125m",
        "uuid": "opt-125m-b",
    },
    {
        "name": "facebook/opt-125m (run-3)",
        "load_name": "facebook/opt-125m",
        "uuid": "opt-125m-c",
    },
]

TEST_PROMPT = "Hello, my name is"
RUNS_PER_MODEL = 1
PREWARM_MODEL_FILES = True
PREWARM_DOWNLOAD_IF_MISSING = True

MODEL_SHARD_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
)

# GPU baseline free memory (MiB) — set in main() after warmup,
# used by measure_worker_controller_cold_start() for cleanup verification.
_gpu_baseline_free_mb: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TeeOutput:
    """Duplicate writes to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def _ts(seconds: float | None) -> str:
    """Format a timestamp offset concisely."""
    if not is_finite_number(seconds):
        return "N/A"
    return f"{float(seconds):.3f}s"


def _dur(seconds: float | None) -> str:
    """Format a duration in parentheses."""
    if not is_finite_number(seconds):
        return ""
    return f"(duration: {float(seconds):.3f}s)"


def _get_gpu_free_mb(gpu_index: int = 0) -> float | None:
    """Query free GPU memory in MiB via nvidia-smi (no CUDA context needed)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", f"--id={gpu_index}"],
            text=True, timeout=5,
        ).strip()
        return float(out.split("\n")[0])
    except Exception:
        return None


def wait_for_gpu_idle(
    gpu_index: int = 0,
    stable_threshold_mb: float = 50.0,
    stable_checks: int = 3,
    poll_interval: float = 1.0,
    timeout: float = 30.0,
    min_free_mb: float | None = None,
    label: str = "",
) -> bool:
    """Wait until GPU free memory stabilizes and optionally reaches a minimum.

    Polls nvidia-smi until the free memory reading changes by less than
    *stable_threshold_mb* for *stable_checks* consecutive polls **and**
    (if *min_free_mb* is set) free memory is at least that value.

    Returns True if conditions were met, False on timeout.
    """
    start = time.time()
    prev_free = _get_gpu_free_mb(gpu_index)
    if prev_free is None:
        time.sleep(5)
        return True

    consecutive = 0
    while time.time() - start < timeout:
        time.sleep(poll_interval)
        free = _get_gpu_free_mb(gpu_index)
        if free is None:
            continue
        delta = abs(free - prev_free)
        if delta < stable_threshold_mb:
            consecutive += 1
            stable = consecutive >= stable_checks
            above_min = min_free_mb is None or free >= min_free_mb
            if stable and above_min:
                elapsed = time.time() - start
                if label:
                    print(f"  [{label}] GPU memory stable at {free:.0f} MiB free "
                          f"(waited {elapsed:.1f}s)")
                return True
        else:
            consecutive = 0
        prev_free = free

    elapsed = time.time() - start
    free = _get_gpu_free_mb(gpu_index) or prev_free
    if label:
        min_info = f", needed {min_free_mb:.0f}" if min_free_mb else ""
        print(f"  [{label}] GPU idle wait timed out after {elapsed:.1f}s "
              f"(free: {free:.0f} MiB{min_info})")
    return False


def print_event_timeline(
    prefix: str,
    events: list[tuple[int, str, float | None, float | None]],
) -> None:
    """Print a clean event timeline.

    Each event is (step_number, label, offset_from_start, duration_or_none).
    """
    print(f"\n  {prefix} Event Timeline:")
    print(f"  {'─' * 64}")
    for step, label, offset, duration in events:
        offset_str = _ts(offset).rjust(9)
        dur_str = ""
        if duration is not None and is_finite_number(duration):
            dur_str = f"  {_dur(duration)}"
        print(f"  [{prefix}] {step}. {label:<30} t={offset_str}{dur_str}")
    print(f"  {'─' * 64}")


# ---------------------------------------------------------------------------
# Worker Controller lifecycle
# ---------------------------------------------------------------------------

def start_worker_controller():
    """Start the worker controller server in a background thread."""
    global _server_thread, _server

    wc_server.worker_controller = WorkerController(start_port=21002)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=21000,
        log_level="warning",
        access_log=False,
    )
    _server = uvicorn.Server(config)

    _server_thread = threading.Thread(target=_server.run, daemon=True)
    _server_thread.start()


def stop_worker_controller():
    """Stop the worker controller server."""
    global _server, _server_thread

    if _server is not None:
        _server.should_exit = True
        if _server_thread is not None:
            _server_thread.join(timeout=5)
        if wc_server.worker_controller is not None:
            wc_server.worker_controller.executor.shutdown()
            wc_server.worker_controller = None
        _server = None
        _server_thread = None


def wait_for_controller():
    """Wait for the worker controller to be ready."""
    for _ in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("Worker Controller did not become ready in time")


# ---------------------------------------------------------------------------
# Worker Controller cold start measurement
# ---------------------------------------------------------------------------

def measure_worker_controller_cold_start(
    model_name: str,
    engine_uuid: str,
    result_label: str | None = None,
) -> dict | None:
    """
    Measure cold start time using Worker Controller.

    Captures these lifecycle events:
      1. API Server START           – we begin the create call
      2. Engine Creation START      – server receives & processes the request
      3. Model Loading START        – workers begin loading model weights
      4. Model Loading END          – workers finish loading model weights
      5. Engine Creation END        – engine fully initialised (KV cache, warmup)
      6. API Server Startup END     – engine /health returns 200
    """
    print_banner(f"Worker Controller Cold Start: {model_name}", width=60)

    total_start = time.time()

    # ── 1. API Server START (= Engine Create request sent) ──────────────
    create_start = time.time()
    print(f"  [WC] 1. API Server START / Engine Create request sent")

    create_payload = {
        "engine_uuid": engine_uuid,
        "model": model_name,
        "gpu_memory_utilization": 0.3,
        "enforce_eager": True,
    }

    resp = requests.post(
        f"{BASE_URL}/engines",
        json=create_payload,
        timeout=300,
    )
    create_time = time.time() - create_start

    if resp.status_code != 200:
        print(f"  [WC] ERROR: create failed ({resp.status_code}): {resp.text[:200]}")
        return None

    create_result = resp.json()
    create_timings = create_result.get("create_timings") or {}
    port = create_result["port"]
    engine_url = f"http://localhost:{port}"

    # ── 2. Engine Creation START (from server-side timings) ─────────────
    resource_alloc = create_timings.get("resource_allocation_time")
    ipc_setup = create_timings.get("ipc_queue_setup_time")
    proxy_register = create_timings.get("proxy_register_time")
    api_spawn = create_timings.get("api_process_spawn_time")

    attach_to_workers_s = None
    attach_parts = [resource_alloc, ipc_setup, proxy_register]
    numeric_parts = [float(v) for v in attach_parts if is_finite_number(v)]
    if numeric_parts:
        attach_to_workers_s = sum(numeric_parts)

    print(f"  [WC] 2. Engine Creation START (create call returned in {_ts(create_time)})")
    if attach_to_workers_s is not None:
        print(f"         attach to pre-warmed workers: {_ts(attach_to_workers_s)}")
    if is_finite_number(api_spawn):
        print(f"         spawn API process: {_ts(api_spawn)}")

    # ── Wait for engine API health ──────────────────────────────────────
    api_ready_start = time.time()
    health_ready = False

    for i in range(60):
        try:
            resp = requests.get(f"{engine_url}/health", timeout=5)
            if resp.status_code == 200:
                health_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    api_ready_time = time.time() - api_ready_start

    if not health_ready:
        print(f"  [WC] ERROR: engine API did not become healthy in time")
        return None

    # ── Fetch timing data from engine endpoints ─────────────────────────
    model_load_timings = None
    model_load_summary = None
    startup_timing = None
    routes_to_health = None

    try:
        startup_resp = requests.get(f"{engine_url}/startup_timing", timeout=3)
        if startup_resp.status_code == 200:
            startup_timing = startup_resp.json()
            maybe = startup_timing.get("api_routes_to_first_health_s")
            if is_finite_number(maybe):
                routes_to_health = float(maybe)
    except Exception:
        pass

    wallclocks = None

    try:
        timings_resp = requests.get(f"{engine_url}/model_load_timings", timeout=3)
        if timings_resp.status_code == 200:
            timings_data = timings_resp.json()
            model_load_timings = timings_data.get("worker_timings")
            model_load_summary = timings_data.get("summary")
            wallclocks = timings_data.get("wallclocks")

        if not model_load_timings:
            ctrl_resp = requests.get(
                f"{BASE_URL}/engines/{engine_uuid}/load_timings", timeout=5
            )
            if ctrl_resp.status_code == 200:
                ctrl_data = ctrl_resp.json()
                model_load_timings = ctrl_data.get("worker_timings")
                summary = ctrl_data.get("summary")
                if isinstance(summary, dict):
                    if not isinstance(model_load_summary, dict):
                        model_load_summary = {}
                    model_load_summary.update(summary)
    except Exception:
        pass

    # ── Extract key durations from timing data ──────────────────────────
    init_engine_time = None
    load_rpc_time = None
    if isinstance(model_load_summary, dict):
        v = model_load_summary.get("init_engine_time_seconds")
        if is_finite_number(v):
            init_engine_time = float(v)
        v = model_load_summary.get("remote_executor_load_model_rpc_time")
        if is_finite_number(v):
            load_rpc_time = float(v)

    worker_weight_load_avg = None
    worker_effective_load_avg = None
    if isinstance(model_load_timings, list) and model_load_timings:
        weight_loads = [
            float(wt.get("weight_load_time", 0))
            for wt in model_load_timings
            if is_finite_number(wt.get("weight_load_time"))
        ]
        effective_loads = [
            float(
                wt.get(
                    "effective_model_load_time",
                    (wt.get("model_runner_init_time", 0) or 0)
                    + (wt.get("weight_load_time", 0) or 0),
                )
            )
            for wt in model_load_timings
            if isinstance(wt, dict)
        ]
        if weight_loads:
            worker_weight_load_avg = sum(weight_loads) / len(weight_loads)
        if effective_loads:
            worker_effective_load_avg = sum(effective_loads) / len(effective_loads)

    # ── 3–5. Compute offsets from wallclock timestamps ─────────────────
    #   Wallclocks are absolute time.time() values from the spawned API
    #   process.  Since both processes share the same clock, we can compute
    #   precise offsets relative to total_start.
    engine_creation_start_offset = None
    model_load_start_offset = None
    model_load_end_offset = None
    engine_creation_end_offset = None
    model_load_duration = None
    engine_creation_duration = None
    has_precise = False

    if isinstance(wallclocks, dict):
        wc_ei_start = wallclocks.get("engine_init_start_wallclock")
        wc_ei_end = wallclocks.get("engine_init_end_wallclock")
        wc_ml_start = wallclocks.get("model_load_start_wallclock")
        wc_ml_end = wallclocks.get("model_load_end_wallclock")

        if is_finite_number(wc_ei_start):
            engine_creation_start_offset = float(wc_ei_start) - total_start
            has_precise = True
        if is_finite_number(wc_ei_end):
            engine_creation_end_offset = float(wc_ei_end) - total_start
        if is_finite_number(wc_ml_start):
            model_load_start_offset = float(wc_ml_start) - total_start
        if is_finite_number(wc_ml_end):
            model_load_end_offset = float(wc_ml_end) - total_start
        if is_finite_number(wc_ml_start) and is_finite_number(wc_ml_end):
            model_load_duration = float(wc_ml_end) - float(wc_ml_start)
        if is_finite_number(wc_ei_start) and is_finite_number(wc_ei_end):
            engine_creation_duration = float(wc_ei_end) - float(wc_ei_start)

    # Fall back to estimated offsets if wallclocks are unavailable
    if not has_precise:
        model_load_start_offset = create_time
        model_load_duration = worker_effective_load_avg
        if is_finite_number(model_load_start_offset) and is_finite_number(model_load_duration):
            model_load_end_offset = model_load_start_offset + model_load_duration
        if is_finite_number(create_time) and is_finite_number(init_engine_time):
            engine_creation_end_offset = create_time + init_engine_time
        engine_creation_duration = init_engine_time

    ts_marker = "t=" if has_precise else "t~="
    print(f"  [WC] 3. Model Loading START           {ts_marker}{_ts(model_load_start_offset)}")
    print(f"  [WC] 4. Model Loading END             {ts_marker}{_ts(model_load_end_offset)}  {_dur(model_load_duration)}")
    print(f"  [WC] 5. Engine Creation END           {ts_marker}{_ts(engine_creation_end_offset)}  {_dur(engine_creation_duration)}")

    # ── 6. API Server Startup END ───────────────────────────────────────
    api_server_end_offset = create_time + api_ready_time
    print(f"  [WC] 6. API Server Startup END        t={_ts(api_server_end_offset)}  {_dur(api_server_end_offset)}")

    # ── First inference ─────────────────────────────────────────────────
    inference_start = time.time()
    resp = requests.post(
        f"{engine_url}/v1/completions",
        json={
            "prompt": TEST_PROMPT,
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=60,
    )
    first_inference_time = time.time() - inference_start

    generated = None
    if resp.status_code == 200:
        generated = resp.json()["choices"][0]["text"]
        print(f"  [WC] First inference: {_ts(first_inference_time)}  output={generated!r}")
    else:
        print(f"  [WC] First inference FAILED: {resp.status_code}")
        first_inference_time = float("inf")

    # ── Print clean timeline ────────────────────────────────────────────
    print_event_timeline("WC", [
        (1, "API Server START",       0.0,                         None),
        (2, "Engine Creation START",  engine_creation_start_offset, None),
        (3, "Model Loading START",    model_load_start_offset,     None),
        (4, "Model Loading END",      model_load_end_offset,       model_load_duration),
        (5, "Engine Creation END",    engine_creation_end_offset,   engine_creation_duration),
        (6, "API Server Startup END", api_server_end_offset,       api_server_end_offset),
    ])

    # ── Print detailed sub-timings (compact) ────────────────────────────
    print(f"\n  Detailed sub-timings:")
    print(f"    create_time (HTTP roundtrip):       {_ts(create_time)}")
    print(f"    api_ready_time (health wait):       {_ts(api_ready_time)}")
    if attach_to_workers_s is not None:
        print(f"    attach to workers:                  {_ts(attach_to_workers_s)}")
    if is_finite_number(api_spawn):
        print(f"    API process spawn:                  {_ts(api_spawn)}")
    if is_finite_number(load_rpc_time):
        print(f"    load_model RPC:                     {_ts(load_rpc_time)}")
    if is_finite_number(worker_weight_load_avg):
        print(f"    weight load (worker avg):           {_ts(worker_weight_load_avg)}")
    if is_finite_number(worker_effective_load_avg):
        print(f"    effective model load (worker avg):  {_ts(worker_effective_load_avg)}")
    if is_finite_number(init_engine_time):
        print(f"    init engine (profile+KV+warmup):   {_ts(init_engine_time)}")
    if is_finite_number(routes_to_health):
        print(f"    API routes -> health ready:         {_ts(routes_to_health)}")

    if isinstance(model_load_timings, list) and model_load_timings:
        t = model_load_timings[0]
        print(f"    first worker breakdown: "
              f"config={t.get('config_time', 0):.3f}s "
              f"dist={t.get('dist_init_time', 0):.3f}s "
              f"runner={t.get('model_runner_init_time', 0):.3f}s "
              f"weight={t.get('weight_load_time', 0):.3f}s "
              f"total={t.get('total_time', 0):.3f}s")

    # ── Cleanup ─────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    print(f"\n  [WC] Deleting engine {engine_uuid}")
    resp = requests.delete(f"{BASE_URL}/engines/{engine_uuid}", timeout=60)
    if resp.status_code != 200:
        print(f"  [WC] WARNING: delete returned {resp.status_code}: {resp.text[:200]}")

    # Wait for GPU memory to return to baseline.
    # The delete triggers unload_model → gc.collect → cuda.empty_cache
    # on the workers, but CUDA driver-level reclamation is async.
    # Use _gpu_baseline_free_mb (captured before first model load) as the
    # target so we don't proceed while memory is still held.
    baseline = _gpu_baseline_free_mb
    # Allow some tolerance — workers hold small residual buffers
    min_target = (baseline - 200) if baseline else None
    ok = wait_for_gpu_idle(min_free_mb=min_target, label="WC", timeout=30)
    if not ok and min_target:
        print(f"  [WC] WARNING: GPU memory did not return to baseline "
              f"({min_target:.0f} MiB). Waiting extra 5s...")
        time.sleep(5)

    # Verify workers report free status
    try:
        workers_resp = requests.get(f"{BASE_URL}/workers", timeout=3)
        if workers_resp.status_code == 200:
            workers = workers_resp.json().get("workers", [])
            free_workers = sum(1 for w in workers if w.get("status") == "free")
            print(f"  [WC] Workers after delete: {free_workers}/{len(workers)} free")
    except Exception:
        pass

    # ── Build result dict ───────────────────────────────────────────────
    result: dict[str, Any] = {
        "model": result_label or model_name,
        "create_time": create_time,
        "api_ready_time": api_ready_time,
        "first_inference_time": first_inference_time,
        "total_cold_start": total_time,
        "has_precise_wallclocks": has_precise,
        "wc_events": {
            "api_server_start": 0.0,
            "engine_creation_start": engine_creation_start_offset,
            "model_loading_start": model_load_start_offset,
            "model_loading_end": model_load_end_offset,
            "engine_creation_end": engine_creation_end_offset,
            "api_server_startup_end": api_server_end_offset,
        },
        "wc_durations": {
            "model_loading": model_load_duration,
            "engine_creation": engine_creation_duration,
            "api_server_startup": api_server_end_offset,
            "first_inference": first_inference_time,
        },
    }

    if model_load_timings:
        result["model_load_timings"] = model_load_timings
    if model_load_summary:
        result["model_load_summary"] = model_load_summary
    if create_timings:
        result["create_timings"] = create_timings
    if is_finite_number(attach_to_workers_s):
        result["wc_attach_to_prewarmed_workers_s"] = float(attach_to_workers_s)
    if startup_timing:
        result["wc_startup_timing"] = startup_timing

    return result


# ---------------------------------------------------------------------------
# Standard vLLM cold start measurement
# ---------------------------------------------------------------------------

def _extract_log_event_time(
    log_lines: list[tuple[float, str]],
    pattern: str,
) -> float | None:
    """Find the received_ts of the first log line containing pattern."""
    for entry in log_lines:
        if isinstance(entry, tuple) and len(entry) == 2:
            received_ts, line = float(entry[0]), str(entry[1])
        else:
            continue
        if pattern in line:
            return received_ts
    return None


def measure_standard_vllm_cold_start(
    model_name: str,
    port: int = 8000,
    result_label: str | None = None,
) -> dict | None:
    """
    Measure cold start time using standard vLLM API server (vllm serve).

    Captures these lifecycle events:
      1. API Server START              – subprocess spawned
      2. Engine Creation START         – "Initializing a V1 LLM engine"
      3. Worker Process START          – MultiprocExecutor / distributed init
      4. Model Loading START           – "Starting to load model"
      5. Model Loading END             – "Model loading took"
      6. Worker Process Startup END    – MultiprocExecutor startup complete
      7. Engine Creation END           – "init engine ... took"
      8. API Server Startup END        – /health returns 200
    """
    print_banner(f"Standard vLLM API Server Cold Start: {model_name}", width=60)

    total_start = time.time()

    # ── 1. API Server START ─────────────────────────────────────────────
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env.pop("VLLM_CONFIGURE_LOGGING", None)
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["TQDM_DISABLE"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.3",
        "--enforce-eager",
    ]

    print(f"  [STD] 1. API Server START (spawning on port {port})")
    spawn_start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=repo_root,
    )
    spawn_time = time.time() - spawn_start
    print(f"         spawn call: {_ts(spawn_time)}")

    # Collect subprocess logs asynchronously with timestamps
    std_log_lines: list[tuple[float, str]] = []
    start_subprocess_log_collector(
        proc,
        std_log_lines,
        prefix="  [std-log] ",
        capture_receive_ts=True,
    )

    # ── Wait for API server health ──────────────────────────────────────
    api_url = f"http://localhost:{port}"
    server_ready = False
    server_ready_time = None

    for i in range(120):
        if proc.poll() is not None:
            tail = "\n".join(line for _, line in std_log_lines[-50:])
            print(f"  [STD] ERROR: API server exited early\n{tail}")
            return None
        try:
            resp = requests.get(f"{api_url}/health", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                server_ready_time = time.time() - spawn_start
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            pass
        time.sleep(0.5)

    if not server_ready:
        print("  [STD] ERROR: Server did not become ready in time")
        proc.terminate()
        proc.wait()
        return None

    # ── Extract event timestamps from subprocess logs ───────────────────
    event_patterns = {
        "engine_creation_start": "Initializing a V1 LLM engine",
        "worker_process_start": "world_size=",
        "model_loading_start": "Starting to load model",
        "model_loading_end": "Model loading took",
        "worker_process_end": "MultiprocExecutor startup breakdown",
        "engine_creation_end": "init engine (profile, create kv cache, warmup model) took",
        "api_routes_ready": "Starting vLLM API server",
    }

    event_offsets: dict[str, float | None] = {}
    for event_name, pattern in event_patterns.items():
        ts = _extract_log_event_time(std_log_lines, pattern)
        if ts is not None:
            event_offsets[event_name] = ts - spawn_start
        else:
            event_offsets[event_name] = None

    event_offsets["api_server_start"] = 0.0
    event_offsets["api_server_startup_end"] = server_ready_time

    # ── Parse durations from log text ───────────────────────────────────
    durations: dict[str, float | None] = {}

    for entry in std_log_lines:
        if isinstance(entry, tuple) and len(entry) == 2:
            line = str(entry[1])
        else:
            line = str(entry)

        m = re.search(r"Loading weights took\s+([0-9.]+)\s+seconds", line)
        if m:
            durations["weight_load"] = float(m.group(1))
        m = re.search(
            r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds",
            line,
        )
        if m:
            durations["model_load_total"] = float(m.group(1))
        m = re.search(
            r"init engine \(profile, create kv cache, warmup model\) took\s+([0-9.]+)\s+seconds",
            line,
        )
        if m:
            durations["engine_init"] = float(m.group(1))
        m = re.search(
            r"MultiprocExecutor startup breakdown: worker_ready=([0-9.]+)s, mq_ready=([0-9.]+)s",
            line,
        )
        if m:
            durations["worker_ready"] = float(m.group(1))
            durations["mq_ready"] = float(m.group(2))

    # Compute phase durations from offsets
    def _offset_dur(start_key: str, end_key: str) -> float | None:
        s = event_offsets.get(start_key)
        e = event_offsets.get(end_key)
        if is_finite_number(s) and is_finite_number(e):
            return max(0.0, float(e) - float(s))
        return None

    model_load_dur = _offset_dur("model_loading_start", "model_loading_end")
    engine_creation_dur = _offset_dur("engine_creation_start", "engine_creation_end")

    # ── Print events ────────────────────────────────────────────────────
    print(f"  [STD] 2. Engine Creation START        t={_ts(event_offsets.get('engine_creation_start'))}")
    print(f"  [STD] 3. Worker Process START         t={_ts(event_offsets.get('worker_process_start'))}")
    print(f"  [STD] 4. Model Loading START          t={_ts(event_offsets.get('model_loading_start'))}")
    print(f"  [STD] 5. Model Loading END            t={_ts(event_offsets.get('model_loading_end'))}  {_dur(model_load_dur)}")
    print(f"  [STD] 6. Worker Process Startup END   t={_ts(event_offsets.get('worker_process_end'))}")
    print(f"  [STD] 7. Engine Creation END          t={_ts(event_offsets.get('engine_creation_end'))}  {_dur(engine_creation_dur)}")
    print(f"  [STD] 8. API Server Startup END       t={_ts(server_ready_time)}  {_dur(server_ready_time)}")

    # ── Print clean timeline ────────────────────────────────────────────
    print_event_timeline("STD", [
        (1, "API Server START",           0.0,                                        None),
        (2, "Engine Creation START",      event_offsets.get("engine_creation_start"),  None),
        (3, "Worker Process START",       event_offsets.get("worker_process_start"),   None),
        (4, "Model Loading START",        event_offsets.get("model_loading_start"),    None),
        (5, "Model Loading END",          event_offsets.get("model_loading_end"),      model_load_dur),
        (6, "Worker Process Startup END", event_offsets.get("worker_process_end"),     None),
        (7, "Engine Creation END",        event_offsets.get("engine_creation_end"),    engine_creation_dur),
        (8, "API Server Startup END",     server_ready_time,                          server_ready_time),
    ])

    # ── Print parsed durations ──────────────────────────────────────────
    print(f"\n  Parsed durations from logs:")
    for key, val in sorted(durations.items()):
        print(f"    {key}: {_ts(val)}")

    # ── First inference ─────────────────────────────────────────────────
    print(f"\n  Running first inference...")
    inference_start = time.time()

    try:
        resp = requests.post(
            f"{api_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": TEST_PROMPT,
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=60,
        )
        first_inference_time = time.time() - inference_start

        if resp.status_code == 200:
            generated = resp.json()["choices"][0]["text"]
            print(f"  [STD] First inference: {_ts(first_inference_time)}  output={generated!r}")
        else:
            print(f"  [STD] First inference FAILED: {resp.status_code} - {resp.text[:200]}")
            first_inference_time = float("inf")
    except Exception as e:
        print(f"  [STD] First inference ERROR: {e}")
        first_inference_time = float("inf")

    total_time = time.time() - total_start

    # ── Cleanup ─────────────────────────────────────────────────────────
    print(f"  Cleaning up server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Wait for GPU memory to fully stabilize after process exit.
    # The terminated process's CUDA allocations are freed asynchronously
    # by the driver — poll until free memory stops changing.
    wait_for_gpu_idle(label="STD")

    # ── Build result dict ───────────────────────────────────────────────
    result: dict[str, Any] = {
        "model": result_label or model_name,
        "server_ready_time": server_ready_time,
        "first_inference_time": first_inference_time,
        "total_cold_start": total_time,
        "std_events": event_offsets,
        "std_durations": durations,
        "std_phase_durations": {
            "model_loading": model_load_dur,
            "engine_creation": engine_creation_dur,
            "api_server_startup": server_ready_time,
            "first_inference": first_inference_time,
        },
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.getcwd(), f"benchmark_output_{ts}.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeOutput(original_stdout, log_file)
    sys.stderr = _TeeOutput(original_stderr, log_file)
    print(f"[log] writing benchmark output to: {log_path}")

    # ── 0. Model file prewarm ──────────────────────────────────────────
    if PREWARM_MODEL_FILES:
        print_banner("0: Model File Prewarm")
        for model_info in MODELS:
            model_name = model_info.get("load_name", model_info["name"])
            print(f"  [PREWARM] {model_name}")
            prewarm = prewarm_model_files(
                model_name,
                prewarm_download_if_missing=PREWARM_DOWNLOAD_IF_MISSING,
                model_shard_extensions=MODEL_SHARD_EXTENSIONS,
            )
            if prewarm.get("ok"):
                gb = prewarm["bytes"] / (1024**3)
                print(
                    f"  [PREWARM] Done: files={prewarm['files']} "
                    f"size={gb:.2f} GiB  time={prewarm['seconds']:.2f}s"
                )
            else:
                print(f"  [PREWARM] Skipped: {prewarm.get('error')}")

    results: dict[str, list[dict]] = {
        "worker_controller": [],
        "standard_vllm": [],
    }

    # ── 1. Worker Controller runs ──────────────────────────────────────
    print_banner("1: Worker Controller")

    start_worker_controller()
    atexit.register(stop_worker_controller)
    wait_for_controller()

    # ── Warmup run (discarded) ──────────────────────────────────────────
    # The first engine create triggers forkserver startup + module preload
    # (~4s overhead) and cold GPU/page caches (~1s).  Running a throwaway
    # create/health/delete cycle here ensures all subsequent measured runs
    # start from an equally warm state.
    warmup_model = MODELS[0].get("load_name", MODELS[0]["name"])
    warmup_uuid = "warmup-000"
    print(f"\n  [WC] Warmup run: {warmup_model} (discarded)")
    warmup_result = measure_worker_controller_cold_start(
        warmup_model,
        warmup_uuid,
        result_label="warmup",
    )
    if warmup_result:
        print(f"  [WC] Warmup complete: {_ts(warmup_result['total_cold_start'])}")
    else:
        print(f"  [WC] Warmup did not complete cleanly (continuing anyway)")
    wait_for_gpu_idle(label="WC-warmup")

    # Capture GPU baseline: free memory with workers alive but no model loaded.
    # This is the target we expect to return to after each engine delete.
    global _gpu_baseline_free_mb
    _gpu_baseline_free_mb = _get_gpu_free_mb()
    if _gpu_baseline_free_mb:
        print(f"  [WC] GPU baseline (workers idle, no model): {_gpu_baseline_free_mb:.0f} MiB free")

    for model_info in MODELS:
        model_name = model_info.get("load_name", model_info["name"])
        model_label = model_info["name"]
        base_uuid = model_info["uuid"]
        for run_idx in range(1, RUNS_PER_MODEL + 1):
            engine_uuid = f"{base_uuid}-run{run_idx}"
            print(f"\n  [Worker Controller] run {run_idx}/{RUNS_PER_MODEL} for {model_label}")
            result = measure_worker_controller_cold_start(
                model_name,
                engine_uuid,
                result_label=model_label,
            )
            if result:
                result["run"] = run_idx
                results["worker_controller"].append(result)
            # GPU cleanup is handled inside measure_worker_controller_cold_start
            # via wait_for_gpu_idle() after engine delete

    stop_worker_controller()
    wait_for_gpu_idle(label="WC-shutdown")

    # ── 2. Standard vLLM runs ──────────────────────────────────────────
    print_banner("2: Standard vLLM")

    # ── Warmup run (discarded) ──────────────────────────────────────────
    # First STD spawn imports all vLLM modules from disk, warming the OS
    # page cache.  Without this, run-1 is artificially slower than runs 2+.
    std_warmup_model = MODELS[0].get("load_name", MODELS[0]["name"])
    print(f"\n  [STD] Warmup run: {std_warmup_model} (discarded)")
    std_warmup_result = measure_standard_vllm_cold_start(
        std_warmup_model,
        port=7999,
        result_label="warmup",
    )
    if std_warmup_result:
        print(f"  [STD] Warmup complete: {_ts(std_warmup_result['total_cold_start'])}")
    else:
        print(f"  [STD] Warmup did not complete cleanly (continuing anyway)")
    wait_for_gpu_idle(label="STD-warmup")

    for model_index, model_info in enumerate(MODELS):
        model_name = model_info.get("load_name", model_info["name"])
        model_label = model_info["name"]
        for run_idx in range(1, RUNS_PER_MODEL + 1):
            port = 8000 + model_index * 10 + run_idx - 1
            print(f"\n  [Standard vLLM] run {run_idx}/{RUNS_PER_MODEL} for {model_label} (port {port})")
            result = measure_standard_vllm_cold_start(
                model_name,
                port=port,
                result_label=model_label,
            )
            if result:
                result["run"] = run_idx
                results["standard_vllm"].append(result)
            # GPU cleanup is handled inside measure_standard_vllm_cold_start
            # via wait_for_gpu_idle() after process termination

    # ── Results Summary ────────────────────────────────────────────────
    print_banner("RESULTS")
    print(f"Averaged over {RUNS_PER_MODEL} run(s) per model\n")

    header = "{:<25} {:>12} {:>12} {:>10} {:>8}".format(
        "Model", "Std (s)", "WC (s)", "Speedup", "Runs"
    )
    print(header)
    print("-" * len(header))

    for model_info in MODELS:
        model_name = model_info["name"]
        std_runs = results_for_model(results["standard_vllm"], model_name)
        wc_runs = results_for_model(results["worker_controller"], model_name)

        if std_runs and wc_runs:
            std_time = avg_metric(std_runs, "total_cold_start")
            wc_time = avg_metric(wc_runs, "total_cold_start")
            speedup = std_time / wc_time if wc_time > 0 else float("inf")
            n = min(len(std_runs), len(wc_runs))
            print(
                "{:<25} {:>12.2f} {:>12.2f} {:>9.1f}x {:>8}".format(
                    model_name[:25], std_time, wc_time, speedup,
                    f"{n}/{RUNS_PER_MODEL}",
                )
            )

    # ── Cumulative totals ──────────────────────────────────────────────
    std_cumulative = sum(r["total_cold_start"] for r in results["standard_vllm"])
    wc_cumulative = sum(r["total_cold_start"] for r in results["worker_controller"])
    diff = std_cumulative - wc_cumulative
    speedup = std_cumulative / wc_cumulative if wc_cumulative > 0 else float("inf")

    print_banner(f"CUMULATIVE ({len(MODELS)} models x {RUNS_PER_MODEL} runs)")
    print(f"  Standard vLLM:      {std_cumulative:.2f}s")
    print(f"  Worker Controller:  {wc_cumulative:.2f}s")
    print(f"  Time saved:         {diff:.2f}s")
    print(f"  Speedup:            {speedup:.2f}x")

    # ── Detailed breakdown per model ───────────────────────────────────
    print_banner("DETAILED BREAKDOWN")

    for model_info in MODELS:
        model_name = model_info["name"]
        std_runs = results_for_model(results["standard_vllm"], model_name)
        wc_runs = results_for_model(results["worker_controller"], model_name)

        std = std_runs[-1] if std_runs else None
        wc = wc_runs[-1] if wc_runs else None

        print(f"\n  {model_name}")
        print(f"  {'─' * 72}")

        # Build side-by-side comparison rows: (label, std_val, wc_val)
        rows: list[tuple[str, str, str]] = []

        # --- Milestone offsets (t= from process start) ---
        std_ev = std.get("std_events", {}) if std else {}
        wc_ev = wc.get("wc_events", {}) if wc else {}

        milestone_rows = [
            ("Engine Creation START",  std_ev.get("engine_creation_start"), wc_ev.get("engine_creation_start")),
            ("Model Loading START",    std_ev.get("model_loading_start"),   wc_ev.get("model_loading_start")),
            ("Model Loading END",      std_ev.get("model_loading_end"),     wc_ev.get("model_loading_end")),
            ("Engine Creation END",    std_ev.get("engine_creation_end"),   wc_ev.get("engine_creation_end")),
            ("API Server Ready",       std_ev.get("api_server_startup_end"), wc_ev.get("api_server_startup_end")),
        ]
        rows.append(("── Milestones (offset) ──", "", ""))
        for label, sv, wv in milestone_rows:
            rows.append((label, _ts(sv), _ts(wv)))

        # --- Phase durations ---
        std_dur = std.get("std_phase_durations", {}) if std else {}
        wc_dur = wc.get("wc_durations", {}) if wc else {}

        duration_rows = [
            ("Model Loading",    std_dur.get("model_loading"),    wc_dur.get("model_loading")),
            ("Engine Creation",  std_dur.get("engine_creation"),  wc_dur.get("engine_creation")),
            ("API Server Startup", std_dur.get("api_server_startup"), wc_dur.get("api_server_startup")),
            ("First Inference",  std_dur.get("first_inference"),  wc_dur.get("first_inference")),
        ]
        rows.append(("── Durations ───────────", "", ""))
        for label, sv, wv in duration_rows:
            rows.append((label, _ts(sv), _ts(wv)))

        # --- Summary ---
        std_total = std.get("total_cold_start") if std else None
        wc_total = wc.get("total_cold_start") if wc else None
        rows.append(("── Summary ─────────────", "", ""))
        rows.append(("Total Cold Start", _ts(std_total), _ts(wc_total)))

        if is_finite_number(std_total) and is_finite_number(wc_total) and wc_total > 0:
            diff_s = float(std_total) - float(wc_total)
            spd = float(std_total) / float(wc_total)
            rows.append(("Speedup", "", f"{spd:.1f}x  (saved {diff_s:.1f}s)"))

        # --- Print table ---
        col_w = 22
        print(f"  {'Metric':<28} {'Std vLLM':>{col_w}} {'Worker Ctrl':>{col_w}}")
        print(f"  {'─' * 28} {'─' * col_w} {'─' * col_w}")
        for label, sv, wv in rows:
            if sv == "" and wv == "":
                # Section header
                print(f"  {label}")
            else:
                print(f"  {label:<28} {sv:>{col_w}} {wv:>{col_w}}")
        print()

    # ── Visual comparison ──────────────────────────────────────────────
    if std_cumulative > 0 and wc_cumulative > 0:
        max_time = max(std_cumulative, wc_cumulative)
        bar_width = 50
        std_bar = int(bar_width * std_cumulative / max_time)
        wc_bar = int(bar_width * wc_cumulative / max_time)

        print_banner("VISUAL COMPARISON")
        print(f"  Standard:  {'#' * std_bar}{'-' * (bar_width - std_bar)} {std_cumulative:.1f}s")
        print(f"  WC:        {'#' * wc_bar}{'-' * (bar_width - wc_bar)} {wc_cumulative:.1f}s")
        savings_pct = (1 - wc_cumulative / std_cumulative) * 100
        print(f"\n  Time saved: {diff:.1f}s ({savings_pct:.0f}% faster)")

    # ── Generate chart ─────────────────────────────────────────────────
    if _HAS_MATPLOTLIB:
        chart_path = _generate_chart(results, log_path)
    else:
        print("  [chart] matplotlib not available — skipping chart generation")
        chart_path = None

    # ── Cleanup ─────────────────────────────────────────────────────────
    stop_worker_controller()

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"[log] benchmark output saved: {log_path}")
    if chart_path:
        print(f"[log] chart saved: {chart_path}")


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

STD_COLOR = "#e74c3c"
WC_COLOR = "#3498db"


def _safe_avg(values: list) -> float:
    return average_finite(values, default=0.0)


def _generate_chart(
    results: dict[str, list[dict]],
    log_path: str,
) -> str | None:
    """Generate a comparison chart from benchmark results and save as PNG."""
    std_results = results.get("standard_vllm", [])
    wc_results = results.get("worker_controller", [])

    if not std_results and not wc_results:
        print("  [chart] No results to plot.")
        return None

    # --- Compute average durations ---
    def _avg_field(items, dur_key, field):
        vals = [r.get(dur_key, {}).get(field) for r in items]
        return _safe_avg(vals)

    std_model_load = _avg_field(std_results, "std_phase_durations", "model_loading")
    std_engine     = _avg_field(std_results, "std_phase_durations", "engine_creation")
    std_startup    = _avg_field(std_results, "std_phase_durations", "api_server_startup")
    std_inference  = _avg_field(std_results, "std_phase_durations", "first_inference")

    wc_model_load  = _avg_field(wc_results, "wc_durations", "model_loading")
    wc_engine      = _avg_field(wc_results, "wc_durations", "engine_creation")
    wc_startup     = _avg_field(wc_results, "wc_durations", "api_server_startup")
    wc_inference   = _avg_field(wc_results, "wc_durations", "first_inference")

    n_runs = max(len(std_results), len(wc_results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Cold Start: Standard vLLM vs Worker Controller\n"
        f"facebook/opt-125m  \u2022  1 GPU  \u2022  {n_runs} run(s)",
        fontsize=14, fontweight="bold",
    )

    # ── Left: Phase durations (averaged) ──
    phases = ["Model\nLoading", "Engine\nCreation", "API Server\nStartup", "First\nInference"]
    std_vals = [std_model_load, std_engine, std_startup, std_inference]
    wc_vals  = [wc_model_load,  wc_engine,  wc_startup,  wc_inference]

    x = np.arange(len(phases))
    w = 0.32

    bars_s = ax1.bar(x - w/2, std_vals, w, color=STD_COLOR, label="Std vLLM", alpha=0.85)
    bars_w = ax1.bar(x + w/2, wc_vals,  w, color=WC_COLOR,  label="Worker Controller", alpha=0.85)

    for bars in (bars_s, bars_w):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                         f"{h:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=10)
    ax1.set_ylabel("Seconds", fontsize=11)
    ax1.set_title("Phase Durations (avg)", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=10)

    # ── Right: Total cold start per run ──
    std_totals = [r["total_cold_start"] for r in std_results]
    wc_totals  = [r["total_cold_start"] for r in wc_results]
    max_runs = max(len(std_totals), len(wc_totals))
    # Pad shorter list with 0 so bars align
    std_totals += [0] * (max_runs - len(std_totals))
    wc_totals  += [0] * (max_runs - len(wc_totals))

    x2 = np.arange(max_runs)
    bars_s2 = ax2.bar(x2 - w/2, std_totals, w, color=STD_COLOR, label="Std vLLM", alpha=0.85)
    bars_w2 = ax2.bar(x2 + w/2, wc_totals,  w, color=WC_COLOR,  label="Worker Controller", alpha=0.85)

    for bars in (bars_s2, bars_w2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                         f"{h:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i in range(max_runs):
        s, wc = std_totals[i], wc_totals[i]
        if s > 0 and wc > 0:
            spd = s / wc
            ax2.text(x2[i], max(s, wc) + 1.0, f"{spd:.1f}x",
                     ha="center", fontsize=10, fontweight="bold",
                     color="#27ae60" if spd > 1 else "#e74c3c")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Run {i+1}" for i in range(max_runs)], fontsize=10)
    ax2.set_ylabel("Seconds", fontsize=11)
    ax2.set_title("Total Cold Start (per run)", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save next to the log file
    chart_path = log_path.replace(".txt", "_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  [chart] Saved: {chart_path}")
    return chart_path


if __name__ == "__main__":
    main()
