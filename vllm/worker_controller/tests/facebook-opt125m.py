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
import subprocess
import sys
import threading
import time
import requests
import uvicorn

import vllm.worker_controller.worker_controller_server as wc_server

# Import WorkerController
from vllm.worker_controller.worker_controller import WorkerController
from vllm.worker_controller.worker_controller_server import app
from vllm.worker_controller.tests.utils.benchmark_utils import (
    avg_metric as _avg_metric,
    avg_nested_metric as _avg_nested_metric,
    configure_debug_logging,
    format_seconds as _format_seconds,
    format_create_timings as _format_create_timings,
    parse_standard_startup_breakdown,
    parse_standard_startup_milestones,
    prewarm_model_files,
    print_aligned_startup_comparison,
    print_banner as _print_banner,
    print_engine_status,
    print_standard_startup_overhead,
    print_visual_comparison,
    print_worker_controller_startup_overhead,
    results_for_model as _results_for_model,
    save_time_savings_bar_chart,
    start_subprocess_log_collector,
    wc_log,
)

BASE_URL = "http://localhost:21000"

# Global reference to server thread
_server_thread = None
_server = None

# Test configuration
MODELS = [
    {"name": "facebook/opt-125m", "uuid": "opt-125m"}
]

TEST_PROMPT = "Hello, my name is"
RUNS_PER_MODEL = 2
PREWARM_MODEL_FILES = True
PREWARM_DOWNLOAD_IF_MISSING = True

MODEL_SHARD_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
)

NOISY_OUTPUT_PATTERNS = (
    "[Gloo] Rank",
)


def start_worker_controller():
    """Start the worker controller server in a background thread."""
    global _server_thread, _server

    wc_log("BOOT", "Starting Worker Controller")

    # Initialize the WorkerController
    wc_server.worker_controller = WorkerController(start_port=21002)
    wc_log(
        "BOOT",
        "Initialized controller with "
        f"{len(wc_server.worker_controller.executor.workers)} workers",
    )

    # Create uvicorn config
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=21000,
        log_level="warning",
        access_log=False,
    )
    _server = uvicorn.Server(config)

    # Run in background thread
    _server_thread = threading.Thread(target=_server.run, daemon=True)
    _server_thread.start()

    wc_log("BOOT", "API server started on port 21000")


def stop_worker_controller():
    """Stop the worker controller server."""
    global _server, _server_thread

    if _server is not None:
        wc_log("STOP", "Stopping Worker Controller")
        _server.should_exit = True
        if _server_thread is not None:
            _server_thread.join(timeout=5)
        wc_log("STOP", "Worker Controller stopped")


def wait_for_controller():
    """Wait for the worker controller to be ready."""
    wc_log("HEALTH", "Waiting for controller health endpoint")
    for i in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                wc_log("HEALTH", f"Controller ready: {resp.json()}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("Worker Controller did not become ready in time")


def measure_worker_controller_cold_start(model_name: str, engine_uuid: str):
    """
    Measure cold start time using Worker Controller.

    Cold start = time from API call to first token generation.
    With Worker Controller, CUDA context and distributed setup are already done.
    """
    _print_banner(f"Worker Controller Cold Start: {model_name}", width=60)

    total_start = time.time()

    wc_log("CREATE", f"Creating engine {engine_uuid}")
    create_start = time.time()
    create_payload = {
        "engine_uuid": engine_uuid,
        "model": model_name,
        "gpu_memory_utilization": 0.3,
        "enforce_eager": True,
    }
    wc_log("CREATE", f"payload={create_payload}")

    resp = requests.post(
        f"{BASE_URL}/engines",
        json=create_payload,
        timeout=300,
    )
    create_time = time.time() - create_start

    if resp.status_code != 200:
        wc_log("ERROR", f"create failed: {resp.text}")
        return None

    create_result = resp.json()
    create_timings = create_result.get("create_timings")
    attach_to_prewarmed_workers_s = None
    if isinstance(create_timings, dict):
        attach_parts = [
            create_timings.get("resource_allocation_time"),
            create_timings.get("ipc_queue_setup_time"),
            create_timings.get("proxy_register_time"),
        ]
        numeric_attach_parts = [
            float(v) for v in attach_parts if isinstance(v, (int, float))
        ]
        if numeric_attach_parts:
            attach_to_prewarmed_workers_s = sum(numeric_attach_parts)
    port = create_result["port"]
    wc_log(
        "CREATE",
        f"done in {create_time:.2f}s | status={create_result.get('status')} "
        f"pid={create_result.get('pid')} port={create_result.get('port')} "
        f"ranks={create_result.get('assigned_ranks')}",
    )
    wc_log("CREATE", f"internal timings: {_format_create_timings(create_timings)}")
    if isinstance(attach_to_prewarmed_workers_s, (int, float)):
        wc_log(
            "ATTACH",
            "attach to pre-warmed workers "
            f"took {attach_to_prewarmed_workers_s:.6f}s",
        )

    engine_url = f"http://localhost:{port}"
    wc_log("READY", f"Waiting for engine API health at {engine_url}")
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

        if i == 0 or (i + 1) % 10 == 0:
            print_engine_status(
                BASE_URL,
                engine_uuid,
                prefix=f"  [WC][READY][{i + 1:02d}] ",
                include_create_timings=(i == 0),
            )

        time.sleep(0.5)

    if not health_ready:
        wc_log("ERROR", "API server did not become healthy in time")
        print_engine_status(BASE_URL, engine_uuid, prefix="  [WC][READY][final] ")
        return None

    api_ready_time = time.time() - api_ready_start
    wc_log("READY", f"API server healthy in {api_ready_time:.2f}s")

    startup_timing = None
    routes_to_health = None
    try:
        startup_resp = requests.get(f"{engine_url}/startup_timing", timeout=3)
        if startup_resp.status_code == 200:
            startup_timing = startup_resp.json()
            wc_log("READY", f"startup_timing={startup_timing}")
            maybe = startup_timing.get("api_routes_to_first_health_s")
            if isinstance(maybe, (int, float)):
                routes_to_health = float(maybe)
    except Exception as e:
        wc_log("READY", f"startup timing fetch failed: {e}")

    model_load_timings = None
    model_load_summary = None

    wc_log("INFER", "Running first inference")
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

    if resp.status_code == 200:
        generated = resp.json()["choices"][0]["text"]
        wc_log("INFER", f"first token response in {first_inference_time:.2f}s")
        wc_log("INFER", f"generated={generated!r}")
    else:
        wc_log("ERROR", f"inference failed: {resp.text}")
        first_inference_time = float("inf")

    try:
        wc_log("LOAD", f"querying timing endpoint: {engine_url}/model_load_timings")
        timings_resp = requests.get(f"{engine_url}/model_load_timings", timeout=3)
        if timings_resp.status_code == 200:
            timings_data = timings_resp.json()
            model_load_timings = timings_data.get("worker_timings")
            model_load_summary = timings_data.get("summary")
            endpoint_debug = timings_data.get("debug")
            wc_log(
                "LOAD",
                "engine endpoint response | "
                f"has_worker_timings={bool(model_load_timings)} "
                f"has_summary={bool(model_load_summary)}",
            )
            if endpoint_debug and not model_load_timings:
                wc_log("LOAD", f"engine endpoint debug={endpoint_debug}")
        else:
            wc_log(
                "LOAD",
                "engine endpoint failed | "
                f"status={timings_resp.status_code} body={timings_resp.text[:300]!r}",
            )

        if not model_load_timings:
            wc_endpoint = f"{BASE_URL}/engines/{engine_uuid}/load_timings"
            wc_log("LOAD", f"fallback timing endpoint: {wc_endpoint}")
            ctrl_timings_resp = requests.get(wc_endpoint, timeout=5)
            if ctrl_timings_resp.status_code == 200:
                ctrl_data = ctrl_timings_resp.json()
                model_load_timings = ctrl_data.get("worker_timings")
                summary = ctrl_data.get("summary")
                ctrl_debug = ctrl_data.get("debug")
                if isinstance(summary, dict):
                    if not isinstance(model_load_summary, dict):
                        model_load_summary = {}
                    model_load_summary.update(summary)
                wc_log(
                    "LOAD",
                    "controller endpoint response | "
                    f"has_worker_timings={bool(model_load_timings)} "
                    f"has_summary={bool(summary)} "
                    f"message={ctrl_data.get('message')!r}",
                )
                if ctrl_debug and not model_load_timings:
                    wc_log("LOAD", f"controller endpoint debug={ctrl_debug}")
            else:
                wc_log(
                    "LOAD",
                    "controller endpoint failed | "
                    f"status={ctrl_timings_resp.status_code} "
                    f"body={ctrl_timings_resp.text[:300]!r}",
                )

        init_engine_time = 0.0
        if isinstance(model_load_summary, dict):
            init_engine_time = (
                model_load_summary.get("init_engine_time_seconds", 0.0) or 0.0
            )

        if init_engine_time > 0:
            wc_log(
                "KV",
                "init engine (profile, create kv cache, warmup model) "
                f"took {init_engine_time:.2f} seconds",
            )
        else:
            wc_log("KV", "init engine (profile, create kv cache, warmup model) took N/A")

        load_rpc_time = 0.0
        if isinstance(model_load_summary, dict):
            load_rpc_time = (
                model_load_summary.get("remote_executor_load_model_rpc_time", 0.0)
                or 0.0
            )
        if load_rpc_time > 0:
            wc_log("LOAD", f"remote executor load_model RPC took {load_rpc_time:.2f}s")

        if model_load_timings:
            t = model_load_timings[0]
            wc_log(
                "LOAD",
                "first worker timings | "
                f"config={t.get('config_time', 0):.3f}s "
                f"dist={t.get('dist_init_time', 0):.3f}s "
                f"runner={t.get('model_runner_init_time', 0):.3f}s "
                f"weight={t.get('weight_load_time', 0):.3f}s "
                f"total={t.get('total_time', 0):.3f}s",
            )

            worker_effective_loads = [
                wt.get(
                    "effective_model_load_time",
                    wt.get("model_runner_init_time", 0)
                    + wt.get("weight_load_time", 0),
                )
                for wt in model_load_timings
            ]
            worker_effective_loads = [
                float(v) for v in worker_effective_loads if isinstance(v, (int, float))
            ]
            if worker_effective_loads:
                avg_effective = sum(worker_effective_loads) / len(worker_effective_loads)
                wc_log(
                    "WARM-WORKERS",
                    "model load time across warmed workers | "
                    f"avg={avg_effective:.3f}s "
                    f"per_worker={[round(v, 3) for v in worker_effective_loads]}",
                )
        else:
            wc_log(
                "LOAD",
                "first worker timings unavailable (N/A) after engine+controller timing endpoint attempts",
            )
            print_engine_status(
                BASE_URL,
                engine_uuid,
                prefix="  [WC][LOAD][status] ",
                include_create_timings=True,
            )
    except Exception as e:
        wc_log("LOAD", f"could not fetch /model_load_timings: {e}")

    print_worker_controller_startup_overhead(
        create_time=create_time,
        api_ready_time=api_ready_time,
        create_timings=create_timings,
        model_load_summary=model_load_summary,
        model_load_timings=model_load_timings,
        startup_timing=startup_timing,
        routes_to_health=routes_to_health,
    )

    total_time = time.time() - total_start

    wc_log("CLEANUP", f"Deleting engine {engine_uuid}")
    requests.delete(f"{BASE_URL}/engines/{engine_uuid}", timeout=60)
    time.sleep(2)

    result = {
        "model": model_name,
        "create_time": create_time,
        "api_ready_time": api_ready_time,
        "first_inference_time": first_inference_time,
        "total_cold_start": total_time,
    }

    if model_load_timings:
        result["model_load_timings"] = model_load_timings
    if model_load_summary:
        result["model_load_summary"] = model_load_summary
    if create_timings:
        result["create_timings"] = create_timings
    if isinstance(attach_to_prewarmed_workers_s, (int, float)):
        result["wc_attach_to_prewarmed_workers_s"] = float(
            attach_to_prewarmed_workers_s
        )
    if startup_timing:
        result["wc_startup_timing"] = startup_timing
    if isinstance(routes_to_health, (int, float)):
        result["wc_api_routes_to_health_ready_s"] = float(routes_to_health)
        if isinstance(api_ready_time, (int, float)):
            result["wc_post_engine_to_api_routes_ready_s"] = max(
                0.0,
                float(api_ready_time) - float(routes_to_health),
            )

    return result

def measure_standard_vllm_cold_start(model_name: str, port: int = 8000):
    """
    Measure cold start time using standard vLLM API server (vllm serve).

    Cold start = time from process start to first token generation via API.
    This includes CUDA context creation, distributed setup, model loading,
    and API server startup - matching what Worker Controller does.
    """
    _print_banner(f"Standard vLLM API Server Cold Start: {model_name}", width=60)

    total_start = time.time()

    # Start vLLM serve in subprocess with startup logs enabled for breakdown
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env.pop("VLLM_CONFIGURE_LOGGING", None)
    env["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformers logs
    env["TQDM_DISABLE"] = "1"  # Disable progress bars
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
    print(f"Starting vLLM API server on port {port}...")
    spawn_begin = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=repo_root,
    )
    spawn_end = time.time()
    print(f"  API process spawn call time: {spawn_end - spawn_begin:.3f}s")
    std_log_lines: list[tuple[float, str]] = []
    start_subprocess_log_collector(
        proc,
        std_log_lines,
        prefix="  [std-log] ",
        capture_receive_ts=True,
    )

    # Wait for API server to be ready
    print("Waiting for API server to be ready...")
    api_url = f"http://localhost:{port}"
    server_ready = False
    server_ready_time = None

    for i in range(120):  # Wait up to 120 seconds
        if proc.poll() is not None:
            stderr_tail = "\n".join(line for _, line in std_log_lines[-20:])
            print("ERROR: API server process exited before becoming ready")
            if stderr_tail:
                print("Last stderr lines:")
                print(stderr_tail)
            return None

        try:
            resp = requests.get(f"{api_url}/health", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                ready_ts = time.time()
                server_ready_time = ready_ts - spawn_end
                print(f"  Server ready: {server_ready_time:.2f}s")
                break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(0.5)

    if not server_ready:
        print("ERROR: Server did not become ready in time")
        proc.terminate()
        proc.wait()
        return None

    std_startup_breakdown = parse_standard_startup_breakdown(std_log_lines)
    std_milestones = parse_standard_startup_milestones(std_log_lines)
    print_standard_startup_overhead(
        server_ready_time=server_ready_time,
        std_startup_breakdown=std_startup_breakdown,
        std_milestones=std_milestones,
    )

    # First inference via API
    print("Running first inference...")
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
            print(f"  First inference: {first_inference_time:.2f}s")
            print(f"  Generated: {generated!r}")
        else:
            print(f"  ERROR: {resp.status_code} - {resp.text}")
            first_inference_time = float("inf")
    except Exception as e:
        print(f"  ERROR: {e}")
        first_inference_time = float("inf")

    total_time = time.time() - total_start

    # Cleanup - terminate server
    print("Cleaning up server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    time.sleep(2)  # Wait for port to be released

    result = {
        "model": model_name,
        "server_ready_time": server_ready_time,
        "first_inference_time": first_inference_time,
        "total_cold_start": total_time,
    }
    result.update(std_startup_breakdown)
    result.update(std_milestones)
    return result


def main():
    configure_debug_logging(NOISY_OUTPUT_PATTERNS)

    if PREWARM_MODEL_FILES:
        _print_banner("0: Model File Prewarm")
        print()
        for model_info in MODELS:
            model_name = model_info["name"]
            print(f"[PREWARM] Starting: {model_name}")
            prewarm = prewarm_model_files(
                model_name,
                prewarm_download_if_missing=PREWARM_DOWNLOAD_IF_MISSING,
                model_shard_extensions=MODEL_SHARD_EXTENSIONS,
            )
            if prewarm.get("ok"):
                gb = prewarm["bytes"] / (1024**3)
                print(
                    f"[PREWARM] Done: {model_name} | files={prewarm['files']} "
                    f"bytes={gb:.2f} GiB time={prewarm['seconds']:.2f}s"
                )
            else:
                print(
                    f"[PREWARM] Skipped: {model_name} | error={prewarm.get('error')}"
                )

    results = {
        "worker_controller": [],
        "standard_vllm": [],
    }
    _print_banner("1: Worker Controller")
    print()

    # Start and wait for controller
    start_worker_controller()
    atexit.register(stop_worker_controller)
    wait_for_controller()
    for model_info in MODELS:
        model_name = model_info["name"]
        base_engine_uuid = model_info["uuid"]
        for run_idx in range(1, RUNS_PER_MODEL + 1):
            engine_uuid = f"{base_engine_uuid}-run{run_idx}"
            print(f"\n[Worker Controller] run {run_idx}/{RUNS_PER_MODEL} for {model_name}")
            result = measure_worker_controller_cold_start(model_name, engine_uuid)
            if result:
                result["run"] = run_idx
                results["worker_controller"].append(result)
            time.sleep(2)

    _print_banner("2: Standard vLLM")
    print()

    for model_index, model_info in enumerate(MODELS):
        model_name = model_info["name"]
        for run_idx in range(1, RUNS_PER_MODEL + 1):
            port = 8000 + model_index * 10 + run_idx - 1
            print(f"\n[Standard vLLM] run {run_idx}/{RUNS_PER_MODEL} for {model_name} (port {port})")
            result = measure_standard_vllm_cold_start(model_name, port=port)
            if result:
                result["run"] = run_idx
                results["standard_vllm"].append(result)
            time.sleep(3)  # Wait for GPU memory to clear
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    _print_banner("RESULTS")
    print(f"Averaged over {RUNS_PER_MODEL} runs per model")
    print(
        "\n{:<25} {:>15} {:>15} {:>12} {:>10} {:>8}".format(
            "Model", "Std vLLM (s)", "Worker Ctrl (s)", "Diff (s)", "Speedup", "Runs"
        )
    )
    print("-" * 80)

    for model_info in MODELS:
        model_name = model_info["name"]

        std_runs = _results_for_model(results["standard_vllm"], model_name)
        wc_runs = _results_for_model(results["worker_controller"], model_name)

        if std_runs and wc_runs:
            std_time = _avg_metric(std_runs, "total_cold_start")
            wc_time = _avg_metric(wc_runs, "total_cold_start")
            diff = std_time - wc_time
            speedup = std_time / wc_time if wc_time > 0 else float("inf")

            print(
                "{:<25} {:>15.2f} {:>15.2f} {:>12.2f} {:>9.1f}x {:>8}".format(
                    model_name[:25], std_time, wc_time, diff, speedup,
                    f"{min(len(std_runs), len(wc_runs))}/{RUNS_PER_MODEL}"
                )
            )

    # Cumulative totals (the key metric for sequential loading)
    _print_banner(
        "CUMULATIVE TIME FOR ALL {} MODELS x {} RUNS:".format(
            len(MODELS), RUNS_PER_MODEL
        )
    )

    std_cumulative = sum(r["total_cold_start"] for r in results["standard_vllm"])
    wc_cumulative = sum(r["total_cold_start"] for r in results["worker_controller"])
    cumulative_diff = std_cumulative - wc_cumulative
    cumulative_speedup = (
        std_cumulative / wc_cumulative if wc_cumulative > 0 else float("inf")
    )

    print(f"\n  Standard vLLM (fresh process each time):  {std_cumulative:.2f}s")
    print(f"  Worker Controller (reuses CUDA context):  {wc_cumulative:.2f}s")
    print(f"  Time saved:                               {cumulative_diff:.2f}s")
    print(f"  Speedup:                                  {cumulative_speedup:.2f}x")

    # Detailed breakdown
    print("\n" + "-" * 80)
    print("\nDetailed Breakdown:")

    for model_info in MODELS:
        model_name = model_info["name"]
        print(f"\n{model_name}:")

        std_runs = _results_for_model(results["standard_vllm"], model_name)
        wc_runs = _results_for_model(results["worker_controller"], model_name)

        std_result = std_runs[-1] if std_runs else None
        wc_result = wc_runs[-1] if wc_runs else None

        if std_result:
            print("  Standard vLLM (spawns new workers each time):")
            print(
                "    Server ready:    "
                f"{_format_seconds(std_result.get('server_ready_time'))}  "
                "(includes CUDA init + process spawn + model load)"
            )
            print(
                f"    First inference: {_format_seconds(std_result.get('first_inference_time'))}"
            )
            print(f"    Total:           {_format_seconds(std_result.get('total_cold_start'))}")

        if wc_result:
            print("  Worker Controller (reuses pre-warmed workers):")
            wc_server_ready = wc_result.get("create_time", 0) + wc_result.get(
                "api_ready_time", 0
            )
            print(
                f"    Server ready:    {wc_server_ready:.2f}s  (reuses CUDA context, only loads model)"
            )
            print(
                f"    First inference: {_format_seconds(wc_result.get('first_inference_time'))}"
            )
            print(f"    Total:           {_format_seconds(wc_result.get('total_cold_start'))}")
            if wc_result.get("model_load_summary"):
                load_summary = wc_result["model_load_summary"]
                print(
                    f"    load_model total(avg): {load_summary.get('avg_total_time', 0):.2f}s"
                )
                print(
                    f"    load_model weight(avg): {load_summary.get('avg_weight_load_time', 0):.2f}s"
                )

    # Side-by-side aligned metric comparison
    print_aligned_startup_comparison(results, MODELS)

    # Save image chart for sharing/reporting
    save_time_savings_bar_chart(results)

    # Visual bar chart comparison
    print_visual_comparison(std_cumulative, wc_cumulative)

    # Cleanup
    stop_worker_controller()


if __name__ == "__main__":
    main()
