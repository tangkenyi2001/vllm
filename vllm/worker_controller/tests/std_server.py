

import os
import time
import sys
import subprocess
import threading
import urllib.request
import urllib.error


def measure_standard_vllm_cold_start(
    model_name: str,
    port: int = 8000,
    result_label: str | None = None,
    timeout: int = 300,
    kv_cache_memory_bytes: int | None = None,
    use_fixed_kv_cache_memory: bool = False,
    load_format: str = "auto",
):
    """
    Measure cold start time using standard vLLM API server (vllm serve).

    Cold start = time from process start to first token generation via API.
    This includes CUDA context creation, distributed setup, model loading,
    and API server startup - matching what Worker Controller does.
    """

    total_start = time.time()

    env = os.environ.copy()
    env["HF_HOME"] = "/dev/shm/models"
    # Pass the wall-clock start time so all child processes (api_server,
    # EngineCore, etc.) can log elapsed time relative to this anchor.
    env["VLLM_START_TIME"] = str(total_start)
    # Skip kernel warmup (FlashInfer autotune on Hopper, deep GEMM, etc.)
    # to measure pure model-loading startup time without warmup overhead.
    env.setdefault("VLLM_SKIP_KERNEL_WARMUP", "1")

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    tensor_parallel_size = 2

    cmd = [
        sys.executable,
        "-m",
        "vllm.worker_controller.entrypoint.standard_api_server",
        "--model",
        model_name,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.8",
        "--enforce-eager",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--load-format",
        load_format,
        "--num-gpu-blocks-override",
        "1",
    ]

    if use_fixed_kv_cache_memory and kv_cache_memory_bytes is None:
        kv_cache_memory_bytes = 16 * 1024**3

    if kv_cache_memory_bytes is not None and "VLLM_KVC_MEM_GB" not in env:
        kv_cache_memory_gib = kv_cache_memory_bytes / (1024**3)
        env["VLLM_KVC_MEM_GB"] = ",".join(
            [f"{kv_cache_memory_gib:g}"] * tensor_parallel_size)

    if kv_cache_memory_bytes is not None:
        cmd.extend([
            "--kv-cache-memory-bytes",
            str(kv_cache_memory_bytes),
        ])

    print(f"[std_server] Starting vLLM API server for model '{model_name}' on port {port}...")
    print(f"[std_server] HF_HOME={env['HF_HOME']}")
    print(f"[std_server] load_format={load_format}")
    if "VLLM_KVC_MEM_GB" in env:
        print(f"[std_server] VLLM_KVC_MEM_GB={env['VLLM_KVC_MEM_GB']}")
    if kv_cache_memory_bytes is not None:
        print(f"[std_server] KV cache memory override={kv_cache_memory_bytes} bytes")
    print(f"[std_server] Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=repo_root,
    )

    # Stream subprocess output to stdout in real time
    def _stream_output():
        for line in proc.stdout:
            print(line, end="", flush=True)

    stream_thread = threading.Thread(target=_stream_output, daemon=True)
    stream_thread.start()

    health_url = f"http://localhost:{port}/health"
    server_ready = False
    deadline = time.time() + timeout

    print(f"[std_server] Waiting for server to become ready (timeout={timeout}s)...")

    while time.time() < deadline:
        # Check if the process died early
        ret = proc.poll()
        if ret is not None:
            stream_thread.join(timeout=5)
            print(f"[std_server] Server process exited early with code {ret}.")
            return None

        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    server_ready = True
                    break
        except Exception:
            pass

        time.sleep(2)

    if not server_ready:
        proc.terminate()
        stream_thread.join(timeout=5)
        print(f"[std_server] Server did not become ready within {timeout}s.")
        return None

    startup_time = time.time() - total_start
    label = result_label or model_name
    print(f"[std_server] Server ready. Startup time for '{label}': {startup_time:.2f}s")

    # Keep the server running until the user interrupts
    print(f"[std_server] Server is running at http://localhost:{port}. Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[std_server] Shutting down server...")
        proc.terminate()
        proc.wait()
        print("[std_server] Server stopped.")

    return startup_time


def main():
    measure_standard_vllm_cold_start(
        "Qwen/Qwen3-8B",
        use_fixed_kv_cache_memory=True,
        load_format="fastsafetensors",
    )


if __name__ == "__main__":
    main()
    