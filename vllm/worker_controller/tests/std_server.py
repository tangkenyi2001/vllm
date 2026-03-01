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
):
    """
    Measure cold start time using standard vLLM API server (vllm serve).

    Cold start = time from process start to first token generation via API.
    This includes CUDA context creation, distributed setup, model loading,
    and API server startup - matching what Worker Controller does.
    """

    total_start = time.time()

    env = os.environ.copy()
    # Pass the wall-clock start time so all child processes (api_server,
    # EngineCore, etc.) can log elapsed time relative to this anchor.
    env["VLLM_START_TIME"] = str(total_start)

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
        "--tensor-parallel-size",
        "2"
    ]

    print(f"[std_server] Starting vLLM API server for model '{model_name}' on port {port}...")
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
    measure_standard_vllm_cold_start("facebook/opt-125m")


if __name__ == "__main__":
    main()
    