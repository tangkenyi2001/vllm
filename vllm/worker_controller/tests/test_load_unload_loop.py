import subprocess
import time
import sys
import os
import requests
import signal


def test_load_unload_loop():
    print("Starting Worker Controller Test Loop...")

    # Start the controller
    # We use setsid to create a new process group so we can kill the whole tree later
    # Use unbuffered output for python
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open("controller.log", "w")

    controller_process = subprocess.Popen(
        [sys.executable, "vllm/worker_controller/main.py"],
        cwd=os.getcwd(),
        stdout=log_file,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        preexec_fn=os.setsid,
        env=env
    )

    print(
        f"Controller started with PID {controller_process.pid}. Logs in controller.log")

    base_url = "http://localhost:8000"
    engine_uuid = "test-engine-1"
    model_name = "facebook/opt-125m"  # Small model for testing

    try:
        # --- Wait for Controller ---
        wait_for_service(f"{base_url}/health", "Controller")

        # --- 1. Load Model ---
        print(f"\n[1/5] Creating engine {engine_uuid}...")
        create_payload = {
            "engine_uuid": engine_uuid,
            "model": model_name,
            "gpu_memory_utilization": 0.3,
            "tensor_parallel_size": 1
        }
        resp = requests.post(f"{base_url}/engines", json=create_payload)
        if resp.status_code != 200:
            print(f"FAILED to create engine: {resp.text}")
            sys.exit(1)

        engine_data = resp.json()
        api_url = engine_data["api_url"]
        print(f"Engine created at {api_url}")

        # Wait for engine API
        wait_for_service(f"{api_url}/health", "Engine")

        # --- 2. Inference ---
        print("\n[2/5] Running inference...")
        run_inference(api_url, model_name)

        # --- 3. Unload Model ---
        print(f"\n[3/5] Deleting engine {engine_uuid}...")
        resp = requests.delete(f"{base_url}/engines/{engine_uuid}")
        if resp.status_code != 200:
            print(f"FAILED to delete engine: {resp.text}")
            sys.exit(1)
        print("Engine deleted successfully.")

        # Verify it's gone
        resp = requests.get(f"{base_url}/engines/{engine_uuid}")
        if resp.status_code != 404:
            print(
                f"WARNING: Engine status is {resp.status_code} (expected 404)")

        # Short pause to ensure cleanup
        time.sleep(2)

        # --- 4. Load Model Again ---
        print(f"\n[4/5] Re-creating engine {engine_uuid}...")
        resp = requests.post(f"{base_url}/engines", json=create_payload)
        if resp.status_code != 200:
            print(f"FAILED to re-create engine: {resp.text}")
            sys.exit(1)

        engine_data = resp.json()
        api_url = engine_data["api_url"]
        print(f"Engine re-created at {api_url}")

        # Wait for engine API
        wait_for_service(f"{api_url}/health", "Engine")

        # --- 5. Inference Again ---
        print("\n[5/5] Running inference again...")
        run_inference(api_url, model_name)

        print("\n Load -> Inference -> Unload -> Load -> Inference successful!")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        try:
            os.killpg(os.getpgid(controller_process.pid), signal.SIGTERM)
        except:
            pass
        log_file.close()


def wait_for_service(url, name, timeout=60):
    print(f"Waiting for {name} at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code == 200:
                print(f"{name} is ready.")
                return
        except:
            pass
        time.sleep(1)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


def run_inference(url, model):
    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 5
    }
    print(f"   Sending prompt: '{payload['prompt']}'")
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference error {resp.status_code}: {resp.text}")

        result = resp.json()
        text = result["choices"][0]["text"]
        print(f"Response: '{text.strip()}'")

    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}")


if __name__ == "__main__":
    test_load_unload_loop()
