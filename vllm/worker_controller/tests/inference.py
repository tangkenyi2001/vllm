import subprocess
import time
import sys
import os
import requests
import signal
import logging

# Set up basic logging for the test script
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_load_unload_loop():
    logger.info("Starting Worker Controller Test Loop...")

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

    logger.info(
        f"Controller started with PID {controller_process.pid}. Logs in controller.log")

    base_url = "http://localhost:8000"
    engine_uuid = "test-engine-1"
    model_name = "Qwen/Qwen3-0.6B"  # Small model for testing

    try:
        # --- Wait for Controller ---
        wait_for_service(f"{base_url}/health", "Controller")

        # --- 1. Load Model ---
        logger.info(f"\n[1/5] Creating engine {engine_uuid}...")
        create_payload = {
            "engine_uuid": engine_uuid,
            "model": model_name,
            "gpu_memory_utilization": 0.3,
            "tensor_parallel_size": 1
        }
        logger.info(f"Sending create engine request: {create_payload}")
        resp = requests.post(f"{base_url}/engines", json=create_payload)
        logger.info(
            f"Received create engine response: {resp.status_code} - {resp.text}")
        if resp.status_code != 200:
            logger.error(
                f"FAILED to create engine: {resp.status_code} - {resp.text}")
            sys.exit(1)

        engine_data = resp.json()
        api_url = engine_data["api_url"]
        logger.info(f"Engine created at {api_url}")

        # Wait for engine API
        wait_for_service(f"{api_url}/health", "Engine")

        # --- 2. Inference ---
        logger.info("\n[2/5] Running inference...")
        run_inference(api_url, model_name)

        # --- 3. Unload Model ---
        logger.info(f"\n[3/5] Deleting engine {engine_uuid}...")
        logger.info(f"Sending delete engine request for {engine_uuid}")
        resp = requests.delete(f"{base_url}/engines/{engine_uuid}")
        logger.info(
            f"Received delete engine response: {resp.status_code} - {resp.text}")
        if resp.status_code != 200:
            logger.error(
                f"FAILED to delete engine: {resp.status_code} - {resp.text}")
            sys.exit(1)
        logger.info("Engine deleted successfully.")

        # Verify it's gone
        logger.info(f"Verifying engine {engine_uuid} is gone.")
        resp = requests.get(f"{base_url}/engines/{engine_uuid}")
        if resp.status_code != 404:
            logger.warning(
                f"WARNING: Engine status is {resp.status_code} (expected 404)")
        else:
            logger.info(f"Engine {engine_uuid} confirmed gone (status 404).")

        # Short pause to ensure cleanup
        logger.info("Pausing for 2 seconds to ensure cleanup.")
        time.sleep(2)

        # --- 4. Load Model Again ---
        logger.info(f"\n[4/5] Re-creating engine {engine_uuid}...")
        logger.info(f"Sending re-create engine request: {create_payload}")
        resp = requests.post(f"{base_url}/engines", json=create_payload)
        logger.info(
            f"Received re-create engine response: {resp.status_code} - {resp.text}")
        if resp.status_code != 200:
            logger.error(
                f"FAILED to re-create engine: {resp.status_code} - {resp.text}")
            sys.exit(1)

        engine_data = resp.json()
        api_url = engine_data["api_url"]
        logger.info(f"Engine re-created at {api_url}")

        # Wait for engine API
        wait_for_service(f"{api_url}/health", "Engine")

        # --- 5. Inference Again ---
        logger.info("\n[5/5] Running inference again...")
        run_inference(api_url, model_name)

        logger.info(
            "\n Load -> Inference -> Unload -> Load -> Inference successful!")

    except Exception as e:
        # logger.exception includes traceback
        logger.exception(f"TEST FAILED: {e}")
        # traceback.print_exc() # No longer needed

    finally:
        logger.info("\nCleaning up...")
        try:
            os.killpg(os.getpgid(controller_process.pid), signal.SIGTERM)
            logger.info(
                f"Sent SIGTERM to process group {os.getpgid(controller_process.pid)}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        log_file.close()


def wait_for_service(url, name, timeout=60):
    logger.info(f"Waiting for {name} at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code == 200:
                logger.info(f"{name} is ready.")
                return
            else:
                logger.info(
                    f"Still waiting for {name}. Status: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Still waiting for {name}. Connection refused.")
        except Exception as e:
            logger.info(f"Still waiting for {name}. Error: {e}")
        time.sleep(1)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


def run_inference(url, model):
    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 5
    }
    logger.info(
        f"   Sending prompt: '{payload['prompt']}' to {url}/v1/completions")
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference error {resp.status_code}: {resp.text}")

        result = resp.json()
        text = result["choices"][0]["text"]
        logger.info(f"Response: '{text.strip()}'")
        logger.info(f"Full inference response: {result}")

    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}")


if __name__ == "__main__":
    test_load_unload_loop()
