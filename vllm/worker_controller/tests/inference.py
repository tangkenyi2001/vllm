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

    with open("controller.log", "w") as log_file:
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
            "Controller started with PID %s. Logs in controller.log", controller_process.pid)

        base_url = "http://localhost:8000"
        engine_uuid = "test-engine-1"
        model_name = "Qwen/Qwen3-0.6B"  # Small model for testing

        try:
            # --- Wait for Controller ---
            wait_for_service(f"{base_url}/health", "Controller")

            # --- 1. Load Model ---
            logger.info("\n[1/5] Creating engine %s...", engine_uuid)
            create_payload = {
                "engine_uuid": engine_uuid,
                "model": model_name,
                "gpu_memory_utilization": 0.3,
                "tensor_parallel_size": 1,
            }
            logger.info("Sending create engine request: %s", create_payload)
            resp = requests.post(f"{base_url}/engines", json=create_payload)
            logger.info("Received create engine response: %s - %s",
                        resp.status_code, resp.text)
            if resp.status_code != 200:
                logger.error("FAILED to create engine: %s - %s",
                             resp.status_code, resp.text)
                sys.exit(1)

            engine_data = resp.json()
            api_url = engine_data["api_url"]
            logger.info("Engine created at %s", api_url)

            # Wait for engine API
            wait_for_service(f"{api_url}/health", "Engine")

            # --- 2. Inference ---
            logger.info("\n[2/5] Running inference...")
            run_inference(api_url, model_name)

            # --- 3. Unload Model ---
            logger.info("\n[3/5] Deleting engine %s...", engine_uuid)
            logger.info("Sending delete engine request for %s", engine_uuid)
            resp = requests.delete(f"{base_url}/engines/{engine_uuid}")
            logger.info("Received delete engine response: %s - %s",
                        resp.status_code, resp.text)
            if resp.status_code != 200:
                logger.error("FAILED to delete engine: %s - %s",
                             resp.status_code, resp.text)
                sys.exit(1)
            logger.info("Engine deleted successfully.")

            # Verify it's gone
            logger.info("Verifying engine %s is gone.", engine_uuid)
            resp = requests.get(f"{base_url}/engines/{engine_uuid}")
            if resp.status_code != 404:
                logger.warning(
                    "WARNING: Engine status is %s (expected 404)", resp.status_code)
            else:
                logger.info(
                    "Engine %s confirmed gone (status 404).", engine_uuid)

            # Short pause to ensure cleanup
            logger.info("Pausing for 2 seconds to ensure cleanup.")
            time.sleep(2)

            # --- 4. Load Model Again ---
            logger.info("\n[4/5] Re-creating engine %s...", engine_uuid)
            logger.info("Sending re-create engine request: %s", create_payload)
            resp = requests.post(f"{base_url}/engines", json=create_payload)
            logger.info("Received re-create engine response: %s - %s",
                        resp.status_code, resp.text)
            if resp.status_code != 200:
                logger.error("FAILED to re-create engine: %s - %s",
                             resp.status_code, resp.text)
                sys.exit(1)

            engine_data = resp.json()
            api_url = engine_data["api_url"]
            logger.info("Engine re-created at %s", api_url)

            # Wait for engine API
            wait_for_service(f"{api_url}/health", "Engine")

            # --- 5. Inference Again ---
            logger.info("\n[5/5] Running inference again...")
            run_inference(api_url, model_name)

            logger.info(
                "\n Load -> Inference -> Unload -> Load -> Inference successful!")

        except Exception as e:
            # logger.exception includes traceback
            logger.exception("TEST FAILED: %s", e)
            # traceback.print_exc() # No longer needed

        finally:
            logger.info("\nCleaning up...")
            try:
                os.killpg(os.getpgid(controller_process.pid), signal.SIGTERM)
                logger.info("Sent SIGTERM to process group %s",
                            os.getpgid(controller_process.pid))
            except Exception as e:
                logger.error("Error during cleanup: %s", e)


def wait_for_service(url, name, timeout=60):
    logger.info("Waiting for %s at %s...", name, url)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code == 200:
                logger.info("%s is ready.", name)
                return
            else:
                logger.info("Still waiting for %s. Status: %s",
                            name, resp.status_code)
        except requests.exceptions.ConnectionError:
            logger.debug("Still waiting for %s. Connection refused.", name)
        except Exception as e:
            logger.info("Still waiting for %s. Error: %s", name, e)
        time.sleep(1)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


def run_inference(url, model):
    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 5
    }
    logger.info("   Sending prompt: '%s' to %s/v1/completions",
                payload['prompt'], url)
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference error {resp.status_code}: {resp.text}")

        result = resp.json()
        text = result["choices"][0]["text"]
        logger.info("Response: '%s'", text.strip())
        logger.info("Full inference response: %s", result)

    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}") from e


if __name__ == "__main__":
    test_load_unload_loop()
