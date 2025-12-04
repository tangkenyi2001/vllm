import subprocess
import time
import sys
import os
import requests
import signal
import logging
import json

# Set up basic logging for the test script
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def wait_for_service(url, name, timeout=180):
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
        time.sleep(2)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


def run_inference(url, model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 15
    }
    logger.info("   Sending prompt to %s: '%s'", url, payload['prompt'])
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference error {resp.status_code}: {resp.text}")

        result = resp.json()
        text = result["choices"][0]["text"]
        logger.info("Response from %s: '%s'", model, text.strip())
        logger.info("Full inference response: %s", result)

    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}") from e


def test_parallel_load():
    logger.info("Starting Parallel Worker Controller Test...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open("controller_parallel.log", "w") as log_file:
        controller_process = subprocess.Popen(
            [sys.executable, "vllm/worker_controller/main.py"],
            cwd=os.getcwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
            env=env
        )

        logger.info(
            "Controller started with PID %s. Logs in controller_parallel.log", controller_process.pid)

        base_url = "http://localhost:8000"

        # Define engines
        # Engine 1: Facebook OPT
        engine1_uuid = "engine-opt"
        model1_name = "facebook/opt-125m"

        engine2_uuid = "engine-qwen"
        model2_name = "Qwen/Qwen3-0.6B"

        try:
            # --- Wait for Controller ---
            wait_for_service(f"{base_url}/health", "Controller")

            # --- 1. Create Engine 1 ---
            logger.info("\n[1/4] Creating Engine 1: %s...", model1_name)
            create_payload1 = {
                "engine_uuid": engine1_uuid,
                "model": model1_name,
                "gpu_memory_utilization": 0.3,
                "tensor_parallel_size": 1,
            }
            resp1 = requests.post(f"{base_url}/engines", json=create_payload1)
            if resp1.status_code != 200:
                logger.error("FAILED to create Engine 1: %s", resp1.text)
                sys.exit(1)

            data1 = resp1.json()
            api_url1 = data1["api_url"]
            logger.info("Engine 1 (%s) created at %s", model1_name, api_url1)

            # --- 2. Create Engine 2 ---
            logger.info("\n[2/4] Creating Engine 2: %s...", model2_name)
            create_payload2 = {
                "engine_uuid": engine2_uuid,
                "model": model2_name,
                "gpu_memory_utilization": 0.3,
                "tensor_parallel_size": 1,
            }
            resp2 = requests.post(f"{base_url}/engines", json=create_payload2)
            if resp2.status_code != 200:
                logger.error("FAILED to create Engine 2: %s", resp2.text)
                sys.exit(1)

            data2 = resp2.json()
            api_url2 = data2["api_url"]
            logger.info("Engine 2 (%s) created at %s", model2_name, api_url2)

            # --- 3. Wait for both engines ---
            logger.info("\n[3/4] Waiting for engines to be ready...")
            # We wait for them to be fully up
            wait_for_service(f"{api_url1}/health", f"Engine 1 ({model1_name})")
            wait_for_service(f"{api_url2}/health", f"Engine 2 ({model2_name})")

            # --- 4. Run Parallel Inference ---
            logger.info("\n[4/4] Running inference on both engines...")

            run_inference(api_url1, model1_name, "The capital of France is")
            run_inference(api_url2, model2_name, "The capital of France is")

            logger.info("\nTest successful!")

        except Exception as e:
            logger.exception("TEST FAILED: %s", e)

        finally:
            logger.info("\nCleaning up...")
            try:
                # Try to delete engines gracefully first
                requests.delete(f"{base_url}/engines/{engine1_uuid}")
                requests.delete(f"{base_url}/engines/{engine2_uuid}")
            except Exception:
                pass

            try:
                os.killpg(os.getpgid(controller_process.pid), signal.SIGTERM)
                logger.info("Sent SIGTERM to process group %s",
                            os.getpgid(controller_process.pid))
            except Exception as e:
                logger.error("Error during process cleanup: %s", e)


if __name__ == "__main__":
    test_parallel_load()
