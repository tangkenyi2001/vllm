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
        time.sleep(2)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


def run_inference(url, model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 15
    }
    logger.info(f"   Sending prompt to {url}: '{payload['prompt']}'")
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference error {resp.status_code}: {resp.text}")

        result = resp.json()
        text = result["choices"][0]["text"]
        logger.info(f"Response from {model}: '{text.strip()}'")
        logger.info(f"Full inference response: {result}")

    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}")


def test_parallel_load():
    logger.info("Starting Parallel Worker Controller Test...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open("controller_parallel.log", "w")

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
        f"Controller started with PID {controller_process.pid}. Logs in controller_parallel.log")

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
        logger.info(f"\n[1/4] Creating Engine 1: {model1_name}...")
        create_payload1 = {
            "engine_uuid": engine1_uuid,
            "model": model1_name,
            "gpu_memory_utilization": 0.3,
            "tensor_parallel_size": 1,
        }
        resp1 = requests.post(f"{base_url}/engines", json=create_payload1)
        if resp1.status_code != 200:
            logger.error(f"FAILED to create Engine 1: {resp1.text}")
            sys.exit(1)

        data1 = resp1.json()
        api_url1 = data1["api_url"]
        logger.info(f"Engine 1 ({model1_name}) created at {api_url1}")

        # --- 2. Create Engine 2 ---
        logger.info(f"\n[2/4] Creating Engine 2: {model2_name}...")
        create_payload2 = {
            "engine_uuid": engine2_uuid,
            "model": model2_name,
            "gpu_memory_utilization": 0.3,
            "tensor_parallel_size": 1,
        }
        resp2 = requests.post(f"{base_url}/engines", json=create_payload2)
        if resp2.status_code != 200:
            logger.error(f"FAILED to create Engine 2: {resp2.text}")
            sys.exit(1)

        data2 = resp2.json()
        api_url2 = data2["api_url"]
        logger.info(f"Engine 2 ({model2_name}) created at {api_url2}")

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
        logger.exception(f"TEST FAILED: {e}")

    finally:
        logger.info("\nCleaning up...")
        try:
            # Try to delete engines gracefully first
            requests.delete(f"{base_url}/engines/{engine1_uuid}")
            requests.delete(f"{base_url}/engines/{engine2_uuid}")
        except:
            pass

        try:
            os.killpg(os.getpgid(controller_process.pid), signal.SIGTERM)
            logger.info(
                f"Sent SIGTERM to process group {os.getpgid(controller_process.pid)}")
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")
        log_file.close()


if __name__ == "__main__":
    test_parallel_load()
