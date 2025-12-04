import subprocess
import time
import sys
import os
import requests
import signal
import logging
import json

# Set up basic logging
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
            logger.debug(f"Still waiting for {name}. Error: {e}")
        time.sleep(2)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


def run_inference(url, model, prompt, name):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 15
    }
    logger.info(f"   [{name}] Sending prompt to {url}: '{payload['prompt']}'")
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference error {resp.status_code}: {resp.text}")

        result = resp.json()
        text = result["choices"][0]["text"]
        logger.info(f"   [{name}] Response: '{text.strip()}'")
        return text
    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}")


def test_dynamic_allocation():
    logger.info(
        "Starting Dynamic Allocation Test (Load -> Load -> Unload -> Load)...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open("controller_dynamic.log", "w")

    controller_process = subprocess.Popen(
        [sys.executable, "vllm/worker_controller/main.py"],
        cwd=os.getcwd(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=env
    )

    logger.info(f"Controller started with PID {controller_process.pid}")

    base_url = "http://localhost:8000"

    # Models
    model_opt = "facebook/opt-125m"
    model_qwen = "Qwen/Qwen3-0.6B"

    uuid_opt = "engine-opt"
    uuid_qwen_1 = "engine-qwen-1"
    uuid_qwen_2 = "engine-qwen-2"

    try:
        wait_for_service(f"{base_url}/health", "Controller")

        # --- Step 1: Load OPT (Engine 1) ---
        logger.info(f"\n[1/6] Loading Engine 1 (OPT)...")
        resp1 = requests.post(f"{base_url}/engines", json={"engine_uuid": uuid_opt,
                              "model": model_opt, "gpu_memory_utilization": 0.3, "tensor_parallel_size": 1})
        if resp1.status_code != 200:
            raise RuntimeError(f"Failed to create OPT: {resp1.text}")
        url_opt = resp1.json()["api_url"]
        port_opt = resp1.json()["port"]
        logger.info(f"Engine 1 (OPT) created at {url_opt} (Port: {port_opt})")

        # --- Step 2: Load Qwen 1 (Engine 2) ---
        logger.info(f"\n[2/6] Loading Engine 2 (Qwen #1)...")
        resp2 = requests.post(f"{base_url}/engines", json={"engine_uuid": uuid_qwen_1,
                              "model": model_qwen, "gpu_memory_utilization": 0.3, "tensor_parallel_size": 1})
        if resp2.status_code != 200:
            raise RuntimeError(f"Failed to create Qwen #1: {resp2.text}")
        url_qwen1 = resp2.json()["api_url"]
        port_qwen1 = resp2.json()["port"]
        logger.info(
            f"Engine 2 (Qwen #1) created at {url_qwen1} (Port: {port_qwen1})")

        # --- Step 3: Verify Both ---
        logger.info(f"\n[3/6] Waiting for readiness and testing inference...")
        wait_for_service(f"{url_opt}/health", "Engine 1 (OPT)")
        wait_for_service(f"{url_qwen1}/health", "Engine 2 (Qwen #1)")

        run_inference(url_opt, model_opt,
                      "Tell me the capital of France", "facebook/opt-125m")
        run_inference(url_qwen1, model_qwen,
                      "Tell me the capital of China", "Qwen/Qwen3-0.6B #1")

        # --- Step 4: Unload OPT ---
        logger.info(f"\n[4/6] Unloading Engine 1 (OPT) to free resources...")
        del_resp = requests.delete(f"{base_url}/engines/{uuid_opt}")
        if del_resp.status_code != 200:
            raise RuntimeError(f"Failed to delete OPT: {del_resp.text}")
        logger.info("Engine 1 (OPT) deleted.")
        time.sleep(2)  # Ensure cleanup

        # --- Step 5: Load Qwen 2 (Engine 3) ---
        # Ideally this takes the place/resources of Engine 1
        logger.info(f"\n[5/6] Loading Engine 3 (Qwen #2) into freed slot...")
        resp3 = requests.post(f"{base_url}/engines", json={"engine_uuid": uuid_qwen_2,
                              "model": model_qwen, "gpu_memory_utilization": 0.3, "tensor_parallel_size": 1})
        if resp3.status_code != 200:
            raise RuntimeError(f"Failed to create Qwen #2: {resp3.text}")
        url_qwen2 = resp3.json()["api_url"]
        port_qwen2 = resp3.json()["port"]
        logger.info(
            f"Engine 3 (Qwen #2) created at {url_qwen2} (Port: {port_qwen2})")

        if port_qwen2 == port_opt:
            logger.info("SUCCESS: Port recycled!")
        else:
            logger.info(
                f"NOTICE: New port assigned (Expected behavior in simple allocator). Old: {port_opt}, New: {port_qwen2}")

        # --- Step 6: Verify Qwen 1 and Qwen 2 ---
        logger.info(
            f"\n[6/6] Verifying Engine 2 (Retained) and Engine 3 (New)...")
        wait_for_service(f"{url_qwen2}/health", "Engine 3 (Qwen #2)")

        # Test Qwen 1 again (should still be alive)
        run_inference(url_qwen1, model_qwen,
                      "Tell me the capital of Spain", "Qwen/Qwen3-0.6B #1")
        # Test Qwen 2
        run_inference(url_qwen2, model_qwen,
                      "Tell me the capital of Japan", "Qwen/Qwen3-0.6B#2")

        logger.info("\nDynamic Allocation Test successful!")

    except Exception as e:
        logger.exception(f"TEST FAILED: {e}")

    finally:
        logger.info("\nCleaning up...")
        for uuid in [uuid_opt, uuid_qwen_1, uuid_qwen_2]:
            try:
                requests.delete(f"{base_url}/engines/{uuid}")
            except:
                pass

        try:
            os.killpg(os.getpgid(controller_process.pid), signal.SIGTERM)
        except Exception as e:
            logger.info(f"Cleanup error: {e}")
        log_file.close()


if __name__ == "__main__":
    test_dynamic_allocation()
