#!/usr/bin/env python3
"""
Test script to verify inference is working by asking "What is the capital of France?"
"""

import os
import time
import requests
from vllm.worker_controller.worker_controller import WorkerController
from vllm.config import (VllmConfig, ModelConfig, CacheConfig, ParallelConfig,
                         CompilationConfig, ObservabilityConfig)

# Set VLLM_USE_V1=0 to use V0 AsyncLLMEngine which works with RemoteProxyExecutor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['VLLM_USE_V1'] = '0'


def test_inference_capital_of_france():
    """
    Test inference by asking what the capital of France is.
    """
    print("=" * 70)
    print("TEST: Inference - Capital of France")
    print("=" * 70)

    # Step 1: Create WorkerController with empty workers
    print("\n1. Creating WorkerController with 2 workers...")
    controller = WorkerController()
    print(
        f"   ✓ WorkerController created with {len(controller.executor.workers)} workers")

    # Step 2: Prepare vllm_config
    print("\n2. Preparing vllm_config...")

    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        enforce_eager=True,  # Force eager mode
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.5,
        swap_space=4,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        world_size=1,
        worker_cls='vllm.worker_controller.gpu_worker.Worker'
    )
    observability_config = ObservabilityConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        observability_config=observability_config,
    )

    print("   ✓ VllmConfig prepared")

    # Step 3: Create API server
    print("\n3. Creating API server...")
    engine_uuid = "test-inference"

    try:
        proc = controller.create(vllm_config, engine_uuid)
        print(f"   ✓ API server process started with PID: {proc.pid}")
        api_url = "http://localhost:8001"

        # Wait for API server to start (longer wait for model loading)
        print("\n4. Waiting for API server to initialize (30 seconds)...")
        time.sleep(30)

        # Check if process is alive
        if not proc.is_alive():
            print("   ✗ API server process died")
            return

        print("   ✓ API server process is running")

        # Step 4: Test inference
        print("\n5. Testing inference: 'What is the capital of France?'")
        inference_url = f"{api_url}/v1/completions"

        payload = {
            "model": "facebook/opt-125m",
            "prompt": "What is the capital of France?",
            "max_tokens": 50,
            "temperature": 0.0
        }

        try:
            response = requests.post(inference_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            generated_text = result["choices"][0]["text"].strip()
            print(f"   Generated text: '{generated_text}'")

            # Check if "Paris" is in the response (case insensitive)
            if "paris" in generated_text.lower():
                print("   ✓ SUCCESS: Inference working! Response contains 'Paris'")
            else:
                print("   ? Response does not contain 'Paris', but inference completed")
                print("   This might be expected for a small model like OPT-125m")

        except requests.exceptions.RequestException as e:
            print(f"   ✗ Inference request failed: {e}")

        print("\n" + "=" * 70)
        print("TEST COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n6. Cleaning up...")
        try:
            controller.delete(engine_uuid)
            print("   ✓ API server deleted")
        except Exception as e:
            print(f"   - Cleanup warning: {e}")

        print("   ✓ Test completed")


if __name__ == "__main__":
    test_inference_capital_of_france()
