#!/usr/bin/env python3
"""
Test script to verify API server can use RemoteProxyExecutor
to communicate with WorkerController's existing workers.
"""

from vllm.worker_controller.worker_controller import WorkerController
from vllm.config import (VllmConfig, ModelConfig, CacheConfig, ParallelConfig,
                         ObservabilityConfig)
import os
import time

# Set VLLM_USE_V1=0 to use V0 AsyncLLMEngine which works with RemoteProxyExecutor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ... rest of your imports ...
os.environ['VLLM_USE_V1'] = '0'


def test_api_server_with_remote_proxy_executor():
    """
    Test creating an API server that uses RemoteProxyExecutor
    to connect to WorkerController's workers.
    """
    print("=" * 70)
    print("TEST: API Server with RemoteProxyExecutor")
    print("=" * 70)

    # Step 1: Create WorkerController with empty workers
    print("\n1. Creating WorkerController with 2 workers...")
    controller = WorkerController()
    print(
        f"   ✓ WorkerController created with {len(controller.executor.workers)} workers")

    # Step 2: Prepare vllm_config for API server
    print("\n2. Preparing vllm_config for API server...")

    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.5,
        swap_space=4,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        world_size=2,
        worker_cls='vllm.worker_controller.gpu_worker.Worker'
    )
    observability_config = ObservabilityConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        observability_config=observability_config,
    )

    print("   ✓ VllmConfig prepared for facebook/opt-125m")

    # Step 3: Create API server using the workers
    print("\n3. Creating API server with RemoteProxyExecutor...")
    engine_uuid = "test-engine-001"

    try:
        proc = controller.create(vllm_config, engine_uuid)
        print(f"   ✓ API server process started with PID: {proc.pid}")
        print(f"   ✓ API server should be available at: http://localhost:8001")

        # Wait a bit for API server to start
        print("\n4. Waiting for API server to initialize (10 seconds)...")
        time.sleep(10)

        # Check if process is still alive
        if proc.is_alive():
            print("   ✓ API server process is running")
            print("\n" + "=" * 70)
            print("SUCCESS: API Server created with RemoteProxyExecutor!")
            print("=" * 70)
            print("\nYou can now test the API server:")
            print("  curl http://localhost:8001/v1/models")
            print("\nPress Ctrl+C to stop the test and clean up...")

            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nStopping test...")
        else:
            print("   ✗ API server process died unexpectedly")
            print("   Check logs for errors")

    except Exception as e:
        print(f"   ✗ Error creating API server: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n5. Cleaning up...")
        try:
            controller.delete(engine_uuid)
            print("   ✓ API server deleted")
        except Exception as e:
            print(f"   - Cleanup warning: {e}")

        print("   ✓ Test completed")


if __name__ == "__main__":
    test_api_server_with_remote_proxy_executor()
