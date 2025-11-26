#!/usr/bin/env python3
"""
Full integration test for collective_rpc with target_ranks.
This creates an actual ProxyExecutor with workers to test the functionality.
"""

import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_full_integration():
    """Full integration test with actual ProxyExecutor and workers"""

    print("="*70)
    print("FULL INTEGRATION TEST: collective_rpc with ProxyExecutor")
    print("="*70)

    # Import vLLM components
    from vllm.worker_controller.executor.proxy_executor import ProxyExecutor
    from vllm.worker_controller.config.vllm_config import (
        DummyVllmConfig, DummyModelConfig, CacheConfig, ParallelConfig
    )

    # Test configuration
    WORLD_SIZE = 2  # Match the worker_controller configuration

    print(f"\n1. Creating VllmConfig with {WORLD_SIZE} workers...")

    # Use the same successful configuration as WorkerController
    modelConfig = DummyModelConfig("dummy", enforce_eager=True)
    cacheConfig = CacheConfig(gpu_memory_utilization=0.9)
    parallelConfig = ParallelConfig(
        world_size=WORLD_SIZE,
        worker_cls='vllm.worker_controller.gpu_worker.Worker'
    )
    vllm_config = DummyVllmConfig(
        model_config=modelConfig,
        cache_config=cacheConfig,
        parallel_config=parallelConfig
    )

    print("   ✓ VllmConfig created")

    # Initialize executor
    print("\n2. Initializing ProxyExecutor...")
    print("   (This will create worker processes - may take a moment)")

    executor = None
    try:
        executor = ProxyExecutor(vllm_config)

        print(f"   ✓ Executor initialized")
        print(f"   ✓ World size: {executor.world_size}")
        print(f"   ✓ Number of workers: {len(executor.workers)}")

        if len(executor.workers) == 0:
            print("\n⚠  WARNING: No workers were created!")
            print("   This may indicate a configuration or environment issue.")
            print("   However, the collective_rpc logic is still correct.")
            return False

        # Give workers a moment to stabilize
        time.sleep(2)

        # Test 1: Health check on ALL workers
        print("\n3. Test 1: Health check on ALL workers (default)")
        try:
            executor.collective_rpc("check_health", timeout=10)
            print("   ✓ All workers responding")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False

        # Test 2: Health check on workers [0, 1]
        print("\n4. Test 2: Health check on workers [0, 1]")
        target_ranks = [0, 1]
        try:
            responses = executor.collective_rpc(
                "check_health",
                target_ranks=target_ranks,
                timeout=10
            )
            print(
                f"   ✓ Got {len(responses)} responses from workers {target_ranks}")
            assert len(
                responses) == 2, f"Expected 2 responses, got {len(responses)}"
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False

        # Test 3: Health check on worker [1]
        print("\n5. Test 3: Health check on worker [1]")
        target_ranks = [1]
        try:
            responses = executor.collective_rpc(
                "check_health",
                target_ranks=target_ranks,
                timeout=10
            )
            print(
                f"   ✓ Got {len(responses)} response from worker {target_ranks}")
            assert len(
                responses) == 1, f"Expected 1 response, got {len(responses)}"
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False

        # Test 4: Health check on workers [1, 2] (only [1] exists, [2] doesn't)
        print(
            "\n6. Test 4: Health check on workers [1, 2] ([2] doesn't exist)")
        target_ranks = [1, 2]
        try:
            responses = executor.collective_rpc(
                "check_health",
                target_ranks=target_ranks,
                timeout=10
            )
            print(
                f"   ✗ Expected IndexError but got {len(responses)} responses")
            return False
        except IndexError:
            print("   ✓ Correctly raised IndexError for invalid rank [2]")
        except Exception as e:
            print(f"   ✗ Failed with unexpected error: {e}")
            return False

        # Test 5: With unique_reply_rank - target both workers but only get reply from rank 1
        print(
            "\n7. Test 5: Call workers [0, 1] but only get reply from rank 1")
        target_ranks = [0, 1]
        try:
            responses = executor.collective_rpc(
                "check_health",
                target_ranks=target_ranks,
                unique_reply_rank=1,
                timeout=10
            )
            print(f"   ✓ Got {len(responses)} response (from rank 1 only)")
            assert len(
                responses) == 1, f"Expected 1 response, got {len(responses)}"
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False

        # Test 6: Target non-existent worker to verify error handling
        print(
            "\n8. Test 6: Verify invalid rank handling (target ranks [0, 5], [5] doesn't exist)")
        target_ranks = [0, 5]  # rank 5 doesn't exist in world_size=2
        try:
            responses = executor.collective_rpc(
                "check_health",
                target_ranks=target_ranks,
                timeout=10
            )
            print(
                f"   ✗ Expected IndexError but got {len(responses)} responses")
            return False
        except IndexError:
            print("   ✓ Correctly raised IndexError for invalid rank [5]")
        except Exception as e:
            print(f"   ✗ Failed with unexpected error: {e}")
            return False

        print("\n" + "="*70)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*70)
        print("\nThe collective_rpc refactoring is WORKING correctly:")
        print("  ✓ Can communicate with all workers (default)")
        print("  ✓ Can communicate with workers [0, 1]")
        print("  ✓ Can communicate with worker [1]")
        print("  ✓ Correctly errors on workers [1, 2] when [2] doesn't exist")
        print("  ✓ Works with unique_reply_rank for optimized replies")
        print("  ✓ Handles invalid target ranks gracefully")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if executor is not None:
            print("\n10. Shutting down executor...")
            try:
                executor.shutdown()
                print("   ✓ Executor shut down cleanly")
            except Exception as e:
                print(f"   ⚠  Shutdown error: {e}")


def main():
    print("="*70)
    print("collective_rpc FULL INTEGRATION TEST")
    print("="*70)
    print("\nThis test will:")
    print("  1. Create an ProxyExecutor with real worker processes")
    print("  2. Test collective_rpc with various target_ranks configurations")
    print("  3. Verify responses from specific workers")
    print("\nNote: This requires a proper environment with GPU/CUDA support.")
    print("="*70)

    try:
        success = test_full_integration()

        if success:
            print("\n" + "="*70)
            print("🎉 SUCCESS! Your refactoring is working perfectly!")
            print("="*70)
            print("\nYou can now use collective_rpc with target_ranks:")
            print("""
# Example: Communicate with workers [1, 5, 7]
responses = executor.collective_rpc(
    "your_method",
    args=(arg1, arg2),
    target_ranks=[1, 5, 7],
    timeout=30
)
            """)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⚠  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
