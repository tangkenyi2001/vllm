# SPDX-License-Identifier: Apache-2.0
"""RemoteExecutor for API servers to communicate with ProxyExecutor via RPC.

This executor runs in the API server process and forwards all calls to the
ProxyExecutor in the parent WorkerController process via MessageQueues.
"""
import logging
from typing import Optional, Union
from concurrent.futures import Future

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.v1.outputs import ModelRunnerOutput
from vllm.logger import init_logger

logger = init_logger(__name__)


class RemoteExecutor(ExecutorBase):
    """Executor that communicates with ProxyExecutor via MessageQueue RPC.

    This executor is used in API servers spawned by ProxyExecutor.
    Instead of managing workers directly, it sends RPC calls through
    MessageQueues to ProxyExecutor, which then uses collective_rpc()
    to communicate with the actual workers.

    Architecture:
        API Server (RemoteExecutor) --> [request_mq] --> ProxyExecutor
        API Server (RemoteExecutor) <-- [response_mq] <-- ProxyExecutor
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        request_queue,
        response_queue,
        engine_uuid: str,
        num_gpu_blocks: int = 0,
        num_cpu_blocks: int = 0,
    ) -> None:
        """Initialize RemoteExecutor.

        Args:
            vllm_config: Configuration for the model/engine
            request_queue: multiprocessing.Queue to send requests to ProxyExecutor
            response_queue: multiprocessing.Queue to receive responses from ProxyExecutor
            engine_uuid: Unique identifier for this API server/engine
            num_gpu_blocks: Pre-calculated number of GPU blocks (from ProxyExecutor)
            num_cpu_blocks: Pre-calculated number of CPU blocks (from ProxyExecutor)
        """
        super().__init__(vllm_config)
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.engine_uuid = engine_uuid

        # Store queues for RPC communication
        self.request_queue = request_queue  # API server -> ProxyExecutor
        self.response_queue = response_queue  # ProxyExecutor -> API server

        # Pre-calculated block counts (already determined by ProxyExecutor after loading model)
        self._num_gpu_blocks = num_gpu_blocks
        self._num_cpu_blocks = num_cpu_blocks

        logger.info(
            "RemoteExecutor initialized for engine '%s' with %s workers",
            engine_uuid, vllm_config.parallel_config.world_size)
        logger.info("Pre-calculated blocks: %s GPU, %s CPU", num_gpu_blocks,
                    num_cpu_blocks)

    def _init_executor(self) -> None:
        """Initialize the executor (already done in __init__)."""
        pass

    def __getattr__(self, name):
        """Forward unknown method calls to ProxyExecutor via RPC."""
        def wrapper(*args, **kwargs):
            return self._send_rpc(name, *args, **kwargs)
        return wrapper

    def _send_rpc(self, method: str, *args, **kwargs):
        """Send RPC request to ProxyExecutor and wait for response.

        Args:
            method: Method name to call on ProxyExecutor
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            The result from ProxyExecutor

        Raises:
            RuntimeError: If ProxyExecutor returns an error
        """
        # Send request
        self.request_queue.put((method, args, kwargs))

        # Wait for response
        status, result = self.response_queue.get()

        if status == "error":
            raise RuntimeError(f"ProxyExecutor error: {result}")

        return result

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Return pre-calculated number of available GPU and CPU blocks.

        These values were already calculated by ProxyExecutor after loading
        the model on the workers, so we don't need to query again.

        Returns:
            Tuple of (num_gpu_blocks, num_cpu_blocks)
        """
        logger.info("Returning pre-calculated blocks: %s GPU, %s CPU",
                    self._num_gpu_blocks, self._num_cpu_blocks)
        return self._num_gpu_blocks, self._num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int
    ) -> None:
        """Initialize KV cache on workers via ProxyExecutor.

        Args:
            num_gpu_blocks: Number of GPU blocks to allocate
            num_cpu_blocks: Number of CPU blocks to allocate
        """
        logger.info("Initializing cache with %s GPU blocks, %s CPU blocks",
                    num_gpu_blocks, num_cpu_blocks)
        self._send_rpc("initialize_cache", num_gpu_blocks, num_cpu_blocks)
        logger.info("Cache initialized successfully")

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Execute model on workers via ProxyExecutor.

        Args:
            scheduler_output: SchedulerOutput from the scheduler

        Returns:
            ModelRunnerOutput from the workers
        """
        # For v0: Remove async_callback if present (contains unpicklable weakrefs)
        # For v1: SchedulerOutput doesn't have async_callback
        if hasattr(scheduler_output, 'async_callback'):
            original_callback = scheduler_output.async_callback
            scheduler_output.async_callback = None
        else:
            original_callback = None

        try:
            result = self._send_rpc("execute_model", scheduler_output)
            return result
        finally:
            # Restore callback if it existed
            if original_callback is not None and hasattr(scheduler_output, 'async_callback'):
                scheduler_output.async_callback = original_callback

    def check_health(self) -> None:
        """Check if workers are healthy via ProxyExecutor."""
        try:
            self._send_rpc("check_health")
        try:
            self._send_rpc("check_health")
            logger.info("Health check passed")
        except Exception as e:
            logger.error("Health check failed: %s", e)
            raise

    def collective_rpc(self, *args, **kwargs):
            "collective_rpc should not be called directly on RemoteExecutor. "
            "All RPC calls are forwarded to ProxyExecutor via MessageQueue."
        )

    def shutdown(self) -> None:
        """Shutdown the executor."""
        logger.info("RemoteExecutor shutting down for engine '%s'",
                    self.engine_uuid)
        # MessageQueues will be cleaned up by ProxyExecutor

    def __del__(self):
        """Cleanup on deletion."""
        pass
