# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import traceback
import weakref
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any, Callable, Optional, Union, cast

import cloudpickle
import logging
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (decorate_logs, get_distributed_init_method,
                        get_loopback_ip, get_mp_context, get_open_port,
                        set_process_title)
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm_config import DummyModelConfig, DummyVllmConfig, CacheConfig, ParallelConfig

logger = init_logger(__name__)
logging.basicConfig(
    level=logging.INFO,   # Show INFO and above
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class ResourceAllocator:
    def __init__(self, num_resources: int):
        # 0 means free/unassigned
        self.resources = {i: 0 for i in range(num_resources)}

    def assign(self, num: int, uid: str = None):
        assigned_ranks = []
        for rank, val in self.resources.items():
            if val == 0 and len(assigned_ranks) < num:
                self.resources[rank] = uid
                assigned_ranks.append(rank)

        if len(assigned_ranks) < num:
            # Not enough free resources
            raise RuntimeError(
                f"Only {len(assigned_ranks)} free resources, requested {num}"
            )

        return assigned_ranks

    def release_by_uuid(self, uid: str):
        """Release all resources assigned to this UUID."""
        assigned_ranks = []
        for rank, val in self.resources.items():
            if val == uid:
                self.resources[rank] = 0
                assigned_ranks.append(rank)
        return assigned_ranks

    def free(self):
        """Return list of currently free ranks."""
        return [rank for rank, val in self.resources.items() if val == 0]

    def num_free(self) -> int:
        """Return the number of currently free ranks."""
        return sum(1 for val in self.resources.values() if val == 0)

    def status(self):
        """Return mapping: rank -> UUID/0."""
        return dict(self.resources)


class WorkerController:

    def __init__(self, vllm_config: VllmConfig) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = vllm_config.parallel_config.world_size
        self.available_workers = vllm_config.parallel_config.world_size
        self.vllm_config = vllm_config
        # hard code to simulate GPU.
        vllm_config.parallel_config.worker_cls = "vllm.worker-controller.gpu_worker.Worker"

        self.resource = ResourceAllocator(
            num_resources=vllm_config.parallel_config.world_size)

        set_multiprocessing_worker_envs(vllm_config.parallel_config)

        # Track engines and their assigned workers
        # engine_uuid -> {"workers": [ranks], "vllm_config": config}
        self.engines = {}
        # Initialize empty workers with minimal setup
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create empty workers (no model loading)
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False

        try:
            for rank in range(self.world_size):
                logger.info(f"Creating Worker${rank}")
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                    ))
            logger.info("config 0 pipe")
            logger.info(unready_workers[0].vllm_config_writer)
            # Wait for workers to be ready (communication setup only)
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.parallel_config = self.vllm_config.parallel_config
        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)

    def _configure_workers(self, worker_ranks: list[int], vllm_config: VllmConfig, engineUUID: str):
        """Send vllm_config to specified workers and wait for them to initialize."""
        handles_to_configure = [
            h for h in self.workers if h.rank in worker_ranks
        ]

        for handle in handles_to_configure:
            if handle.engineUUID is not None:
                raise RuntimeError(
                    f"Worker {handle.rank} already assigned to engine {handle.engineUUID}"
                )
            try:
                # Send config to worker via pipe
                handle.vllm_config_writer.send(vllm_config)
                handle.engineUUID = engineUUID
                logger.info(
                    f"Sent config to worker {handle.rank} for engine {engineUUID}")
            except (BrokenPipeError, EOFError) as e:
                logger.error(
                    f"Failed to send config to worker {handle.rank}: {e}")
                raise RuntimeError(
                    f"Worker {handle.rank} not available for configuration")

    def _reset_workers(self, worker_ranks: list[int]):
        """Reset specified workers back to empty state."""
        handles_to_reset = [
            h for h in self.workers if h.rank in worker_ranks
        ]

        for handle in handles_to_reset:
            try:
                # Send reset signal (None config)
                handle.vllm_config_writer.send(None)
                handle.engineUUID = None
                logger.info(f"Reset worker {handle.rank} to empty state")
            except (BrokenPipeError, EOFError) as e:
                logger.error(f"Failed to reset worker {handle.rank}: {e}")

    def create(self, vllm_config: VllmConfig, engineUUID: str):
        """Create an engine by assigning and configuring workers."""
        required = vllm_config.parallel_config.world_size
        available = self.available_workers
        if required > available:
            raise RuntimeError(
                f"Not enough resources: requested world_size={required}, "
                f"but only {available} workers are available."
            )

        # Assign workers to this engine
        assigned_ranks = self.resource.assign(required, engineUUID)
        self.available_workers -= required

        # Configure assigned workers with the vllm_config
        self._configure_workers(assigned_ranks, vllm_config, engineUUID)

        # Store engine info
        self.engines[engineUUID] = {
            "workers": assigned_ranks,
            "vllm_config": vllm_config
        }

        logger.info(
            f"Engine {engineUUID} created with workers {assigned_ranks}")

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        outputs = self.collective_rpc("take_draft_token_ids",
                                      unique_reply_rank=self.output_rank)
        return outputs[0]

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue(
                (send_method, args, kwargs, unique_reply_rank))

            workers = (self.workers[unique_reply_rank],
                       ) if unique_reply_rank is not None else self.workers
            responses = []

            def get_response(w: WorkerProcHandle,
                             dequeue_timeout: Optional[float] = None,
                             cancel_event: Optional[threading.Event] = None):
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause")
                return result

            for w in workers:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())

                if non_block:
                    result = self.io_thread_pool.submit(  # type: ignore
                        get_response, w, dequeue_timeout, self.shutdown_event)
                else:
                    result = get_response(w, dequeue_timeout)

                responses.append(result)

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()

            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None

            if workers := getattr(self, 'workers', None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        """Check health of all workers."""
        self.collective_rpc("check_health", timeout=10)
        return

    @property
    def max_concurrent_batches(self) -> int:
        # For empty workers, return 1 as default
        # This will be overridden when engines are created
        return 1

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    vllm_config_writer: Connection
    death_writer: Optional[Connection] = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue  # The worker process writes to this MQ
    vllm_config_writer: Connection
    engineUUID: str | None = None
    death_writer: Optional[Connection] = None

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            vllm_config_writer=unready_handle.vllm_config_writer,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: DummyVllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
    ):

        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper
        logger.info(f"wrapper.init_worker {rank} completed")
        logger.info(wrapper.worker)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
        self.worker.init_device()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        # create pipe to update the vllm_config later
        vllm_config_reader, vllm_config_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "vllm_config_pipe": vllm_config_reader,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer, vllm_config_writer)

    @staticmethod
    def hold(
        proc_handles: list[WorkerProcHandle], vllm_config: VllmConfig, updated_ranks: list, engineUUID: str
    ) -> list[WorkerProcHandle]:
        handles_to_update = [
            h for h in proc_handles if h.rank in updated_ranks]

        for handle in handles_to_update:
            if handle.engineUUID is not None:
                raise RuntimeError(
                    f"Worker {handle.rank} already has engineUUID {handle.engineUUID}. Cannot overwrite."
                )
            try:
                handle.vllm_config_writer.send(vllm_config)
                handle.engineUUID = engineUUID
            except (BrokenPipeError, EOFError):
                logger.error(
                    f"Worker {handle.rank} is not available for config update")

        return proc_handles

    @staticmethod
    def unhold(
        proc_handles: list[WorkerProcHandle], updated_ranks: list
    ) -> list[WorkerProcHandle]:
        handles_to_update = [
            h for h in proc_handles if h.rank in updated_ranks]

        for handle in handles_to_update:
            handle.engineUUID = None

        return proc_handles

    @staticmethod
    def load_model(self, vllm_config: VllmConfig):
        """Load model on this worker."""
        if self.worker is not None:
            self.worker.load_model()

    def unload_model(self):
        """Unload model from this worker."""
        if self.worker is not None:
            self.worker.unload_model()

    def update_process_vllm_config(self,
                                   vllm_config: VllmConfig, rank: int,
                                   ) -> None:
        """Update this worker process with new vllm_config and initialize model."""
        # change the vllmConfig
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)

        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        self.all_kwargs[rank]["vllm_config"] = vllm_config
        self.all_kwargs[rank]["rank"] = rank
        self.all_kwargs[rank]["is_driver_worker"] = is_driver_worker

        wrapper.init_worker(self.all_kwargs)
        self.worker = wrapper

        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        pp_str = f"PP{rank // tp_size}" if pp_size > 1 else ""
        tp_str = f"TP{rank % tp_size}" if tp_size > 1 else ""
        suffix = f"{pp_str}{'_' if pp_str and tp_str else ''}{tp_str}"
        process_name = "VllmWorker"
        if suffix:
            set_process_title(suffix, append=True)
            process_name = f"{process_name} {suffix}"
        decorate_logs(process_name)
        self.worker.init_device()

    def reset_to_empty_state(self):
        """Reset this worker to empty state (unload model)."""
        if hasattr(self, 'worker') and self.worker is not None:
            # Unload model and reset worker
            self.worker = None
            logger.info(f"Worker {self.rank} reset to empty state")

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    # Extract the message queue handle.
                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """
        logger = init_logger("worker")
        logger.info(f"Worker {os.getpid()} started")

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            # Mark unused parameters
            _ = signum, frame
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        vllm_config_pipe = kwargs.pop("vllm_config_pipe", None)

        logger.info(f"Worker {os.getpid()}")
        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    os.kill(os.getpid(), signal.SIGTERM)
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(target=monitor_parent_death,
                                   daemon=True,
                                   name="WorkerDeathMonitor")
            death_monitor.start()

        if vllm_config_pipe is not None:
            def monitor_config_updates():
                nonlocal worker
                logger.info(
                    "Config monitor thread started, waiting for configs...")
                try:
                    while True:
                        logger.info(f"Worker {os.getpid()} thread polling")
                        if vllm_config_pipe.poll(timeout=1.0):
                            new_cfg = vllm_config_pipe.recv()
                            logger.info("recv logs")
                            if new_cfg is None:
                                # Reset signal
                                logger.info(
                                    f"Worker {os.getpid()} resetting to empty state")
                                if worker is not None:
                                    worker.reset_to_empty_state()
                                    worker = None
                            else:
                                # New config - create worker if it doesn't exist
                                logger.info(
                                    f"Worker {os.getpid()} received new config")
                                if worker is None:
                                    # Create new worker with config
                                    logger.info("Worker is None")
                                    logger.info(new_cfg)
                                    worker = WorkerProc(new_cfg, kwargs['local_rank'],
                                                        kwargs['rank'], kwargs['distributed_init_method'],
                                                        kwargs['input_shm_handle'])
                                else:
                                    logger.info("Worker is not None")
                                    worker.update_process_vllm_config(
                                        new_cfg, kwargs['rank'])
                except EOFError:
                    logger.info("Config pipe closed, no more updates")
                except Exception as e:
                    logger.warning(f"Config monitoring error: {e}")

            config_monitor = Thread(target=monitor_config_updates,
                                    daemon=True,
                                    name="WorkerVllmConfigMonitor")
            config_monitor.start()

        try:
            reader.close()

            # Redirect subprocess output to log files

            logger.info(f"Worker{kwargs['local_rank']} going to be set up")
            # Only create worker if vllm_config is provided
            worker = WorkerProc(*args, **kwargs)

            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            # Start minimal busy loop for empty worker
            empty_worker_busy_loop(worker.rpc_broadcast_mq, worker.worker_response_mq,
                                   vllm_config_pipe, kwargs)

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue

            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, output))


def empty_worker_busy_loop(rpc_broadcast_mq, worker_response_mq, vllm_config_pipe, worker_kwargs):
    """Busy loop for empty workers that can receive configs and become full workers."""
    worker = None

    while True:
        try:
            # Check for new vllm_config from controller
            if vllm_config_pipe and vllm_config_pipe.poll():
                logger.info(f"Worker {os.getpid()} polling")
                new_cfg = vllm_config_pipe.recv()
                if new_cfg is None:
                    # Reset signal
                    logger.info(
                        f"Worker {os.getpid()} resetting to empty state")
                    if worker is not None:
                        worker.reset_to_empty_state()
                        worker = None
                else:
                    # New config - create worker if it doesn't exist
                    logger.info(f"Worker {os.getpid()} received new config")
                    if worker is None:
                        # Create new worker with config
                        worker = WorkerProc(new_cfg, worker_kwargs['local_rank'],
                                            worker_kwargs['rank'], worker_kwargs['distributed_init_method'],
                                            worker_kwargs['input_shm_handle'])
                        # Start the normal worker busy loop
                        logger.info("Worker configured, starting busy loop")
                        worker.worker_busy_loop()
                        return  # Exit empty loop when worker is ready
                    else:
                        worker.update_process_vllm_config(new_cfg, worker.rank)

            # Handle RPC calls with timeout to prevent blocking
            try:
                method, args, kwargs, output_rank = rpc_broadcast_mq.dequeue(
                    timeout=0.01)

                # Handle basic health checks for empty workers
                if method == "check_health" and worker is None:
                    if output_rank is None or worker_kwargs['rank'] == output_rank:
                        worker_response_mq.enqueue(
                            (WorkerProc.ResponseStatus.SUCCESS, "empty_worker_healthy"))
                elif worker is not None:
                    # Worker is configured, handle normally
                    try:
                        if isinstance(method, str):
                            func = getattr(worker.worker, method)
                        elif isinstance(method, bytes):
                            func = partial(cloudpickle.loads(
                                method), worker.worker)
                        output = func(*args, **kwargs)

                        if output_rank is None or worker_kwargs['rank'] == output_rank:
                            worker_response_mq.enqueue(
                                (WorkerProc.ResponseStatus.SUCCESS, output))
                    except Exception as e:
                        if hasattr(e, "add_note"):
                            e.add_note(traceback.format_exc())
                        logger.exception("WorkerProc hit an exception.")
                        if output_rank is None or worker_kwargs['rank'] == output_rank:
                            worker_response_mq.enqueue(
                                (WorkerProc.ResponseStatus.FAILURE, str(e)))
                else:
                    # Empty worker can't handle this method
                    if output_rank is None or worker_kwargs['rank'] == output_rank:
                        worker_response_mq.enqueue(
                            (WorkerProc.ResponseStatus.FAILURE, "worker_not_configured"))
            except TimeoutError:
                # No RPC message to handle, continue looping
                pass

            time.sleep(0.001)  # Small sleep to prevent busy waiting

        except EOFError:
            logger.info("Config pipe closed, no more updates")
            break
        except Exception as e:
            logger.warning(f"Empty worker loop error: {e}")
            time.sleep(0.1)
