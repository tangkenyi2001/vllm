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
from vllm.v1.executor.abstract import Executor
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
from vllm.worker_controller.config.vllm_config import DummyModelConfig, DummyVllmConfig, CacheConfig, ParallelConfig
from vllm.engine.llm_engine import LLMEngine

from multiprocessing import Pipe, Process
from typing import List
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.worker_controller.entrypoint.api_server import run_server
from vllm.utils import (Device, FlexibleArgumentParser, decorate_logs,
                        get_open_zmq_ipc_path, is_valid_ipv6_address,
                        set_ulimit)
import uvloop
logger = init_logger(__name__)
logging.basicConfig(
    level=logging.INFO,   # Show INFO and above
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class CloudpickleQueue:
    """A Queue wrapper that uses cloudpickle for serialization.

    This allows pickling of complex objects like weakrefs and lambdas
    that standard pickle cannot handle.
    """

    def __init__(self):
        self._queue = multiprocessing.Queue()

    def put(self, obj):
        """Put an object on the queue using cloudpickle."""
        pickled = cloudpickle.dumps(obj)
        self._queue.put(pickled)

    def get(self, block=True, timeout=None):
        """Get an object from the queue and unpickle it."""
        pickled = self._queue.get(block=block, timeout=timeout)
        return cloudpickle.loads(pickled)

    def empty(self):
        """Check if queue is empty."""
        return self._queue.empty()


class ProxyExecutor((Executor)):
    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        vllm_config = self.vllm_config
        self.world_size = vllm_config.parallel_config.world_size
        self.vllm_config = vllm_config

        set_multiprocessing_worker_envs(vllm_config.parallel_config)

        self.pipes = {}
        self.engines = {}
        self.resource = None  # Will be set by WorkerController

        # Track API server message queues: engine_uuid -> {request_mq, response_mq}
        self.api_request_queues = {}

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
                logger.info("Creating Worker$%d", rank)
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                    ))
            # Wait for workers to be ready (communication setup only)
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()
                self.pipes[w.rank] = {
                    "parent_send": w.parent_send, "parent_recv": w.parent_recv}

            logger.info("PipeDictionary")
            logger.info("%s", self.pipes)

            self.start_worker_monitor()
            self.start_api_server_request_loop()
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

    def pipecall(self, rank: int, method: str, vllmconfig: dict):
        parentsend: Connection = self.pipes[rank]["parent_send"]
        parentsend.send((method, (vllmconfig,), {}, None))
        parentrecv: Connection = self.pipes[rank]["parent_recv"]
        return parentrecv.recv()

    def create(self, vllm_config: Union[VllmConfig, DummyVllmConfig], engineUUID: str):
        """Create an API server with MessageQueue-based RPC to this ProxyExecutor."""
        required = vllm_config.parallel_config.world_size
        logger.info("World Size %s", required)

        # Disable CUDA graphs for worker controller
        vllm_config.compilation_config.use_cudagraph = False

        # Assign workers to this engine
        assigned_ranks, port = self.resource.assign(required, engineUUID)
        logger.info("Assigned ranks %s to engine %s",
                    assigned_ranks, engineUUID)

        # Send "hold" RPC to workers via collective_rpc
        results = self.collective_rpc(
            "hold",
            args=(vllm_config, engineUUID),
            target_ranks=assigned_ranks
        )
        logger.info("Workers held: %s", results)

        # Send "load_model" RPC to workers via collective_rpc
        results = self.collective_rpc(
            "load_model",
            args=(vllm_config,),
            target_ranks=assigned_ranks
        )
        logger.info("Models loaded: %s", results)

        # Calculate available blocks after model is loaded
        # Query first worker only since they all have the same model
        results = self.collective_rpc(
            "determine_num_available_blocks",
            target_ranks=assigned_ranks,
            unique_reply_rank=assigned_ranks[0]
        )
        num_gpu_blocks, num_cpu_blocks = results[0]
        logger.info(
            "Available blocks: %s GPU, %s CPU", num_gpu_blocks, num_cpu_blocks)

        # Initialize KV Cache on workers
        self.collective_rpc(
            "initialize_cache",
            args=(num_gpu_blocks, num_cpu_blocks),
            target_ranks=assigned_ranks
        )
        logger.info("KV Cache initialized on workers")

        # Create CloudpickleQueue pair for API server communication
        # Use cloudpickle to handle complex objects like scheduler_output with weakrefs
        request_queue = CloudpickleQueue()  # API server -> ProxyExecutor
        response_queue = CloudpickleQueue()  # ProxyExecutor -> API server

        # Store queues for this engine
        self.api_request_queues[engineUUID] = {
            "request_queue": request_queue,
            "response_queue": response_queue,
            "num_gpu_blocks": num_gpu_blocks,  # Pre-calculated
            "num_cpu_blocks": num_cpu_blocks,  # Pre-calculated
        }

        # Prepare args for API server
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args()

        args.request_queue = request_queue
        args.response_queue = response_queue
        args.engine_uuid = engineUUID
        args.vllmconfig = vllm_config
        args.model = vllm_config.model_config.model
        args.port = port
        args.num_gpu_blocks = num_gpu_blocks
        args.num_cpu_blocks = num_cpu_blocks

        def run_api_server(args):
            uvloop.run(run_server(args))

        proc = Process(target=run_api_server, args=(args,), daemon=False)
        proc.start()

        # Store engine info
        self.engines[engineUUID] = {
            "workers": assigned_ranks,
            "vllm_config": vllm_config,
            "proc": proc,
        }

        logger.info(
            "Engine %s created with workers %s on port %s",
            engineUUID, assigned_ranks, port)

        return proc

    '''
    Delete the worker modelrunner and unhold the model
    '''

    def delete(self, engineUUID: str):
        """Delete an API server and clean up its resources."""
        # Get assigned ranks before releasing
        assigned_ranks = self.resource.get_ranks_by_uuid(engineUUID)

        # Release workers from resource allocator
        released_ranks, port = self.resource.release_by_uuid(engineUUID)
        logger.info("Released ranks %s and port %s", released_ranks, port)

        # Unload models from workers
        results = self.collective_rpc(
            "unload_model",
            args=(),
            target_ranks=assigned_ranks
        )
        logger.info("Unload model results: %s", results)

        # Unhold workers
        results = self.collective_rpc(
            "unhold",
            args=(),
            target_ranks=assigned_ranks
        )
        logger.info("Unhold results: %s", results)

        # Clean up API server message queues
        if engineUUID in self.api_request_queues:
            del self.api_request_queues[engineUUID]

        # Remove engine info
        if engineUUID in self.engines:
            del self.engines[engineUUID]

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

    def start_api_server_request_loop(self):
        """Start a thread that listens for RPC requests from API servers."""

        def api_server_request_loop():
            logger.info("API server request loop started")
            while not self.shutdown_event.is_set():
                # Check each API server's request queue
                for engine_uuid, queues in list(self.api_request_queues.items()):
                    request_queue = queues["request_queue"]
                    response_queue = queues["response_queue"]

                    try:
                        # Non-blocking check for requests (timeout=0.1s)
                        method, args, kwargs = request_queue.get(timeout=0.1)

                        # Handle the RPC request
                        try:
                            result = self._handle_api_server_request(
                                engine_uuid, method, args, kwargs
                            )
                            response_queue.put(("success", result))
                        except Exception as e:
                            logger.exception(
                                "Error handling API server request: %s", e)
                            response_queue.put(("error", str(e)))
                    except Exception:
                        # Timeout or queue closed, continue
                        continue

        Thread(target=api_server_request_loop,
               daemon=True,
               name="APIServerRequestLoop").start()

    def _handle_api_server_request(self, engine_uuid: str, method: str, args: tuple, kwargs: dict):
        """Handle an RPC request from an API server."""
        logger.debug(
            "Handling API server request: %s for engine %s", method, engine_uuid)

        # Get the target ranks for this engine
        target_ranks = self.resource.get_ranks_by_uuid(engine_uuid)

        if not target_ranks:
            raise RuntimeError(f"No workers assigned to engine {engine_uuid}")

        # Map API server method calls to ProxyExecutor methods
        if method == "execute_model":
            # Forward to execute_model with engine_uuid
            return self.execute_model(engine_uuid, *args, **kwargs)

        elif method == "determine_num_available_blocks":
            # Call collective_rpc to query first worker
            results = self.collective_rpc(
                "determine_num_available_blocks",
                args=args,
                kwargs=kwargs,
                unique_reply_rank=target_ranks[0]
            )
            return results[0]

        elif method == "initialize_cache":
            # Call all workers to initialize cache
            results = self.collective_rpc(
                "initialize_cache",
                args=args,
                kwargs=kwargs,
                target_ranks=target_ranks
            )
            return results

        elif method == "check_health":
            # Ping all workers for this engine
            results = self.collective_rpc(
                "ping",
                target_ranks=target_ranks
            )
            return results

        else:
            # Generic RPC forwarding
            results = self.collective_rpc(
                method,
                args=args,
                kwargs=kwargs,
                target_ranks=target_ranks
            )
            return results

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    # need to rewrite it to include the ranks of the specified workers

    def execute_model(
        self,
        engine_uuid,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        target_ranks = self.resource.get_ranks_by_uuid(engine_uuid)

        # --- FIX: Transform scheduler_output to ExecuteModelRequest ---
        from vllm.sequence import ExecuteModelRequest
        if not isinstance(scheduler_output, ExecuteModelRequest):
            # You may need to adjust this transformation depending on scheduler_output's structure
            scheduler_output = ExecuteModelRequest(**scheduler_output)
        # ------------------------------------------------------------

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            # Use the first rank in target_ranks as the reply rank
            output_rank = target_ranks[0] if target_ranks else self.output_rank
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
                target_ranks=target_ranks)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            target_ranks=target_ranks)

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
                       unique_reply_rank: Optional[int] = None,
                       target_ranks: Optional[list[int]] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Determine which ranks to send the RPC to
        if target_ranks is None:
            # Default: broadcast to all workers
            send_ranks = list(range(self.world_size))
        else:
            # Send only to specified workers
            send_ranks = target_ranks

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)

            # Enqueue the method call to the specified ranks
            for rank in send_ranks:
                self.rpc_broadcast_mq.enqueue(
                    (send_method, args, kwargs, rank))

            # Determine which workers to collect responses from
            if unique_reply_rank is not None:
                # Only get response from one specific worker
                workers = [self.workers[unique_reply_rank]]
            elif target_ranks is not None:
                # Get responses from the specified target workers
                workers = [self.workers[rank] for rank in target_ranks]
            else:
                # Get responses from all workers
                workers = self.workers

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

    def check_health(self) -> list[str]:
        """Check health of all workers and return their responses."""
        return self.collective_rpc("check_health", timeout=10)

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
    parent_send: Connection
    parent_recv: Connection
    death_writer: Optional[Connection] = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue  # The worker process writes to this MQ
    parent_send: Connection
    parent_recv: Connection
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
            parent_send=unready_handle.parent_send,
            parent_recv=unready_handle.parent_recv,
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
        logger.info("wrapper.init_worker %s completed", rank)
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

        # Control channel: parent -> worker (send), worker -> parent (recv)
        child_recv, parent_send, = Pipe(duplex=False)
        parent_recv, child_send = Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "child_recv": child_recv,
            "child_send": child_send
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
        return UnreadyWorkerProcHandle(
            proc, rank, reader, parent_send, parent_recv, death_writer)

    def reset_to_empty_state(self):
        """Reset this worker to empty state (unload model)."""
        if hasattr(self, 'worker') and self.worker is not None:
            # Unload model and reset worker
            self.worker = None
            logger.info("Worker %s reset to empty state", self.rank)

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
        logger.info("Worker %s started", os.getpid())

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

        child_recv = kwargs.pop("child_recv", None)
        child_send = kwargs.pop("child_send", None)

        logger.info("Worker %s", os.getpid())
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

        try:
            reader.close()

            # Redirect subprocess output to log files

            logger.info("Worker%s going to be set up", kwargs['local_rank'])
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
            worker.worker_busy_loop([child_send, child_recv])

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

    def worker_busy_loop(self, pipes: List[Connection]):
        """Main busy loop for Multiprocessing Workers"""
        logger = init_logger(f"Worker {os.getpid()}")
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()

            # Only execute and respond if this worker is the target (or broadcast)
            if output_rank is None or self.rank == output_rank:
                try:
                    if isinstance(method, str):
                        func = getattr(self.worker, method)
                    elif isinstance(method, bytes):
                        func = partial(cloudpickle.loads(method), self.worker)
                    output = func(*args, **kwargs)

                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.SUCCESS, output))

                except Exception as e:
                    # Notes have been introduced in python 3.11
                    if hasattr(e, "add_note"):
                        e.add_note(traceback.format_exc())
                    logger.exception("WorkerProc hit an exception.")
                    # exception might not be serializable, so we convert it to
                    # string, only for logging purpose.
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
