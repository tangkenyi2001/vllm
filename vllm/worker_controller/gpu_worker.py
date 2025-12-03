# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import copy
import gc
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.worker.worker_base import LocalOrDistributedWorkerBase
from vllm.worker.model_runner import ModelRunner
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.worker_controller.config.vllm_config import DummyVllmConfig

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


class Worker(LocalOrDistributedWorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # V0 WorkerBase only takes vllm_config
        LocalOrDistributedWorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        # set invoke status to false
        self.invoke: str | None = None

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self,
                                       tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be "
                    "used for one instance per process.")
            context = allocator.use_memory_pool(tag=tag)
        else:
            context = nullcontext()
        return context

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache."""
        from vllm.worker.cache_engine import CacheEngine

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Create cache engine for each pipeline parallel stage
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        # Extract gpu_cache from cache_engine
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.baseline_snapshot = MemorySnapshot()  # For V0 compatibility
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            if self.init_snapshot.free_memory < self.requested_memory:
                def GiB(b): return round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # report_usage_stats requires hf.architecture
        # if self.rank == 0:
        #     # If usage stat is enabled, collect relevant info.
        #     report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def reload(self):
        vllm_config = self.vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config
        from vllm.platforms import current_platform
        self.current_platform = current_platform

    def hold(self, vllmconfig: VllmConfig, invokerId: str):
        logger.info(f"{vllmconfig}")
        if self.invoke and self.invoke != invokerId:
            # Allow re-holding by the same invoker or unhold first
            logger.warning(
                f"Worker already held by {self.invoke}, being re-held by {invokerId}")
            self.unhold()
        # set invoked
        self.invoke = invokerId
        # Set vllmconfig
        self.vllm_config = vllmconfig
        # need to reload vllmconfig
        self.reload()
        logger.info(f"{os.getpid()} Hold worker called by {invokerId}")
        res = f"{os.getpid()} Model held"
        return res

    def unhold(self):
        if not self.invoke:
            raise RuntimeError("This worker has has not been invoked")
        logger.info(f"{os.getpid()} unhold worker called by {self.invoke}")
        self.invoke = None
        res = f"{os.getpid()} Model unheld"
        return res

    def update_vllm_config(self, vllmconfig: VllmConfig) -> str:
        """Update the vllm config at runtime without checking invoke status.

        This allows updating worker configuration after initialization,
        which is useful for dynamic model loading scenarios.
        """
        logger.info(f"{os.getpid()} Updating vllm config")
        logger.info(f"{vllmconfig}")
        # Set vllmconfig
        self.vllm_config = vllmconfig
        # Reload all config components
        self.reload()
        logger.info(f"{os.getpid()} vllm config updated")
        return f"{os.getpid()} Config updated"

    def is_model_loaded(self) -> dict:
        """Check if model is loaded and return status info."""
        if not hasattr(self, 'model_runner') or self.model_runner is None:
            return {
                "loaded": False,
                "pid": os.getpid(),
                "model": None
            }

        try:
            model = self.model_runner.get_model()
            return {
                "loaded": True,
                "pid": os.getpid(),
                "model_class": type(model).__name__,
                "device": str(self.device),
                "model_name": self.model_config.model
            }
        except Exception as e:
            return {
                "loaded": False,
                "pid": os.getpid(),
                "error": str(e)
            }

    def load_model(self, vllmconfig: VllmConfig) -> str:
        logger.info(f"{os.getpid()} LOAD MODEL CALLED")
        logger.info(f"{vllmconfig}")
        # create ModelRunner (V0) - it only takes vllm_config and a few optional params
        self.model_runner: ModelRunner = ModelRunner(
            vllm_config=vllmconfig,
            kv_cache_dtype=vllmconfig.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model()
        res = f"{os.getpid()} Model Loaded"
        return res

    def unload_model(self) -> str:
        logger.info(f"{os.getpid()} unload model called")
        # Release model runner
        self.model_runner = None
        gc.collect()
        torch.cuda.empty_cache()

        res = f"{os.getpid()} model unloaded"
        return res

    def ping(self) -> str:
        """Simple ping method for health checks."""
        return f"{os.getpid()} pong"

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.reload_weights()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        def GiB(b): return b / GiB_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(
                    self.model_runner.model_memory_usage)) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container.")
        available_kv_cache_memory = self.requested_memory \
            - profile_result.non_kv_cache_memory

        unrequested_memory = self.init_snapshot.free_memory \
            - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; "
            "Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), "
            "%.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info("Available KV cache memory: %.2f GiB",
                    GiB(available_kv_cache_memory))
        gc.collect()

        return int(available_kv_cache_memory)

    def compile_or_warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)
            # activate building attn_metadata for this dummy run to avoid
            # potential illegal memory access for full cudagraph relay.
            attn_cudagraph = self.compilation_config.full_cuda_graph and\
                not self.model_config.enforce_eager

            # We skip EPLB here since we don't want to record dummy metrics
            # skip cuda graph
            # hidden_states, last_hidden_states = \
            #     self.model_runner._dummy_run(
            #         num_tokens=max_num_reqs,
            #         skip_eplb=True,
            #     )
            # if self.model_runner.is_pooling_model:
            #     self.model_runner._dummy_pooler_run(hidden_states)
            # else:
            #     self.model_runner._dummy_sampler_run(
            #         hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes."""
        from vllm.worker.cache_engine import CacheEngine
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.baseline_snapshot,
                weights_memory=self.model_runner.model_memory_usage) as result:
            self.model_runner.profile_run()

        self._assert_memory_footprint_increased_during_profiling()

        memory_for_current_instance = total_gpu_memory * \
            self.cache_config.gpu_memory_utilization
        available_kv_cache_memory = (memory_for_current_instance -
                                     result.non_kv_cache_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        msg = (f"Memory profiling takes {result.profile_time:.2f} seconds\n"
               "the current vLLM instance can use "
               "total_gpu_memory "
               f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
               " x gpu_memory_utilization "
               f"({self.cache_config.gpu_memory_utilization:.2f})"
               f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
               "model weights take "
               f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
               " non_torch_memory takes "
               f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
               " PyTorch activation peak memory takes "
               f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
               " the rest of the memory reserved for KV Cache is "
               f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB.")

        logger.info(msg)
        # Final cleanup
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        free_gpu_memory, total = torch.cuda.mem_get_info()
        cuda_memory = total - free_gpu_memory
        assert self.baseline_snapshot.cuda_memory < cuda_memory, (
            "Error in memory profiling. "
            f"Initial used memory {self.baseline_snapshot.cuda_memory}, "
            f"currently used memory {cuda_memory}. "
            f"This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

    @property
    def do_metadata_broadcast(self) -> bool:
        """Return True if tensor parallelism > 1, requiring metadata broadcast."""
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self):
        """Return the KV cache (gpu_cache in this implementation)."""
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest):
        """Prepare worker input from execution request."""
        from vllm.worker.worker_base import WorkerInput

        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input) -> None:
        """Execute cache operations (swap in/out, copy)."""
        from vllm.worker.worker_base import WorkerInput

        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[list[SamplerOutput]]:
        """Executes at least one model step on the given sequences."""
        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None

        # output is List[SamplerOutput]
        return output

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> str:
        # Return a string identifying the worker process.
        return f"hi from process worker {self.rank}"

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)
