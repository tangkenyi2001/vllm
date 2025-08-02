# vLLM Worker

The `Worker` class in vLLM is responsible for executing a partition of the model on a single GPU. It is a crucial component for both single-GPU and distributed inference, managing the model, KV cache, and the execution of the forward pass.

## Key Responsibilities:

1.  **Model Execution**:
    *   The primary role of the worker is to execute the model on its assigned GPU. It receives a batch of sequence group metadata from the engine and performs the forward pass.
    *   It uses a `ModelRunner` to manage the model's execution, which can be a standard `ModelRunner`, `EncoderDecoderModelRunner`, or `PoolingModelRunner` depending on the model type.

2.  **KV Cache Management**:
    *   Each worker maintains a portion of the KV cache. The `CacheEngine` is responsible for managing the allocation, swapping, and copying of KV cache blocks.
    *   The worker executes cache operations (swap-in, swap-out, copy) as instructed by the scheduler before running the model.

3.  **Distributed Inference**:
    *   In a distributed setting, each worker is assigned a rank and a partition of the model.
    *   It initializes the distributed environment and ensures that model-parallel and KV-transfer communication is set up correctly.
    *   Workers communicate with each other during the forward pass to exchange activations and gradients, enabling tensor and pipeline parallelism.

4.  **Initialization and Profiling**:
    *   The worker initializes its assigned GPU device and loads its partition of the model weights.
    *   It includes a `determine_num_available_blocks()` method that profiles the model's memory usage to calculate the maximum number of KV cache blocks that can be allocated without causing out-of-memory errors.
    *   It also warms up the model by running a few dummy forward passes, which may include capturing CUDA graphs for performance optimization.

## Core Components:

*   **`ModelRunner`**: This class is responsible for loading the model, running the forward pass, and managing the model's state. It prepares the input tensors, executes the model, and returns the sampler output.
*   **`CacheEngine`**: Manages the GPU and CPU KV caches. It provides methods for swapping blocks between GPU and and CPU memory and for copying blocks within the GPU.
*   **`LoRARequest` Management**: The worker can dynamically load, unload, and switch between different LoRA adapters as requested by the engine.

## Important Code Snippets

### `determine_num_available_blocks()`

This method is fundamental to vLLM's efficient memory management. It profiles the model to find the peak activation memory and calculates how many KV blocks can fit in the remaining GPU memory.

```python
# From: vllm/worker/worker.py

@torch.inference_mode()
def determine_num_available_blocks(self) -> Tuple[int, int]:
    """Profiles the peak memory usage of the model to determine how many
    KV blocks may be allocated without OOMs."""
    
    # Profile the memory usage of the model and get the maximum number of
    # cache blocks that can be allocated with the remaining free memory.
    torch.cuda.empty_cache()
    
    # Execute a forward pass with dummy inputs to profile memory usage.
    with memory_profiling(...) as result:
        self.model_runner.profile_run()

    # Calculate the memory available for the KV cache.
    memory_for_current_instance = total_gpu_memory * \
        self.cache_config.gpu_memory_utilization
    available_kv_cache_memory = (memory_for_current_instance -
                                 result.non_kv_cache_memory)

    # Calculate the number of blocks that can be allocated.
    cache_block_size = self.get_cache_block_size_bytes()
    num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
    num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                         cache_block_size)
    
    return num_gpu_blocks, num_cpu_blocks
```

### `initialize_cache()`

After determining the number of available blocks, this method allocates the KV cache and warms up the model, which may involve capturing CUDA graphs for optimization.

```python
# From: vllm/worker/worker.py

def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
    """Allocate GPU and CPU KV cache with the specified number of blocks."""
    
    # ... (validation checks) ...

    self.cache_config.num_gpu_blocks = num_gpu_blocks
    self.cache_config.num_cpu_blocks = num_cpu_blocks

    # Initialize the cache engines for GPU and CPU.
    self._init_cache_engine()
    
    # Warm up the model, which may capture CUDA graphs.
    self._warm_up_model()
```

### `execute_worker()`

This method is called for each step of the inference process. It's responsible for executing the cache operations (swapping, copying) that the scheduler has planned for the current iteration. The actual model execution happens in a subsequent step (`execute_model`).

```python
# From: vllm/worker/worker.py

@torch.inference_mode()
def execute_worker(self, worker_input: WorkerInput) -> None:
    virtual_engine = worker_input.virtual_engine
    
    # Issue cache operations. These are CUDA memory operations that are
    # scheduled by the engine to manage the KV cache.
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
```

The `Worker` is a fundamental building block of vLLM's distributed architecture, enabling it to scale inference across multiple GPUs and nodes efficiently.