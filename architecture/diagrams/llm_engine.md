# vLLM Engine

The vLLM engine is the core component responsible for managing the lifecycle of inference requests. It is comprised of two main classes: `LLMEngine` and `AsyncLLMEngine`. This document provides an overview of their architecture and responsibilities, along with key code snippets.

## LLMEngine

The `LLMEngine` is the synchronous, central orchestrator of the vLLM system. It is designed for high-throughput, iteration-level scheduling and efficient memory management.

### Key Responsibilities:

1.  **Request Management**:
    *   Receives generation requests from clients, which can be either `SamplingParams` for text generation or `PoolingParams` for embedding tasks.
    *   Manages a pool of active, waiting, and swapped sequence groups.

2.  **Initialization**:
    *   Initializes the tokenizer, model executor, and KV cache.
    *   Profiles the model to determine the optimal number of GPU and CPU KV cache blocks that can be allocated.

3.  **Scheduling**:
    *   The `Scheduler` is a critical component that determines which sequence groups to run in each iteration. It employs a scheduling policy (e.g., FCFS, priority) to batch requests and manage the KV cache efficiently.
    *   It handles preemption, swapping, and copying of KV cache blocks to maximize GPU utilization.

4.  **Execution**:
    *   The `step()` method is the main entry point for a single decoding iteration. It performs the following actions:
        *   Schedules the next batch of sequence groups.
        *   Calls the distributed `ModelExecutor` to run the model forward pass.
        *   Processes the model outputs, including decoding, sampling, and applying logits processors.
        *   Updates the state of sequence groups (e.g., appending new tokens, marking them as finished).

5.  **State Management**:
    *   Maintains the state of all sequence groups, including their status (e.g., `RUNNING`, `WAITING`, `FINISHED`).
    *   Manages LoRA adapters, allowing for dynamic loading and unloading.

### Core Components:

*   **`ModelExecutor`**: An abstraction for the underlying model runner, which can be distributed across multiple GPUs or nodes.
*   **`Scheduler`**: Manages the KV cache and decides which requests to execute.
*   **`CacheEngine`**: Manages the allocation and deallocation of KV cache blocks on both GPU and CPU.
*   **`TokenizerGroup`**: Manages tokenizers, including support for LoRA-specific tokenizers.

### Important Code Snippet: `step()` Method

The `step()` method is the heart of the `LLMEngine`, driving the entire inference process for a single iteration. It encapsulates the schedule-execute-process loop.

```python
# From: vllm/engine/llm_engine.py

def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
    """Performs one decoding iteration and returns newly generated results."""
    
    # 1. Schedule the next batch of requests
    # This is skipped if the engine is performing multi-step decoding
    # and has remaining steps.
    if not self._has_remaining_steps(seq_group_metadata_list):
        (seq_group_metadata_list, scheduler_outputs,
         allow_async_output_proc) = self.scheduler[virtual_engine].schedule()

    # ... (error handling and state management) ...

    if not scheduler_outputs.is_empty():
        # 2. Execute the model
        # The model_executor runs the forward pass on the workers.
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            # ... other parameters
        )
        outputs = self.model_executor.execute_model(
            execute_model_req=execute_model_req
        )
    else:
        outputs = []

    # 3. Process model outputs and update sequence states
    # This is where tokens are appended, and requests are marked as finished.
    if not allow_async_output_proc:
        self._process_model_outputs(ctx=ctx)
        self.do_log_stats(scheduler_outputs, outputs)
        self.do_tracing(scheduler_outputs)

    return ctx.request_outputs
```

## AsyncLLMEngine

The `AsyncLLMEngine` is an asynchronous wrapper around the `LLMEngine`, designed for online serving and concurrent request handling. It uses `asyncio` to run the engine's processing loop in the background.

### Key Features:

1.  **Asynchronous API**:
    *   Provides an `async` API for methods like `generate()` and `encode()`, allowing for non-blocking request submission.
    *   Returns an `AsyncGenerator` that streams the results back to the client as they are generated.

2.  **Background Loop**:
    *   The `run_engine_loop()` method runs continuously in the background, processing requests from the queue.
    *   It waits for new requests and kicks the `LLMEngine`'s `step_async()` method to perform decoding iterations.

3.  **Request Tracking**:
    *   The `RequestTracker` class manages the state of all incoming requests, including new, running, and aborted requests.
    *   It uses `asyncio.Queue` to handle communication between the client-facing API and the background engine loop.

### Important Code Snippets

#### `generate()` Method

This is the primary user-facing method for submitting a request. It adds the request to the tracker and returns an async generator that yields the results.

```python
# From: vllm/engine/async_llm_engine.py

async def generate(
    self,
    prompt: PromptType,
    sampling_params: SamplingParams,
    request_id: str,
    # ... other args
) -> AsyncGenerator[RequestOutput, None]:
    """Generate outputs for a request."""
    
    # Start the background loop if it's not already running.
    if not self.is_running:
        if self.start_engine_loop:
            self.start_background_loop()
        else:
            raise AsyncEngineDeadError("Background loop is not running.")

    # Add the request to the tracker. The background loop will pick it up.
    stream = self._request_tracker.add_request(
        request_id,
        verbose=self.log_requests,
        prompt=prompt,
        params=sampling_params,
        # ... other args
    )

    # Yield results from the stream as they become available.
    try:
        async for output in stream.generator():
            yield LLMEngine.validate_output(output, RequestOutput)
    except asyncio.CancelledError:
        await self.abort(request_id)
        raise
```

#### `run_engine_loop()` Method

This static method contains the `while True` loop that drives the engine. It waits for new requests and continuously calls `engine_step` to process them.

```python
# From: vllm/engine/async_llm_engine.py

@staticmethod
async def run_engine_loop(engine_ref: ReferenceType):
    """We use a weakref to the engine so that the running loop
    doesn't prevent the engine being garbage collected."""
    engine: Optional[AsyncLLMEngine] = engine_ref()
    if not engine:
        return

    has_requests_in_progress = [False] * pipeline_parallel_size
    while True:
        if not any(has_requests_in_progress):
            # If no requests are running, wait for new ones to arrive.
            await request_tracker.wait_for_new_requests()
            
        # Kick the engine to perform one step of processing.
        # This runs the scheduler, model execution, and output processing.
        requests_in_progress = [
            asyncio.create_task(engine.engine_step(ve))
            for ve in range(pipeline_parallel_size)
        ]
        
        # ... (logic to handle completed steps and continue the loop) ...
        
        await asyncio.sleep(0)
```
This architecture allows vLLM to achieve high performance and scalability for both offline batch processing and online serving.