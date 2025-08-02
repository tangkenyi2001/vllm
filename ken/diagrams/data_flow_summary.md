# vLLM Data and Control Flow Summary

This document summarizes the data and control flow between the vLLM engine and its workers during an inference request. The process can be broken down into four main stages:

### 1. Request Ingestion and Scheduling (Engine)

This stage is handled entirely by the `LLMEngine` and its components. It acts as the central controller, preparing requests for execution.

*   **Control Flow**:
    1.  A client submits a request to the `AsyncLLMEngine` (e.g., via `generate()`).
    2.  The request is added to a queue in the `RequestTracker`.
    3.  The background `run_engine_loop` picks up the request and passes it to the `LLMEngine`.
    4.  The `LLMEngine`'s `Scheduler` is invoked. It evaluates all waiting requests and the state of the KV cache.
    5.  The `Scheduler` decides which requests to run (`schedule()`), which to preempt, and what cache blocks to swap or copy. This is the primary control decision point.

*   **Data Flow**:
    1.  The initial `prompt` (string) is tokenized into `prompt_token_ids`.
    2.  A `SequenceGroup` is created to track the state of the request.
    3.  The `Scheduler` produces `SchedulerOutputs`, which contains:
        *   `seq_group_metadata_list`: The batch of requests to be executed, containing token IDs, positions, and KV cache pointers. This is the main data payload for the workers.
        *   `blocks_to_swap_in`, `blocks_to_swap_out`, `blocks_to_copy`: Instructions for the worker's `CacheEngine`.

### 2. Dispatch to Workers (Engine → Worker)

The engine packages the scheduler's plan and dispatches it to the workers for execution.

*   **Control Flow**:
    1.  The `LLMEngine` wraps the `SchedulerOutputs` into an `ExecuteModelRequest` object. This object serves as the primary command sent to the workers.
    2.  The `ModelExecutor` (e.g., `RayDistributedExecutor`) sends this command to the `Worker` instances.

*   **Data Flow**:
    *   The `ExecuteModelRequest` containing the `seq_group_metadata_list` and cache instructions is serialized and sent over the network or through IPC to the workers.

### 3. Execution (Worker)

The worker is the "leaf" node that performs the actual computation on the GPU.

*   **Control Flow**:
    1.  The `Worker` receives the `ExecuteModelRequest`.
    2.  It first calls `execute_worker()`, which processes the cache instructions, triggering `swap_in`, `swap_out`, and `copy` operations on its `CacheEngine`.
    3.  It then calls `execute_model()`, which invokes the `ModelRunner`.

*   **Data Flow**:
    1.  The `ModelRunner` uses the `seq_group_metadata_list` to prepare the input tensors for the model (e.g., `input_ids`, `attention_mask`).
    2.  The model's forward pass is executed on the GPU.
    3.  For tensor-parallel inference, workers communicate with each other (`All-Reduce`) to exchange intermediate activations.
    4.  The `ModelRunner` produces a `SamplerOutput`, which contains the logits or the sampled token IDs for each sequence in the batch. This is the result of the execution.

### 4. Result Aggregation and Post-processing (Worker → Engine)

The results are sent back to the engine, which processes them and delivers the final output to the client.

*   **Control Flow**:
    1.  The `Worker` returns the `SamplerOutput` to the `LLMEngine` via the `ModelExecutor`.
    2.  The `LLMEngine`'s `_process_model_outputs()` method is called.
    3.  The engine updates the state of the `SequenceGroup`s based on the results. It checks for stop conditions (e.g., EOS token, max length).

*   **Data Flow**:
    1.  The `SamplerOutput` (containing token IDs) is received by the engine.
    2.  The engine appends the new tokens to the sequences.
    3.  The `OutputProcessor` detokenizes the new tokens into text.
    4.  A `RequestOutput` object is created, containing the generated text delta and status.
    5.  This output is put into the `AsyncStream` for the request, which is then yielded back to the client.

