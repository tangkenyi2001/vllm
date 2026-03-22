# Worker Controller README

## Abstract

This document describes the `vllm.worker_controller` subsystem as an extension to upstream vLLM V1. The core design objective is to decouple worker process lifecycle from model-engine lifecycle. Instead of constructing workers and loading models in the same startup path, Worker Controller prewarms GPU workers with dummy configuration and binds them to concrete engines at runtime. This shifts startup overhead out of the online critical path, enables worker reuse across engine instances, and introduces explicit engine-to-worker IPC routing semantics.

## 1. Problem Statement

In conventional vLLM usage, engine startup includes multiple heavyweight operations in one path:

1. Process creation and interpreter/module initialization.
2. CUDA context and distributed group setup.
3. Model runner instantiation and weight loading.

For multi-model or elastic serving, repeated execution of this startup path increases cold-start latency and operational churn. Worker Controller addresses this by introducing a broker process that owns a long-lived worker pool and dynamically assigns worker subsets to engines.

## 2. Design Overview

### 2.1 Architectural decomposition

- **Worker Controller process**: owns reusable workers and resource allocation.
- **API engine processes**: created per engine UUID, connected to assigned workers through IPC queues.
- **Pool-side executor (`ProxyExecutor`)**: receives method requests and routes them to target ranks.
- **Engine-side executor (`RemoteExecutor`)**: presents Executor API while delegating execution over IPC.

### 2.2 High-level dataflow

1. Prewarm worker pool with dummy config.
2. Allocate ranks/port for an incoming engine.
3. Spawn API process and pass `request_queue`/`response_queue`.
4. Use `RemoteExecutor` inside API process to invoke worker RPCs.
5. Perform lazy `load_model(vllm_config)` on assigned workers only.
6. Serve requests.
7. On deletion, unload model and recycle ranks.

## 3. Implementation Contributions (with code snippets)

### 3.1 Dummy config bootstrapping for worker creation

**Contribution:** Introduces `DummyModelConfig` and `DummyVllmConfig` so workers can be created before the final model is known.

```python
# worker_controller.py (abridged)
model_config = DummyModelConfig("dummy", enforce_eager=enforce_eager)
cache_config = CacheConfig(gpu_memory_utilization=gpu_memory_utilization)
parallel_config = ParallelConfig(
  tensor_parallel_size=tensor_parallel_size,
  pipeline_parallel_size=pipeline_parallel_size,
  world_size=gpu_count,
  worker_cls="vllm.worker_controller.worker.gpu_worker.Worker",
)

dummy_vllm_config = DummyVllmConfig(
  model_config=model_config,
  cache_config=cache_config,
  parallel_config=parallel_config,
)
self.executor = ProxyExecutor(vllm_config=dummy_vllm_config)
```

**Impact:** process/device bring-up no longer depends on concrete model selection.

---

### 3.2 Resource allocation and engine lifecycle control

**Contribution:** Adds explicit allocator for rank ownership and serving port mapping.

```python
# worker_controller.py (abridged)
class ResourceAllocator:
  def assign(self, num: int, uuid: str):
    if uuid in self.uuid_to_port:
      port = self.uuid_to_port[uuid]
    else:
      port = self.next_port
      self.uuid_to_port[uuid] = port
      self.next_port += 1

    assigned_ranks = []
    for rank, val in self.resources.items():
      if val == 0 and len(assigned_ranks) < num:
        self.resources[rank] = uuid
        self.rank_to_uid[rank] = uuid
        assigned_ranks.append(rank)
    return assigned_ranks, port
```

```python
# worker_controller.py:create (abridged)
ctx = multiprocessing.get_context("forkserver")
request_queue = ctx.Queue()
response_queue = ctx.Queue()
self.executor.add_engine(engine_uuid, assigned_ranks, request_queue, response_queue, dist_port)

proc = ctx.Process(
  target=run_api_server,
  args=(request_queue, response_queue, engine_uuid, vllm_config, port),
  name=f"APIServer-{engine_uuid}",
)
proc.start()
```

**Impact:** deterministic engine provisioning and reclamation semantics.

---

### 3.3 Multiprocess executor extension into a routing broker

**Contribution:** `ProxyExecutor` extends multiprocess behavior by introducing per-engine routing state and request polling.

```python
# executor/proxy_executor.py (abridged)
class ProxyExecutor(Executor):
  def __init__(self, vllm_config, monitor_workers=True):
    self.monitor_workers = monitor_workers
    self.engines = {}
    self.running = True
    super().__init__(vllm_config)

  def add_engine(self, engine_uuid, ranks, request_queue, response_queue, dist_port=None):
    self.engines[engine_uuid] = {
      "ranks": ranks,
      "request_queue": request_queue,
      "response_queue": response_queue,
      "dist_port": dist_port,
    }
```

```python
# executor/proxy_executor.py (abridged)
def run_loop(self):
  while self.running:
    had_work = False
    for engine_uuid, engine in list(self.engines.items()):
      try:
        req = engine["request_queue"].get_nowait()
        method, args, kwargs = req
        target_ranks = engine["ranks"]

        self._broadcast_request(target_ranks, method, args, kwargs)
        self._collect_and_forward_response(engine["response_queue"], target_ranks)
        had_work = True
      except queue.Empty:
        pass
    if not had_work:
      time.sleep(0.0001)
```

**Impact:** enables one worker pool to serve multiple engines over isolated IPC channels.

---

### 3.4 Rank-filtered dispatch within broadcast transport

**Contribution:** wraps RPC call with rank guard so only assigned workers execute.

```python
# executor/proxy_executor.py (abridged)
def check_rank_and_execute(target_ranks, method, worker, *args, **kwargs):
  if worker.rank in target_ranks:
    logical_rank = target_ranks.index(worker.rank)
    worker.logical_rank = logical_rank
    if isinstance(method, str):
      func = getattr(worker, method)
      return func(*args, **kwargs)
  return SKIP_RESPONSE
```

```python
# executor/proxy_executor.py (abridged)
def _broadcast_request(self, target_ranks, method, args, kwargs):
  wrapped_method = partial(check_rank_and_execute, target_ranks, method)
  send_method = cloudpickle.dumps(wrapped_method, protocol=pickle.HIGHEST_PROTOCOL)
  self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, None))
```

**Impact:** preserves broadcast transport efficiency while enforcing engine/rank isolation.

---

### 3.5 Engine-side IPC executor abstraction

**Contribution:** `RemoteExecutor` emulates executor API over `request_queue` / `response_queue`.

```python
# executor/remote_executor.py (abridged)
class RemoteExecutor(Executor):
  def __init__(self, vllm_config, request_queue, response_queue):
    self.request_queue = request_queue
    self.response_queue = response_queue
    self._response_pool = ThreadPoolExecutor(max_workers=4)
    super().__init__(vllm_config)

  def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False,
             unique_reply_rank=None, kv_output_aggregator=None):
    req = (method, args, kwargs or {})
    self.request_queue.put(req)
    res = self.response_queue.get(timeout=timeout if timeout else None)
    return self._process_response(res, unique_reply_rank, kv_output_aggregator)
```

**Impact:** API process remains executor-compatible while physical execution happens in prewarmed workers.

---

### 3.6 Worker lifecycle redesign (lazy runner + explicit unload)

**Contribution:** modifies GPU worker lifecycle semantics from eager to lazy model binding.

```python
# worker/gpu_worker.py (abridged)
def init_device(self):
  init_worker_distributed_environment(...)
  set_random_seed(self.model_config.seed)
  self.init_snapshot = MemorySnapshot()
  self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
  self.model_runner = None
```

```python
# worker/gpu_worker.py (abridged)
def load_model(self, vllm_config: VllmConfig) -> dict:
  self.vllm_config = vllm_config
  self.model_config = vllm_config.model_config
  self.parallel_config = vllm_config.parallel_config

  new_tp = vllm_config.parallel_config.tensor_parallel_size
  new_pp = vllm_config.parallel_config.pipeline_parallel_size
  old_tp = getattr(self, "_last_tp_size", None)
  old_pp = getattr(self, "_last_pp_size", None)
  if old_tp != new_tp or old_pp != new_pp:
    destroy_model_parallel()
    ensure_model_parallel_initialized(
      tensor_model_parallel_size=new_tp,
      pipeline_model_parallel_size=new_pp,
    )

  self.model_runner = GPUModelRunner(vllm_config, self.device)
  self.model_runner.load_model(eep_scale_up=eep_scale_up)
  return timings
```

```python
# worker/gpu_worker.py (abridged)
def unload_model(self) -> None:
  if self.model_runner is not None:
    if hasattr(self.model_runner, "model") and self.model_runner.model is not None:
      del self.model_runner.model
      self.model_runner.model = None
    if hasattr(self.model_runner, "kv_caches") and self.model_runner.kv_caches is not None:
      for kv_cache in self.model_runner.kv_caches:
        if kv_cache is not None:
          del kv_cache
      self.model_runner.kv_caches = None
    del self.model_runner
    self.model_runner = None
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
```

**Impact:** worker reuse across models without process restart; reduced repeated bring-up costs.

---

### 3.7 API server integration and in-process engine core path

**Contribution:** API server detects Worker Controller context and selects custom AsyncLLM + RemoteExecutor path.

```python
# entrypoint/worker_controller_api_server.py (abridged)
if hasattr(args, "request_queue"):
  engine_args.request_queue = args.request_queue
if hasattr(args, "response_queue"):
  engine_args.response_queue = args.response_queue
if hasattr(args, "engine_uuid"):
  engine_args.engine_uuid = args.engine_uuid
```

```python
# entrypoint/worker_controller_api_server.py (abridged)
if hasattr(engine_args, "request_queue") and hasattr(engine_args, "response_queue"):
  from vllm.worker_controller.engine.async_llm import AsyncLLM as InprocAsyncLLM
  from vllm.worker_controller.executor.remote_executor import RemoteExecutor

  class RemoteExecutorFactory(RemoteExecutor):
    def __init__(self, config):
      super().__init__(config, engine_args.request_queue, engine_args.response_queue)

  async_llm = InprocAsyncLLM(
    vllm_config=vllm_config,
    executor_class=RemoteExecutorFactory,
    log_stats=not engine_args.disable_log_stats,
    usage_context=usage_context,
  )
```

**Impact:** avoids extra control-plane overhead by using an in-process engine core in this path.

## 4. Delta Analysis vs Upstream vLLM

| Dimension | Upstream reference behavior | Worker Controller behavior |
|---|---|---|
| Multiprocess executor role | `vllm/v1/executor/multiproc_executor.py` serves one engine context | `executor/proxy_executor.py` acts as a broker for many engines over one worker pool |
| Worker runner lifecycle | runner constructed eagerly in `init_device` | runner constructed lazily in `load_model(vllm_config)` |
| Model unload path | no explicit multi-model recycle path in baseline flow | explicit `unload_model` for teardown and memory hygiene |
| Engine-worker channel | internal executor channels | explicit per-engine queue IPC (`request_queue`, `response_queue`) |
| RPC target selection | collective executor semantics | rank-filtered dispatch with `target_ranks` |
| Control-plane startup | conventional async engine path | Worker Controller-specific in-process AsyncLLM + RemoteExecutor branch |

## 5. Detailed End-to-End Control Flow

### 5.1 System startup

1. Worker Controller computes GPU count and validates TP×PP divisibility.
2. Dummy config is constructed.
3. ProxyExecutor creates worker processes and transport queues.
4. Workers initialize device/distributed runtime but remain model-unbound.

### 5.2 Engine creation

1. Controller receives a create request with concrete `VllmConfig` and `engine_uuid`.
2. Allocator returns a rank subset and serving port.
3. Controller creates engine-specific request/response queues.
4. API process is spawned with queue handles and config.
5. API process builds Worker Controller AsyncLLM with RemoteExecutor.
6. RemoteExecutor triggers `load_model(vllm_config)` on assigned workers.

### 5.3 Serving

1. Engine-side executor sends method RPC to request queue.
2. ProxyExecutor run loop dequeues request and rank-filters dispatch.
3. Worker responses are aggregated and forwarded to engine response queue.
4. Engine receives model outputs and returns API responses.

### 5.4 Engine deletion

1. Controller terminates API process.
2. ProxyExecutor broadcasts `unload_model` to assigned ranks.
3. Allocator releases rank/port ownership.
4. Workers remain alive for subsequent model assignment.

## 6. Measurement and Observability Hooks

The implementation includes timing and startup instrumentation that is directly useful for a systems paper:

- worker-level model load timing decomposition (`config_time`, `dist_init_time`, `model_runner_init_time`, `weight_load_time`, `total_time`), returned by `worker/gpu_worker.py:load_model`.
- executor-level collection of worker timing in `executor/remote_executor.py`.
- engine startup summary logging in `engine/core.py`.
- consolidated qualitative/quantitative notes in `COLD_START_FINDINGS.md`.

## 7. Threats to Validity and Scope

1. **Instrumentation scope mismatch:** if compared against baseline using non-identical log anchors, sub-stage deltas may be biased.
2. **Model heterogeneity effects:** benefits depend on model size, TP/PP topology, and cache behavior.
3. **Control-plane vs dataplane gains:** design primarily optimizes startup/control-plane latency; steady-state token throughput may change less.
4. **Single-node assumption in many deployments:** current behavior is most directly evaluated in single-node multiprocess settings.

## 8. Setup and Running Scripts

### 8.1 Installation

Clone the repo and install in editable mode using `uv`:

```bash
git clone <repo-url>
cd vllm
uv pip install -e .
```

If you don't have `uv`, install it first:

```bash
pip install uv
```

### 8.2 Environment variables

Most scripts expect the following environment variables:

| Variable | Description | Example |
|---|---|---|
| `HF_HOME` | Hugging Face cache directory | `/dev/shm/models` |
| `VLLM_SKIP_KERNEL_WARMUP` | Skip kernel warmup for faster cold start benchmarks | `1` |
| `VLLM_KVC_MEM_GB` | KV cache memory cap per GPU (comma-separated) | `16,16` |
| `HUGGING_FACE_HUB_TOKEN` | HF token for gated models (optional) | |

### 8.3 Running servers locally

**Standard vLLM server:**

```bash
bash vllm/worker_controller/scripts/run_std_server.sh
```

This starts the standard vLLM API server on port 8000 and logs to `vllm/worker_controller/logs/std_server.log`.

**Worker Controller server:**

```bash
bash vllm/worker_controller/scripts/run_worker_controller_server.sh
```

This starts the Worker Controller server on port 8000, sets `VLLM_KVC_MEM_GB=16,16`, and logs to `vllm/worker_controller/logs/workerlogs.log`.

### 8.4 Running benchmarks

**Cold start benchmark (Python — recommended):**

```bash
# Full benchmark: 1 warmup + 30 measured runs for both server types
bash vllm/worker_controller/scripts/benchmark_cold_start.sh

# Customize runs and server type
python vllm/worker_controller/scripts/benchmark_cold_start.py -n 10 --warmup 1
python vllm/worker_controller/scripts/benchmark_cold_start.py -n 5 --only std
python vllm/worker_controller/scripts/benchmark_cold_start.py -n 5 --only wc
```

Results are saved to `vllm/worker_controller/logs/benchmark_results.json`.

**Cold start benchmark (shell — alternative):**

```bash
bash vllm/worker_controller/scripts/run_benchmark.sh
```

Runs 10 iterations each of the standard server and Worker Controller server, watches for `Engine total` in output, and prints a summary.

**PCIe bandwidth test:**

```bash
python vllm/worker_controller/scripts/benchmark_pcie.py
```

Measures host-to-device and device-to-host PCIe bandwidth (1 GB transfers with pinned memory) for all available GPUs.

### 8.5 Parsing benchmark logs

```bash
# Parse a single log file
python vllm/worker_controller/scripts/parse_logs.py vllm/worker_controller/logs/std_server.log

# Compare two log files side-by-side with Welch's t-test
python vllm/worker_controller/scripts/parse_logs.py \
    vllm/worker_controller/logs/std_server.log \
    vllm/worker_controller/logs/workerlogs.log \
    --labels "Standard" "WorkerController"
```

