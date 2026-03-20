# Worker Controller Extension Report (MLSys-Focused)


~/.cache/huggingface/hub
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

## 8. Reproducibility Checklist (for paper appendix)

- Report exact commit hash.
- Fix GPU model, driver, CUDA, and PyTorch versions.
- Report TP/PP configuration and number of visible GPUs.
- Distinguish:
  - first engine creation after controller boot,
  - subsequent engine creations with reused workers.
- Report p50/p95 for each startup stage and end-to-end startup.
- Include at least one small and one medium/large model.

## 9. Primary Artifact Index

- `worker_controller.py`
- `config/model.py` (`DummyModelConfig`)
- `config/vllm.py` (`DummyVllmConfig`)
- `executor/proxy_executor.py`
- `executor/remote_executor.py`
- `worker/gpu_worker.py`
- `entrypoint/worker_controller_api_server.py`
- `engine/async_llm.py`
- `engine/core.py`
- `COLD_START_FINDINGS.md`

## 10. Appendix: Publication-Ready Pseudo-code

This section provides algorithmic pseudo-code intended for direct inclusion in a paper appendix.

### Notation

- `R`: set of all worker ranks.
- `Free(r)`: rank `r` is unassigned.
- `E`: mapping `engine_uuid -> {ranks, request_queue, response_queue, proc, port}`.
- `Assign(num, uuid)`: allocator primitive returning `num` free ranks and a serving port.
- `Broadcast(ranks, method, args, kwargs)`: rank-filtered RPC enqueue to worker transport.
- `Collect(ranks)`: gather responses from target ranks.

### Algorithm 1: `CreateEngine`

```text
Algorithm 1 CreateEngine(vllm_config, engine_uuid)
Input:
  vllm_config: concrete model/runtime configuration
  engine_uuid: unique identifier for the requested engine
State:
  resource_allocator, executor (ProxyExecutor), E
Output:
  proc: API server process handle

1:  num_gpus ← vllm_config.parallel_config.world_size
2:  (assigned_ranks, port) ← Assign(num_gpus, engine_uuid)
3:  if num_gpus > 1 then
4:      dist_port ← resource_allocator.next_port
5:      resource_allocator.next_port ← resource_allocator.next_port + 1
6:  else
7:      dist_port ← None
8:  end if
9:
10: ctx ← multiprocessing.get_context("forkserver")
11: request_queue ← ctx.Queue()
12: response_queue ← ctx.Queue()
13:
14: executor.add_engine(
15:      engine_uuid,
16:      assigned_ranks,
17:      request_queue,
18:      response_queue,
19:      dist_port)
20:
21: proc ← ctx.Process(
22:      target=run_api_server,
23:      args=(request_queue, response_queue, engine_uuid, vllm_config, port),
24:      name=f"APIServer-{engine_uuid}")
25: proc.start()
26:
27: E[engine_uuid] ← {
28:      ranks: assigned_ranks,
29:      request_queue: request_queue,
30:      response_queue: response_queue,
31:      proc: proc,
32:      port: port
33: }
34: return proc
```

**Safety invariants:**

- Rank exclusivity: a rank is assigned to at most one live engine (`∀r ∈ R, |owners(r)| ≤ 1`).
- Queue isolation: each engine has private request/response queues.

### Algorithm 2: `RouteRPC` (ProxyExecutor main loop)

```text
Algorithm 2 RouteRPC()
State:
  E: engine map, running: bool
  response_mqs: per-rank response channels
Input per engine:
  request_queue items of form (method, args, kwargs)

1: while running do
2:     had_work ← false
3:     for each (engine_uuid, engine_state) in snapshot(E) do
4:         try
5:             req ← engine_state.request_queue.get_nowait()
6:             (method, args, kwargs) ← req
7:             target_ranks ← engine_state.ranks
8:
9:             wrapped ← partial(check_rank_and_execute, target_ranks, method)
10:            Broadcast(target_ranks, wrapped, args, kwargs)
11:
12:            responses ← []
13:            for each rank in target_ranks do
14:                (status, result) ← response_mqs[rank].dequeue(timeout=120)
15:                if status ≠ SUCCESS then
16:                    result ← Exception("Worker failed or timed out")
17:                end if
18:                append(responses, result)
19:            end for
20:
21:            engine_state.response_queue.put(responses)
22:            had_work ← true
23:         catch EmptyQueue
24:            continue
25:         catch Exception as err
26:            log_error(engine_uuid, err)
27:         end try
28:     end for
29:
30:     if had_work = false then
31:         sleep(100 microseconds)
32:     end if
33: end while
```

**Liveness note:** adaptive sleeping avoids busy-spin under low load while preserving low-latency polling under active load.

### Algorithm 3: `DeleteEngine`

```text
Algorithm 3 DeleteEngine(engine_uuid)
Input:
  engine_uuid
State:
  E, executor, resource_allocator
Output:
  success/failure

1: if engine_uuid ∈ E then
2:     proc ← E[engine_uuid].proc
3:     if proc ≠ None and proc.is_alive() then
4:         proc.terminate()
5:         proc.join(timeout=5)
6:         if proc.is_alive() then
7:             proc.kill()
8:         end if
9:     end if
10: end if
11:
12: executor.delete_engine(engine_uuid)
13:   // internally broadcasts unload_model to assigned ranks
14:   // drains rank responses to avoid stale queue state
15:
16: resource_allocator.release_by_uuid(engine_uuid)
17: remove E[engine_uuid] if present
18: return success
```

**Postconditions:**

- Workers previously assigned to `engine_uuid` are returned to free pool.
- No active API process remains for `engine_uuid`.
- Model state is unloaded on previously assigned ranks (best-effort with timeout handling).

### Complexity discussion (control-plane)

- `CreateEngine`: $O(|R|)$ worst-case allocation scan + process spawn overhead.
- `RouteRPC` per request: $O(k)$ where $k$ is number of assigned ranks for the engine.
- `DeleteEngine`: $O(k)$ for unload + response drain, plus process termination overhead.

### Mapping to concrete implementation

- Algorithm 1 maps to `WorkerController.create(...)` in `worker_controller.py`.
- Algorithm 2 maps to `ProxyExecutor.run_loop(...)` and helper methods in `executor/proxy_executor.py`.
- Algorithm 3 maps to `WorkerController.delete(...)` plus `ProxyExecutor.delete_engine(...)`.
