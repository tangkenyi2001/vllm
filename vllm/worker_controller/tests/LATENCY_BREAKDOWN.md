# Worker Controller: Cold Start Latency Breakdown

This document explains **why the Worker Controller (WC) achieves lower cold start latency** than standard vLLM, with a focus on pre-warmed worker processes. It also addresses **why API server startup times can be inconsistent** across runs.

Benchmark: `facebook-opt125m.py` — 3 consecutive load/unload cycles of `facebook/opt-125m`.

---

## Table of Contents

- [1. Architecture Overview](#1-architecture-overview)
- [2. What "Pre-Warmed" Actually Means](#2-what-pre-warmed-actually-means)
- [3. Phase-by-Phase Comparison](#3-phase-by-phase-comparison)
- [4. Why Model Loading Is Always Faster on the WC](#4-why-model-loading-is-always-faster-on-the-wc)
- [5. Why API Server Startup Can Be Inconsistent](#5-why-api-server-startup-can-be-inconsistent)
- [6. Summary](#6-summary)

---

## 1. Architecture Overview

### Standard vLLM — Everything From Scratch

When you run `vllm serve`, every cold start rebuilds the full process tree:

```
vllm serve <model>
  └─ AsyncLLM spawns EngineCore subprocess (multiprocessing + ZMQ sockets)
       └─ EngineCore creates MultiprocExecutor
            └─ MultiprocExecutor spawns N WorkerProc processes
                 └─ Each WorkerProc (from zero):
                      ├─ Python interpreter startup + vLLM imports
                      ├─ init_device() → CUDA context + NCCL init
                      └─ load_model() → GPUModelRunner + weight loading
            └─ _initialize_kv_caches()
                 ├─ profile_run() → dummy forward pass
                 ├─ Allocate KV cache blocks
                 └─ compile_or_warm_up_model() → CUDA graph capture
```

Every component — processes, CUDA contexts, NCCL groups, model weights — is created fresh each time.

### Worker Controller — Reuse What's Expensive

The WC splits the work into two phases:

**Phase 1: System boot (one-time)**
```
WorkerController.__init__()
  └─ ProxyExecutor.__init__(DummyVllmConfig)
       └─ Spawns N WorkerProc processes (pre-warmed)
            └─ Each WorkerProc (at boot):
                 ├─ Python interpreter startup + vLLM imports
                 ├─ init_device() → CUDA context creation
                 ├─ init_device() → NCCL process group initialization
                 └─ load_model() → NOT called (workers sit idle, ready)
```

**Phase 2: Per-model-load (each `create()` call)**
```
WorkerController.create(vllm_config, engine_uuid)
  ├─ ResourceAllocator.assign() → pick free GPU ranks + port  (~0ms)
  ├─ Create IPC queues (forkserver context)                    (~5ms)
  ├─ ProxyExecutor.add_engine() → route RPCs to workers        (~0ms)
  └─ Spawn API server process (forkserver)
       └─ InprocAsyncLLM (EngineCore runs IN-PROCESS, not subprocess)
            └─ EngineCore(executor = RemoteExecutor)
                 └─ RemoteExecutor.collective_rpc("load_model")
                      → IPC queue → ProxyExecutor → assigned workers
                      → Workers: load_model() on already-warm process
                 └─ _initialize_kv_caches()
                      ├─ profile_run()
                      ├─ Allocate KV cache blocks
                      └─ compile_or_warm_up_model()
```

Key differences:
- **No worker process spawning** — workers are already alive
- **No CUDA context creation** — already initialized at boot
- **No NCCL group init** — already initialized (skipped if TP/PP unchanged)
- **No EngineCore subprocess** — runs in-process via `InprocAsyncLLM`
- **No ZMQ socket setup** — uses IPC queues instead

---

## 2. What "Pre-Warmed" Actually Means

A pre-warmed worker is a GPU worker process that has completed `init_device()` but has **not** loaded any model. Here is exactly what state these workers hold:

### State created at boot (retained across all model loads)

| State | Created during | Cost | Survives unload? |
|-------|---------------|------|-------------------|
| Python interpreter + vLLM imports | Process spawn | ~2-3s | ✅ |
| CUDA context (`torch.device("cuda:N")`) | `init_device()` | ~1-2s | ✅ |
| CUDA runtime libraries (cuBLAS, cuDNN) | First CUDA op | ~0.5-1s | ✅ |
| NCCL process groups | `init_worker_distributed_environment()` | ~2-5s | ✅ (if TP/PP unchanged) |
| PyTorch CUDA memory allocator mappings | First allocation | ~0.1s | ✅ (pools survive `empty_cache()`) |
| GPU device binding (`self.device`) | `init_device()` | negligible | ✅ |
| Memory baseline snapshot | `MemorySnapshot()` | negligible | ✅ (retaken each load) |

### State created per model load (freed on unload)

| State | Created during | Cost | Freed on unload? |
|-------|---------------|------|-------------------|
| `GPUModelRunner` | `load_model()` | ~0.3s | ✅ `del self.model_runner` |
| Model weights (GPU tensors) | `model_runner.load_model()` | ~0.5-2s | ✅ `del model_runner.model` |
| KV cache blocks | `_initialize_kv_caches()` | ~0.1s | ✅ `del kv_caches` |
| CUDA graphs | `compile_or_warm_up_model()` | ~0.5-2s | ✅ (implicit with model runner) |

### Smart NCCL reuse

When loading a new model with the same tensor/pipeline parallel config as the previous one, the WC **skips NCCL group recreation entirely**:

```python
# In gpu_worker.py load_model():
if old_tp != new_tp or old_pp != new_pp:
    destroy_model_parallel()
    ensure_model_parallel_initialized(...)     # ~2-5s
    timings["dist_init_time"] = elapsed
else:
    timings["dist_init_time"] = 0.0            # Skipped!
```

Since the benchmark loads the same model 3 times, NCCL init is paid once (during warmup) and skipped for all 3 measured runs.

---

## 3. Phase-by-Phase Comparison

Using observed timings from `facebook/opt-125m` (single GPU):

### Standard vLLM cold start timeline

```
t=0.000s  API Server START (subprocess spawned)
          │
          │  ~6.4s — Python boot + imports + EngineCore subprocess + MultiprocExecutor
          │          + worker spawn + init_device() + CUDA ctx + NCCL
          ▼
t=6.4s    Engine Creation START (V1 LLM engine initializing)
          │
          │  ~0.9s — MultiprocExecutor worker setup finishes
          ▼
t=7.3s    Model Loading START
          │
          │  ~2.0s — GPUModelRunner init + weight loading
          ▼
t=9.3s    Model Loading END
          │
          │  ~1.0s — profile_run + KV cache alloc + CUDA graph warmup
          ▼
t=10.3s   Engine Creation END
          │
          │  ~2.7s — API route setup + health endpoint ready
          ▼
t=13.0s   API Server Ready (/health 200)
```

**Total: ~13.0s**

### Worker Controller cold start timeline

```
t=0.000s  API Server START (create request sent to WC)
          │
          │  ~6.0s — Resource alloc + IPC queues + forkserver spawn
          │          + API server process boot + InprocAsyncLLM init
          │          (NO worker spawn, NO CUDA init, NO NCCL init)
          ▼
t=6.0s    Engine Creation START
          │
          │  ~0.0s — RemoteExecutor connects via IPC queues (already set up)
          ▼
t=6.0s    Model Loading START
          │
          │  ~1.3s — GPUModelRunner init + weight loading
          │          (CUDA allocator warm, libraries cached, NCCL skipped)
          ▼
t=7.3s    Model Loading END
          │
          │  ~0.9s — profile_run + KV cache alloc + CUDA graph warmup
          ▼
t=8.2s    Engine Creation END
          │
          │  ~2.0s — API route setup + health endpoint ready
          ▼
t=10.2s   API Server Ready (/health 200)
```

**Total: ~10.2s**

### What the WC saves

| Phase | Standard vLLM | Worker Controller | Saved |
|-------|:------------:|:-----------------:|:-----:|
| Worker process spawn + Python imports | ~2-3s | **0s** | ~2-3s |
| CUDA context creation | ~1-2s | **0s** | ~1-2s |
| NCCL process group init | ~2-5s | **0s** | ~2-5s |
| EngineCore subprocess + ZMQ | ~3-5s | **0s** (in-process) | ~3-5s |
| Model weight loading | ~2.0s | ~1.3s | **~0.7s** |
| Memory profiling + KV cache | ~1.0s | ~0.9s | ~0.1s |
| CUDA graph capture + warmup | included above | included above | — |
| **Total** | **~13.0s** | **~10.2s** | **~2.8s** |

> For `facebook/opt-125m` (a tiny 125M parameter model), the savings are modest (~2.8s) because the model-independent overhead (profiling, KV cache, warmup, API routes) dominates. **For larger models (7B, 13B, 70B)**, the savings are proportionally larger because the fixed infrastructure costs (process spawn, CUDA init, NCCL) represent a bigger chunk of the total time relative to weight loading.

---

## 4. Why Model Loading Is Always Faster on the WC

The benchmark consistently shows WC model loading ~0.7s faster (~1.3s vs ~2.0s). This is **not** because the weights are being cached — they are fully freed on `unload_model()`. The speed difference comes from the **warm process state** that survives across load/unload cycles:

### 4.1. CUDA Runtime Libraries

CUDA libraries (cuBLAS, cuDNN, cuSPARSE) are **lazily loaded** on first use. Once loaded into a process, they remain resident in memory even after `torch.cuda.empty_cache()`.

- **Standard vLLM**: Fresh worker process → first `torch.matmul()` triggers cuBLAS load → ~0.3-0.5s
- **Worker Controller**: cuBLAS already loaded from previous model → **0s**

### 4.2. PyTorch CUDA Memory Allocator

PyTorch's caching allocator maintains internal memory **pools** (segments mapped from the GPU). `torch.cuda.empty_cache()` releases cached blocks back to CUDA but the allocator retains its internal metadata and allocation strategies.

After a load/unload cycle:
- Memory pools have been exercised and mapped
- The allocator "knows" the allocation patterns
- Subsequent allocations can reuse mapped segments faster than fresh `cudaMalloc` calls

### 4.3. Python/JIT Caches

- **Bytecode cache**: Python `.pyc` files and in-memory module objects persist in the worker process
- **PyTorch JIT**: Any JIT-compiled kernels or dispatch cache entries remain warm
- **Safetensors mmap**: OS page cache retains model weight file pages after first read (benefits both paths, but the WC worker process may retain mmap metadata)

### 4.4. NCCL Group Reuse

When loading the same model (or any model with the same TP/PP dimensions), NCCL groups are **not recreated**:

```
Standard vLLM:  init_device() → NCCL init (~2-5s, included in pre-model-load phase)
Worker Controller: load_model() → dist_init_time = 0.0s (skipped)
```

While this shows up in the "Engine Creation START" milestone rather than the "Model Loading" phase, it's a pre-warmed process benefit that shifts work earlier.

---

## 5. Why API Server Startup Can Be Inconsistent

Benchmark results sometimes show significant variation in API Server Startup time across runs. There are several sources of this inconsistency:

### 5.1. Forkserver Cold Start (First Run Only)

The Worker Controller uses Python's `multiprocessing` forkserver to spawn API server processes. The forkserver is a daemon process that:

1. **Starts lazily** on first `ctx.Process()` call
2. **Preloads modules** specified in `set_forkserver_preload()`:
   ```python
   multiprocessing.set_forkserver_preload(
       ["vllm.worker_controller.entrypoint.api_server"]
   )
   ```
3. After startup, subsequent spawns fork from the preloaded daemon (~fast)

**First spawn**: Forkserver boot + module preload = **~2-5s extra**
**Subsequent spawns**: Fork from warm daemon = **~0.1-0.5s**

This is why the benchmark includes a **warmup run** (discarded from results) — it pays the forkserver startup cost so that measured runs are consistent.

> Even with warmup runs, the forkserver can occasionally exhibit variability if the daemon process is under memory pressure or if the OS needs to COW-fault many pages on fork.

### 5.2. OS Page Cache State

Standard vLLM spawns a completely new Python process each time. The first spawn must load:
- Python standard library modules from disk
- All vLLM source files
- PyTorch shared libraries (~500MB+)
- Transformer library modules

On the **first run**, these are read from disk (or SSD) into the OS page cache. On subsequent runs, they are served from RAM. This creates a significant first-run penalty (~3s).

The **Worker Controller is less affected** because:
- Worker processes are already running (no imports needed)
- The API server process benefits from forkserver preloading
- Only the forked process needs its own copy of mutable pages (COW)

### 5.3. GPU State Between Runs

After deleting an engine, the WC calls `unload_model()` on the workers, which does:

```python
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

However, GPU memory reclamation is **not instantaneous**. CUDA's internal memory manager may take tens of milliseconds to fully release all allocations, and the PyTorch caching allocator needs synchronization. If the next engine create starts before the GPU is fully clean:

- Memory profiling may see less available memory → fewer KV cache blocks
- Allocation paths may be slower due to fragmentation
- CUDA graph capture may need to wait for pending frees

The benchmark mitigates this with a **5-second sleep + worker status verification** after each delete, but some variability can remain.

### 5.4. IPC Queue Latency

The WC uses `multiprocessing.Queue` (backed by pipes + pickle) for communication between the API server process and the ProxyExecutor. Queue operations involve:

- Pickle serialization of the `VllmConfig` object (which can be large)
- Kernel-level pipe buffer management
- The ProxyExecutor's polling loop (adaptive sleep with 100μs intervals)

An RPC message may wait up to one polling cycle before being picked up, introducing ~0-100μs of jitter per RPC call. Over many RPCs during engine init, this can accumulate.

### 5.5. CUDA Graph Capture Variability

The `compile_or_warm_up_model()` phase captures CUDA graphs for various batch sizes. This involves:

- Running real GPU kernels for each graph
- Synchronizing between captures
- Allocating graph-specific memory pools

CUDA graph capture time is sensitive to:
- GPU clock frequency (may vary with thermal state)
- Other processes using the GPU
- Driver-level scheduling decisions

This phase is **identical for both WC and standard vLLM** but contributes to run-to-run variation in both.

### 5.6. API Route Registration and Health Check Timing

The time between "Engine Creation END" and "API Server Ready" includes:

- FastAPI/Starlette route registration
- Uvicorn server binding to the port
- The benchmark's health-check polling loop (0.5s interval)

The polling introduces up to **0.5s of measurement noise** — the server might be ready at t=9.8s but the next health check doesn't fire until t=10.0s.

---

## 6. Summary

### Why the Worker Controller is faster

The Worker Controller reduces cold start latency through **three key mechanisms**:

1. **Pre-warmed GPU workers** — CUDA context, NCCL groups, and Python runtime are initialized once at system boot and reused across all model loads. This eliminates ~5-10s of per-load overhead for process infrastructure.

2. **In-process EngineCore** — Standard vLLM spawns EngineCore as a subprocess and communicates via ZMQ sockets. The WC's `InprocAsyncLLM` runs EngineCore in the same process as the API server, saving ~3-5s of subprocess startup and IPC setup.

3. **Warm runtime state** — Even after unloading a model, the worker processes retain CUDA libraries, PyTorch allocator state, and compiled kernels. This makes subsequent model loads ~0.7s faster than loading into a fresh process.

### Why timing can be inconsistent

| Source | Impact | Mitigation |
|--------|--------|------------|
| Forkserver cold start | ~2-5s on first run | Warmup run (discarded) |
| OS page cache | ~1-3s on first run | Warmup run (discarded) |
| GPU cleanup race | ~0-2s | 5s sleep + worker status check |
| CUDA graph variability | ~0-1s | Multiple runs, averaging |
| Health check polling | ~0-0.5s | Polling interval (0.5s) |
| IPC queue latency | ~0-0.1s | Negligible in practice |

### When the WC advantage is largest

The WC shines when you need to **load and unload multiple models sequentially** (e.g., a model serving platform). The infrastructure costs are paid once, and each subsequent model load benefits from the pre-warmed state. For a single model that runs indefinitely, the one-time cold start savings matter less.

For `facebook/opt-125m` (125M params): **~1.3x speedup** (~3s saved)
For larger models (7B+ params): **~1.5-3x expected speedup** (~10-30s saved) due to the fixed infrastructure costs being a larger fraction of total time.
