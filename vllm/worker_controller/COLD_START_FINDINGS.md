# Worker Controller Cold-Start Findings

## Summary
We compared cold-start behavior between:
- **Standard vLLM**: creates worker processes during engine startup.
- **Worker Controller**: attaches a new engine to **prewarmed workers**.

Across repeated runs with `facebook/opt-125m`, Worker Controller consistently reduced total cold-start time (typically around **~1-2+ seconds saved** in our runs).

## What improved most
The largest gains came from startup orchestration phases, especially:
- **Engine creation path** (faster when workers are prewarmed and reused)
- **Worker/process bring-up related work** (reduced or shifted out of per-engine startup)

Inference latency differences were small (tens of milliseconds), so the main benefit is startup, not per-request compute speed.

## Why API startup still looks large
In the charts, **"API Server Start Up"** is a residual bucket (unattributed startup overhead), not a single direct timer. It can include:
- process bootstrap/import overhead
- app/state initialization
- route/server readiness transitions
- health polling delay and timestamp granularity effects

So this value is expected to look large even when model load is relatively small.

## Scope notes (important)
Some metrics are from different instrumentation scopes between Standard and Worker Controller. In particular:
- **Engine creation** in Standard and WC is not a perfectly identical timer source.
- **Worker startup = 0.00s** can mean either truly tiny cost or that coarse log timing rounded it down / log line not available.

Interpret these rows directionally unless they are derived from the same log anchors.

## Practical conclusion
Worker Controller is most valuable when you repeatedly create/delete engines (multi-model or sequential loading workflows), because it amortizes one-time worker/CUDA setup and shortens per-engine startup.

## Suggested next step
For stricter apples-to-apples reporting, add a shared-scope metric derived from identical log anchors for both paths, and report p50/p95 over more runs.
