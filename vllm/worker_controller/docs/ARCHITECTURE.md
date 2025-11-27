# Worker Controller Architecture

This document outlines the architecture and functionality of the vLLM Worker Controller.

## Overview

The Worker Controller is a centralized service responsible for managing a shared pool of GPU worker processes. Its primary purpose is to eliminate the cold start time associated with loading models by decoupling the lifecycle of worker processes from the lifecycle of inference engines.

It exposes a REST API (by default on port 8000) to dynamically create and destroy vLLM engines. When a new engine is requested, the controller spawns a dedicated API server for that engine on a new port, which then utilizes the shared pool of pre-warmed workers for inference.

## Key Components

*   **Worker Controller**: The main process that runs the central FastAPI server (e.g., on port 8000). It initializes and manages the `ProxyExecutor`.

*   **Proxy Executor**: A long-running, core component that lives within the Worker Controller. It owns and manages the pool of `Worker Nodes` and orchestrates communication with them via RPC. It is the single point of contact for all spawned API Servers.

*   **Worker Node (`gpu_worker.py`)**: A dedicated process running on a GPU that performs the actual model inference. These are spawned at startup by the `ProxyExecutor` and wait for commands.

*   **API Server Process (`api_server.py`)**: A separate process spawned by the Worker Controller for each created engine. It runs its own FastAPI server on a dedicated port (e.g., 8001, 8002). It hosts the `vLLM Engine` and exposes endpoints like `/v1/completions`.

*   **vLLM Engine (`AsyncLLMEngine`)**: The main entry point for running inference within an API Server process. It is configured to use the `RemoteExecutor`.

*   **Remote Executor**: An executor client that runs inside the API Server process. It acts as the communication bridge, forwarding all execution requests (like `execute_model`) to the `ProxyExecutor` via a message queue RPC system.

## Architecture Diagram

```
+---------------------+                            +----------------------------+
|   External Client   |                            |       Worker Processes     |
+---------------------+                            | +----------+ +----------+  |
       |        ^                                    | | Worker 1 | | Worker 2 |...|
       | 1. POST /engines                           | +----------+ +----------+  |
       | 3. POST /completions                       +----------------------------+
       v        |                                                ^
+----------------------------------+                             |
| Worker Controller (port 8000)    |                             | RPC
|                                  |                             | (Message Queues)
|  +----------------------------+  |   2. Spawns API Server      |
|  |       Proxy Executor       |------------------------------->
|  +----------------------------+  |                             |
+----------------------------------+                             |
                 |                                               |
                 |                                               |
                 +-----------------------------------------------+
                                       |
                                       v
                          +-----------------------------------+
                          | API Server Process (port 8001+)   |
                          | +-----------------+             |
                          | |  vLLM Engine    |             |
                          | | (RemoteExecutor)|             |
                          | +-----------------+             |
                          +-----------------------------------+
```

## Workflow for Engine Creation

1.  **Initialization**: The `WorkerController` is started. It immediately initializes the `ProxyExecutor`, which in turn spawns the pool of `gpu_worker` processes. The controller now listens for API requests on its main port (e.g., 8000).
2.  **Engine Creation Request**: An external client sends a `POST` request to the `/engines` endpoint on the `WorkerController`.
3.  **API Server Spawning**: The `ProxyExecutor` receives this request. It assigns a unique port (e.g., 8001) and spawns a new `api_server.py` process. Crucially, it creates a request/response message queue pair and passes them to this new process.
4.  **Inference-Ready**: The new API Server process starts, initializes its `vLLM Engine` with the `RemoteExecutor`, and begins listening on its assigned port. It is now ready to serve inference requests for the specified model.

## Inference Flow

1.  **Request Reception**: A client sends an inference request (e.g., for a completion) to the dedicated **API Server Process** on its specific port (e.g., 8001).
2.  **Forward to Executor**: The `vLLM Engine` in the API Server calls `execute_model` on its `RemoteExecutor`.
3.  **RPC Request**: The `RemoteExecutor` serializes the request and places it onto the **request queue** connected to the `ProxyExecutor`.
4.  **Proxy Executor Dequeue**: The `ProxyExecutor`'s main loop retrieves the request from the queue.
5.  **Dispatch to Workers**: The `ProxyExecutor` uses its `collective_rpc` mechanism to send the `execute_model` command to the appropriate GPU workers assigned to that engine.
6.  **Model Execution**: The `gpu_worker` processes execute the model inference.
7.  **RPC Response**: The result is sent back from the workers to the `ProxyExecutor`, which then places it onto the **response queue** for the specific API Server.
8.  **Result Propagation**: The `RemoteExecutor` receives the result from the response queue and returns it to the `vLLM Engine`.
9.  **Stream to Client**: The engine streams the final output back to the client via the API Server.
