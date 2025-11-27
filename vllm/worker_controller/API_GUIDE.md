# Worker Controller REST API Guide

Quick guide for using the Worker Controller REST API to manage vLLM engines with RemoteExecutor.

## Quick Start

### 1. Start the Worker Controller

```bash
cd vllm/worker_controller
python main.py
```

API available at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### 2. Create an Engine

```bash
curl -X POST http://localhost:8000/engines \
  -H "Content-Type: application/json" \
  -d '{
    "engine_uuid": "my-engine",
    "model": "facebook/opt-125m",
    "dtype": "float16",
    "gpu_memory_utilization": 0.9
  }'
```

Response:
```json
{
  "engine_uuid": "my-engine",
  "status": "created",
  "api_url": "http://localhost:8001",
  "port": 8001,
  "assigned_ranks": [0],
  "model": "facebook/opt-125m",
  "pid": 12345
}
```

### 3. Send Inference Requests

```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "San Francisco is a",
    "max_tokens": 20
  }'
```

### 4. Delete the Engine

```bash
curl -X DELETE http://localhost:8000/engines/my-engine
```

## API Endpoints

### Controller API (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/engines` | POST | Create engine |
| `/engines` | GET | List all engines |
| `/engines/{uuid}` | GET | Get engine status |
| `/engines/{uuid}` | DELETE | Delete engine |
| `/workers` | GET | List worker assignments |

### Engine API (Port 8001+)

Each created engine exposes standard vLLM OpenAI-compatible endpoints:
- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`
- `/health`

## Test Script

Run the full workflow test:

```bash
# Terminal 1: Start controller
python main.py

# Terminal 2: Run test
cd tests
python test_main_api.py
```

## Create Engine Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_uuid` | string | required | Unique ID for engine |
| `model` | string | required | Model name/path |
| `dtype` | string | "float16" | Data type |
| `gpu_memory_utilization` | float | 0.9 | GPU memory % |
| `tensor_parallel_size` | int | 1 | TP size |
| `pipeline_parallel_size` | int | 1 | PP size |

## Example: Python Client

```python
import requests

# Create engine
response = requests.post("http://localhost:8000/engines", json={
    "engine_uuid": "test-001",
    "model": "facebook/opt-125m",
    "gpu_memory_utilization": 0.9
})
engine = response.json()
api_url = engine["api_url"]

# Send inference request
response = requests.post(f"{api_url}/v1/completions", json={
    "model": "facebook/opt-125m",
    "prompt": "Hello, world!",
    "max_tokens": 10
})
result = response.json()
print(result["choices"][0]["text"])

# Delete engine
requests.delete("http://localhost:8000/engines/test-001")
```

## Architecture

```
Client Request
     ↓
Controller API (port 8000)
     ↓
ProxyExecutor (assigns workers, loads model)
     ↓
API Server Process (port 8001+)
     ↓
RemoteExecutor (forwards RPC)
     ↓
GPU Workers (execute inference)
```

## Features

- **Shared Worker Pool**: Multiple engines share GPU workers
- **Dynamic Allocation**: Workers assigned on-demand
- **Auto Port Assignment**: Ports allocated automatically
- **Process Isolation**: Each engine in separate process
- **Clean Lifecycle**: Proper resource cleanup on delete

See `README.md` for detailed architecture documentation.
