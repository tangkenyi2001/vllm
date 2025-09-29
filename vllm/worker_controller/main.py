from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from worker_controller import WorkerController
from vllm_config import DummyModelConfig, DummyVllmConfig, CacheConfig, ParallelConfig
from vllm.config import VllmConfig
import logging
import threading
import time

logger = logging.getLogger(__name__)


class EngineCreateRequest(BaseModel):
    engine_uuid: str
    vllm_config: Dict[str, Any]


class PipeRequest(BaseModel):
    rank: int
    method: str
    vllmConfig: Optional[Dict[str, Any]] = {}


class EngineExecuteRequest(BaseModel):
    engine_uuid: str
    scheduler_output: Dict[str, Any]


class WorkerStatusResponse(BaseModel):
    available_workers: int
    total_workers: int
    resource_status: Dict[int, Optional[str]]
    active_engines: Dict[str, Dict[str, Any]]


app = FastAPI(title="Worker Controller API")
# Initialize WorkerController with lazy loading
worker_controller = None


@app.get("/")
def read_root():
    return {"message": "Worker Controller API", "status": "running"}


@app.post("/engines")
def create_engine(request: EngineCreateRequest):
    """Create a new engine by assigning workers and loading vllm_config."""
    try:
        # Convert dict to VllmConfig object if needed
        vllm_config = request.vllm_config if isinstance(
            request.vllm_config, VllmConfig) else VllmConfig(**request.vllm_config)

        # Create engine: assign workers and load config
        worker_controller.create(vllm_config, request.engine_uuid)

        engine_info = worker_controller.executor.engines[request.engine_uuid]
        return {
            "message": f"Engine {request.engine_uuid} created successfully",
            "assigned_workers": engine_info["workers"],
            "world_size": vllm_config.parallel_config.world_size
        }
    except Exception as e:
        logger.error(f"Failed to create engine {request.engine_uuid}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pipecall")
def pipe_call(request: PipeRequest):
    vllm_config = request.vllmConfig if isinstance(
        request.vllmConfig, VllmConfig) else VllmConfig(**request.vllmConfig)

    kwargs = {
        "rank": request.rank,
        "method": request.method,
        "vllmconfig": vllm_config,
    }
    output = worker_controller.pipecall(**kwargs)
    return {
        "output": output
    }


@app.delete("/engines/{engine_uuid}")
def delete_engine(engine_uuid: str):
    """Delete an engine and reset its workers to empty state."""
    try:
        worker_controller.delete(engine_uuid)
        return {
            "message": f"Engine {engine_uuid} deleted and workers reset successfully",
            "available_workers": worker_controller.resource.num_free()
        }
    except Exception as e:
        logger.error(f"Failed to delete engine {engine_uuid}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/engines/{engine_uuid}/execute")
def execute_model(engine_uuid: str, request: EngineExecuteRequest):
    """Execute model on the specified engine."""
    try:
        if engine_uuid not in worker_controller.engines:
            raise HTTPException(
                status_code=404, detail=f"Engine {engine_uuid} not found")

        # Execute model using the configured workers
        # Note: This would require implementing execute_model method on WorkerController
        # that operates on specific engine's workers
        result = worker_controller.execute_model(
            engine_uuid, request.scheduler_output)
        return {"result": result}
    except Exception as e:
        logger.error(f"Failed to execute model on engine {engine_uuid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines/{engine_uuid}")
def get_engine_info(engine_uuid: str):
    """Get information about a specific engine."""
    if engine_uuid not in worker_controller.engines:
        raise HTTPException(
            status_code=404, detail=f"Engine {engine_uuid} not found")

    info = worker_controller.engines[engine_uuid]
    return {
        "engine_uuid": engine_uuid,
        "workers": info["workers"],
        "world_size": info["vllm_config"].parallel_config.world_size,
        "model": info["vllm_config"].model_config.model,
        "tensor_parallel_size": info["vllm_config"].parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": info["vllm_config"].parallel_config.pipeline_parallel_size
    }


def init_worker_controller():
    """Initialize WorkerController in background thread."""
    global worker_controller, controller_error
    try:
        logger.info("Initializing WorkerController...")
        worker_controller = WorkerController(number_of_gpus=2)
        logger.info(
            f"WorkerController initialized with {worker_controller.world_size} workers")
    except Exception as e:
        logger.error(f"Failed to initialize WorkerController: {e}")
        controller_error = str(e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,   # Show INFO and above
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # parallelConfig = ParallelConfig(world_size=3)
    # print(parallelConfig)
    modelConfig = DummyModelConfig("dummy", enforce_eager=True)
    cacheConfig = CacheConfig(gpu_memory_utilization=0.9)
    parallelConfig = ParallelConfig(
        world_size=2, worker_cls='vllm.worker_controller.gpu_worker.Worker')
    # parallelConfig = ParallelConfig(
    #     world_size=2)
    dummyvllmConfig = DummyVllmConfig(
        model_config=modelConfig, cache_config=cacheConfig, parallel_config=parallelConfig)
    worker_controller = WorkerController(vllm_config=dummyvllmConfig)
    logger.info("WorkerController started")

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # logging.basicConfig(level=logging.INFO)
    # logger.info("Starting FastAPI server...")

    # # Initialize WorkerController in background thread
    # init_thread = threading.Thread(target=init_worker_controller, daemon=True)
    # init_thread.start()

    # uvicorn.run(app, host="0.0.0.0", port=8000)
