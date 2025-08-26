from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from worker_controller import WorkerController
from vllm.config import VllmConfig, ParallelConfig
import logging
import threading
import time

logger = logging.getLogger(__name__)


class EngineCreateRequest(BaseModel):
    engine_uuid: str
    vllm_config: Dict[str, Any]


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
controller_ready = threading.Event()
controller_error = None


def wait_for_controller():
    """Wait for controller to be ready, raise error if failed."""
    if not controller_ready.wait(timeout=30):
        raise HTTPException(
            status_code=503, detail="Worker controller initialization timeout")
    if controller_error:
        raise HTTPException(
            status_code=500, detail=f"Worker controller failed: {controller_error}")


@app.get("/")
def read_root():
    return {"message": "Worker Controller API", "status": "running", "controller_ready": controller_ready.is_set()}


@app.post("/engines")
def create_engine(request: EngineCreateRequest):
    """Create a new engine by assigning workers and loading vllm_config."""
    wait_for_controller()
    try:
        # Convert dict to VllmConfig object if needed
        vllm_config = request.vllm_config if isinstance(
            request.vllm_config, VllmConfig) else VllmConfig(**request.vllm_config)

        # Create engine: assign workers and load config
        worker_controller.create(vllm_config, request.engine_uuid)

        engine_info = worker_controller.engines[request.engine_uuid]
        return {
            "message": f"Engine {request.engine_uuid} created successfully",
            "assigned_workers": engine_info["workers"],
            "world_size": vllm_config.parallel_config.world_size
        }
    except Exception as e:
        logger.error(f"Failed to create engine {request.engine_uuid}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/engines/{engine_uuid}")
def delete_engine(engine_uuid: str):
    """Delete an engine and reset its workers to empty state."""
    wait_for_controller()
    try:
        worker_controller.delete(engine_uuid)
        return {
            "message": f"Engine {engine_uuid} deleted and workers reset successfully",
            "available_workers": worker_controller.available_workers
        }
    except Exception as e:
        logger.error(f"Failed to delete engine {engine_uuid}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/workers/status", response_model=WorkerStatusResponse)
def get_worker_status():
    """Get current worker and engine status."""
    wait_for_controller()
    return WorkerStatusResponse(
        available_workers=worker_controller.available_workers,
        total_workers=worker_controller.world_size,
        resource_status=worker_controller.resource.status(),
        active_engines={
            engine_id: {
                "workers": info["workers"],
                "world_size": info["vllm_config"].parallel_config.world_size
            }
            for engine_id, info in worker_controller.engines.items()
        }
    )


@app.post("/engines/{engine_uuid}/execute")
def execute_model(engine_uuid: str, request: EngineExecuteRequest):
    """Execute model on the specified engine."""
    wait_for_controller()
    try:
        if engine_uuid not in worker_controller.engines:
            raise HTTPException(
                status_code=404, detail=f"Engine {engine_uuid} not found")

        # Execute model using the configured workers
        # Note: This would require implementing execute_model method on WorkerController
        # that operates on specific engine's workers
        result = worker_controller.execute_model_for_engine(
            engine_uuid, request.scheduler_output)
        return {"result": result}
    except Exception as e:
        logger.error(f"Failed to execute model on engine {engine_uuid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines")
def list_engines():
    """List all active engines."""
    wait_for_controller()
    return {
        "engines": {
            engine_id: {
                "workers": info["workers"],
                "world_size": info["vllm_config"].parallel_config.world_size,
                "model": info["vllm_config"].model_config.model
            }
            for engine_id, info in worker_controller.engines.items()
        }
    }


@app.get("/engines/{engine_uuid}")
def get_engine_info(engine_uuid: str):
    """Get information about a specific engine."""
    wait_for_controller()
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
        controller_ready.set()
    except Exception as e:
        logger.error(f"Failed to initialize WorkerController: {e}")
        controller_error = str(e)
        controller_ready.set()


if __name__ == "__main__":

    # parallelConfig = ParallelConfig(world_size=3)
    # print(parallelConfig)
    worker_controller = WorkerController(number_of_gpus=3)
    print(worker_controller)
    # logging.basicConfig(level=logging.INFO)
    # logger.info("Starting FastAPI server...")

    # # Initialize WorkerController in background thread
    # init_thread = threading.Thread(target=init_worker_controller, daemon=True)
    # init_thread.start()

    # uvicorn.run(app, host="0.0.0.0", port=8000)
