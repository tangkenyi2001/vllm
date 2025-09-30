from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from worker_controller import WorkerController
from vllm.worker_controller.config.vllm_config import DummyModelConfig, DummyVllmConfig, CacheConfig, ParallelConfig
from vllm.config import VllmConfig
import logging
import threading
import time

logger = logging.getLogger(__name__)


class EngineCreateRequest(BaseModel):
    engine_uuid: str
    vllm_config: Dict[str, Any]


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,   # Show INFO and above
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    worker_controller = WorkerController()
    logger.info("WorkerController started")

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # logging.basicConfig(level=logging.INFO)
    # logger.info("Starting FastAPI server...")

    # # Initialize WorkerController in background thread
    # init_thread = threading.Thread(target=init_worker_controller, daemon=True)
    # init_thread.start()

    # uvicorn.run(app, host="0.0.0.0", port=8000)
