# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import traceback
import weakref
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any, Callable, Optional, Union, cast

import cloudpickle
import logging
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (decorate_logs, get_distributed_init_method,
                        get_loopback_ip, get_mp_context, get_open_port,
                        set_process_title)
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm_config import DummyModelConfig, DummyVllmConfig, CacheConfig, ParallelConfig
from vllm.engine.llm_engine import LLMEngine

from multiprocessing import Pipe, Process
from typing import List
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.worker_controller.api_server import run_server
from vllm.utils import (Device, FlexibleArgumentParser, decorate_logs,
                        get_open_zmq_ipc_path, is_valid_ipv6_address,
                        set_ulimit)
from vllm.worker_controller.executor import ModifiedExecutor
import uvloop
logger = init_logger(__name__)
logging.basicConfig(
    level=logging.INFO,   # Show INFO and above
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class ResourceAllocator:
    def __init__(self, numberOfGPUs: int, start_port: int = 8001):
        self.resources = {i: 0 for i in range(numberOfGPUs)}
        self.uuid_to_port = {}  # uid -> port
        self.rank_to_uid = {}  # rank -> uid
        self.start_port = start_port
        self.next_port = start_port

    def assign(self, num: int, uuid: str):
        if uuid in self.port_map:
            # UID already has a port
            port = self.port_map[uuid]
        else:
            port = self.next_port
            self.port_map[uuid] = port
            self.next_port += 1

        assigned_ranks = []
        for rank, val in self.resources.items():
            if val == 0 and len(assigned_ranks) < num:
                self.resources[rank] = uuid
                self.rank_to_uid[rank] = uuid
                assigned_ranks.append(rank)

        if len(assigned_ranks) < num:
            # Rollback partial assignment
            for rank in assigned_ranks:
                self.resources[rank] = 0
                del self.rank_to_uid[rank]
            raise RuntimeError(
                f"Only {len(assigned_ranks)} free resources, requested {num}"
            )

        return assigned_ranks, port

    def get_ranks_by_uuid(self, uid: str):
        """Return list of ranks assigned to the specified UUID."""
        return [rank for rank, val in self.resources.items() if val == uid]

    def get_port_by_uuid(self, uid: str):
        """Return the port assigned to the specified UID."""
        return self.port_map.get(uid)

    def release_by_uuid(self, uid: str):
        """Release all ranks assigned to this UID."""
        released_ranks = []
        for rank, val in self.resources.items():
            if val == uid:
                self.resources[rank] = 0
                del self.rank_to_uid[rank]
                released_ranks.append(rank)
        if uid in self.port_map:
            port = self.port_map[uid]
            del self.port_map[uid]
        return released_ranks, port


class WorkerController:

    def __init__(self, vllm_config: VllmConfig) -> None:
        # Executor will create the empty worker processes
        self.executor = ModifiedExecutor(vllm_config=vllm_config)
        self.resourceAllocater = ResourceAllocator(
            numberOfGPUs=vllm_config.parallel_config.world_size)

    # Create API server using our own executor
    def create(self, vllm_config: VllmConfig, engineUUID: str):
        # We will allocate first.
        # When it is created, we will create an api layer and use our own executor
        assigned_ranks, port = self.resourceAllocater.assign(
            num=vllm_config.parallel_config.world_size, uuid=engineUUID)
        # need to create the api layer using the port allocated and use our own executor and take notes of the ranks of the pipes to use.

        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args()

        args.executor = self.executor
        args.assigned_ranks = assigned_ranks
        args.vllmconfig = vllm_config
        args.port = port
        # should pass the executor in to run,
        # the executor also has the pipes available

        def run_api_server(args):
            uvloop.run(run_server(args))

        proc = Process(target=run_api_server, args=(args,), daemon=False)
        proc.start()

    def delete(self, engineUUID: str):
        releasedrank, port = self.resourceAllocater.release_by_uuid(engineUUID)
        # kill the process on the port, kills the api layer.
