# Documentation
The current vllm project has the workers tied to the executor. When a user creates a vllm instance, it creates a vllm engine which creates an executor which then creates the worker processes. The models are loaded onto the worker processes. Since we know that the number of worker processes can be determined before the runtime by the number of GPUs the system has, we want to decouple the executor from the worker processes to prevent cold start for the worker processes. You can view more [here](https://v5ira59l7f1.sg.larksuite.com/docx/Fb8CduOCSo0nqkx2H7vlIw0wgFf)

The code that has been modified is in the `vllm.worker_controller` folder.

## worker_controller.py
`init`
We will initialise the workecontroller with a dummy vllmconfig `vllm.worker_controller.config.vllmconfig`. An example is shown in `vllm.worker_controller.worker_controller`. This is to allow the worker processes to be set up successfully. Currently, the worker processes do not have a model runner, and this will be created when the model is loaded. When the user instantiates a workercontroller instance, we will create an empty executor in `vllm.worker_controller.executor.empty_executor`, which will create a Resorce allocater, which allocates which workers are in charge of which engine, and also create the empty worker processes, the worker controller will also have the message queue which will allow communication with the worker processes via RPC.

`create`
We will first check using the `Resource Allocater` whether we have enough GPUs to accomodate the request, we will create a process that `run_api_server` to listen on a different port. The `run_api_server` will create an engine which creates a executor.

**Not yet implemented** We want to pass in the message queue into the `run_api_server` such that the executor we create will use the message queue to communicate with our already created worker processes. After communication is set up, we will then load the model in the worker process.

## 