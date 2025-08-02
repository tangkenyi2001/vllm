# vLLM Architecture Diagrams

This document contains two diagrams that illustrate the architecture of the vLLM engine and its workers.

## 1. vLLM Engine Architecture

This diagram shows the flow of requests from the client to the vLLM engine. The engine manages tokenization, scheduling, and processing, then dispatches tasks to the workers.

```mermaid
graph TD
    subgraph "Client & Entrypoints"
        direction LR
        ClientApp[User Application] --> LLM_Class[LLM Class]
        ClientApp -- "HTTP" --> API_Server[API Server]
    end

    subgraph "vLLM Engine"
        direction LR
        subgraph "Entrypoint Interface"
            AsyncEngine[AsyncLLMEngine]
        end
        
        subgraph "Core Engine"
            Engine[LLMEngine]
            Tokenizer[Tokenizer]
            Scheduler[Scheduler]
            OutputProcessor[Output Processor]
        end

        AsyncEngine --> Engine
        Engine --> Tokenizer
        Engine --> Scheduler
        Engine --> OutputProcessor
    end

    %% Define Flow
    LLM_Class -- "Adds request" --> AsyncEngine
    API_Server -- "Adds request" --> AsyncEngine
    
    AsyncEngine -- "1. Tokenize" --> Tokenizer
    Tokenizer -- "Input IDs" --> Scheduler
    
    Scheduler -- "2. Dispatch Batched Inputs" --> ToWorkers((To Workers))
    
    FromWorkers((From Workers)) -- "3. Receive Generated IDs" --> OutputProcessor
    OutputProcessor -- "4. Detokenize Text" --> AsyncEngine
    
    AsyncEngine -- "5. Stream Final Output" --> ClientApp

    %% Styling
    classDef client fill:#c9d,stroke:#333,stroke-width:2px;
    classDef entrypoint fill:#ccf,stroke:#333,stroke-width:2px;
    classDef engine fill:#9cf,stroke:#333,stroke-width:2px;
    classDef flow fill:#f99,stroke:#333,stroke-width:2px;

    class ClientApp,LLM_Class,API_Server client;
    class AsyncEngine,Engine,Scheduler,Tokenizer,OutputProcessor engine;
    class ToWorkers,FromWorkers flow;
```

## 2. vLLM Worker Architecture

This diagram illustrates how workers receive tasks from the engine, execute the model in a distributed fashion across multiple GPUs, and send results back.

```mermaid
graph TD
    FromEngine((From Engine))

    subgraph "Workers (Distributed on GPUs)"
        subgraph "Worker 1"
            W1[Worker] --> MR1[ModelRunner] --> M1[Model Partition 1] --> GPU1[(GPU 0)]
        end
        subgraph "Worker 2"
            W2[Worker] --> MR2[ModelRunner] --> M2[Model Partition 2] --> GPU2[(GPU 1)]
        end
        subgraph "..."
            Wn[...]
        end
    end

    ToEngine((To Engine))

    %% Define Flow
    FromEngine -- "Batched Inputs" --> W1
    FromEngine -- "Batched Inputs" --> W2
    FromEngine -- "Batched Inputs" --> Wn

    W1 <== "All-Reduce / P2P Communication" ==> W2
    W2 <== "All-Reduce / P2P Communication" ==> Wn

    W1 -- "Generated Token IDs" --> ToEngine
    W2 -- "Generated Token IDs" --> ToEngine
    Wn -- "Generated Token IDs" --> ToEngine

    %% Styling
    classDef worker fill:#f99,stroke:#333,stroke-width:2px;
    classDef flow fill:#9cf,stroke:#333,stroke-width:2px;

    class W1,MR1,M1,GPU1,W2,MR2,M2,GPU2,Wn worker;
    class FromEngine,ToEngine flow;
```
