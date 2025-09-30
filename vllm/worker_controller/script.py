
from vllm import LLM, SamplingParams, config
from vllm.worker_controller.config.vllm_config import DummyModelConfig, DummyVllmConfig, CacheConfig, ParallelConfig

# modelConfig = DummyModelConfig("dummy", enforce_eager=True)
# cacheConfig = CacheConfig(gpu_memory_utilization=0.9)
# dummyvllmConfig = DummyVllmConfig(
#     model_config=modelConfig, cache_config=cacheConfig)


# print(dummyvllmConfig)
# print(modelConfig)
# llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", enforce_eager=True,
#           gpu_memory_utilization=0.9)
llm = LLM(model="facebook/opt-125m", enforce_eager=True,
          gpu_memory_utilization=0.9)
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
