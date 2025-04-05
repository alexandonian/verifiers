import os
from openai import OpenAI

import verifiers as vf
from verifiers.tools import calculator
from verifiers.utils import preprocess_dataset, write_json
from verifiers.prompts import DEFAULT_CODE_TOOL_PROMPT_TEMPLATE

"""
Evaluating multi-turn reasoning before/after training.

CUDA_VISIBLE_DEVICES=0,1 vllm serve 'Qwen/Qwen2.5-7B-Instruct' --tensor_parallel_size 2 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching \
    --host 0.0.0.0 --port 8001

uv run verifiers/examples/math_eval.py
"""

RESULTS_DIR = "results"
dataset_name = "project_euler"
split = "train"
dataset = preprocess_dataset(dataset_name, split)
print(dataset)
vf_env = vf.CodeToolEnv(
    eval_dataset=dataset,
    system_prompt=DEFAULT_CODE_TOOL_PROMPT_TEMPLATE,
    few_shot=[],
    tools=[calculator],
    max_steps=8,
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "output/checkpoint-100"
base_url = "http://0.0.0.0:8001/v1"
client = OpenAI(base_url=base_url, api_key="EMPTY")
output = vf_env.eval_api(
    client, model_name, max_concurrent=10, sampling_args={"temperature": 0.6}
)
output_filename = os.path.join(
    RESULTS_DIR, f"{dataset_name}-{split}-{model_name.replace('/', '-')}.json"
)
write_json(output, output_filename)
