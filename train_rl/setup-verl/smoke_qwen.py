"""Downloads Qwen2.5-1.5B (~3 GB) and tests FP16 vLLM generation."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'XFORMERS')
from vllm import LLM, SamplingParams
model = LLM(model='Qwen/Qwen2.5-1.5B', dtype='float16',
    tensor_parallel_size=1, max_model_len=256, max_num_seqs=2,
    gpu_memory_utilization=0.65, enforce_eager=True, enable_chunked_prefill=False)
outputs = model.generate(['Question: What is 2 + 2?\nAnswer:'],
                         SamplingParams(temperature=0, max_tokens=16))
assert len(outputs) == 1 and len(outputs[0].outputs[0].token_ids) > 0
print('QWEN2.5-1.5B VLLM GENERATION PASS', repr(outputs[0].outputs[0].text))
