"""GPU dependencies and actual verl actor gradient smoke test; no model download."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3,4,5,6')
import importlib
import json
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.workers.actor.dp_actor import DataParallelPPOActor

for name in ['verl', 'vllm', 'xformers', 'transformers', 'ray', 'tensordict',
             'datasets', 'peft', 'verl.trainer.main_ppo',
             'verl.workers.fsdp_workers', 'verl.workers.sharding_manager.fsdp_vllm']:
    importlib.import_module(name)
    print('IMPORT PASS', name, flush=True)
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    x = torch.randn(64, 64, device=f'cuda:{i}', dtype=torch.float16)
    assert torch.isfinite(x @ x).all()
    print('CUDA PASS', i, torch.cuda.get_device_name(i), flush=True)
torch.manual_seed(42)
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=64,
    intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, max_position_embeddings=128,
    attn_implementation='sdpa')).cuda().float()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
actor = DataParallelPPOActor(OmegaConf.create(dict(use_remove_padding=False,
    ulysses_sequence_parallel_size=1, use_torch_compile=False, grad_clip=1.0)), model, optimizer)
ids = torch.randint(0, 256, (2, 16), device='cuda')
batch = dict(input_ids=ids, responses=ids[:, -4:], attention_mask=torch.ones_like(ids),
    position_ids=torch.arange(16, device='cuda').expand(2, -1))
before = next(model.parameters()).detach().clone()
entropy, log_probs = actor._forward_micro_batch(batch, temperature=1.0)
loss = -log_probs.mean() - 0.001 * entropy.mean()
assert torch.isfinite(loss)
loss.backward()
grad = actor._optimizer_step()
assert torch.isfinite(grad)
assert not torch.equal(before, next(model.parameters()).detach())
print('VERL ACTOR FORWARD/BACKWARD/OPTIMIZER PASS', float(loss), flush=True)
print('ALL GPU/ACTOR SMOKE TESTS PASSED', flush=True)
