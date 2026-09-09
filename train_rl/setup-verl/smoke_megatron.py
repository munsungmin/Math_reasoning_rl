"""Probe Megatron/Apex and the FlashAttention kernel required by verl Qwen."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
import json
import tempfile
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from apex.optimizers import FusedAdam
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
import flash_attn

report = {}
torch.cuda.set_device(0)
report['gpu'] = torch.cuda.get_device_name(0)
report['compute_capability'] = torch.cuda.get_device_capability(0)
print('GPU', report['gpu'], report['compute_capability'], flush=True)
with tempfile.TemporaryDirectory(prefix='verl-mcore-smoke-') as temp:
    dist.init_process_group('nccl', init_method='file://'+temp+'/rdzv', rank=0,
                            world_size=1, timeout=timedelta(seconds=120))
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                                  pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        cfg = ModelParallelConfig(use_cpu_initialization=True, params_dtype=torch.float32,
                                  gradient_accumulation_fusion=False)
        layer = tensor_parallel.ColumnParallelLinear(8, 16, config=cfg,
                    init_method=torch.nn.init.xavier_uniform_, gather_output=True).cuda()
        opt = FusedAdam(layer.parameters(), lr=1e-3)
        before = layer.weight.detach().clone()
        out, bias = layer(torch.randn(4, 2, 8, device='cuda'))
        loss = out.square().mean()
        loss.backward()
        assert torch.isfinite(layer.weight.grad).all()
        opt.step()
        assert not torch.equal(before, layer.weight)
        report['megatron_tensor_parallel_layer_and_apex_optimizer'] = 'PASS (TP=1)'
        print('MEGATRON FORWARD/BACKWARD + APEX OPTIMIZER PASS', flush=True)
        x = torch.randn(2, 16, device='cuda', dtype=torch.float16, requires_grad=True)
        weight = torch.ones(16, device='cuda', dtype=torch.float16, requires_grad=True)
        y = fused_rms_norm_affine(x, weight, (16,), 1e-6)
        y.float().sum().backward()
        assert torch.isfinite(x.grad).all()
        report['apex_rmsnorm'] = 'PASS'
        print('APEX RMSNORM FORWARD/BACKWARD PASS', flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
try:
    from verl.models.registry import ModelRegistry
    cls = ModelRegistry.load_model_cls('Qwen2ForCausalLM')
    assert cls is not None
    import verl.workers.megatron_workers
    report['verl_megatron_import'] = 'PASS: '+cls.__name__
    print('VERL MEGATRON MODEL/WORKER IMPORT PASS', flush=True)
except Exception as error:
    report['verl_megatron_import'] = f'{type(error).__name__}: {error}'
    print('VERL MEGATRON IMPORT FAILED', report['verl_megatron_import'], flush=True)
try:
    q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.float16)
    result = flash_attn.flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    report['flash_attention_2'] = 'PASS'
except Exception as error:
    report['flash_attention_2'] = f'{type(error).__name__}: {error}'
    print('FLASH ATTENTION 2 BLOCKED', report['flash_attention_2'], flush=True)
path = Path(__file__).with_name('megatron-smoke-result.json')
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
if not report['flash_attention_2'].startswith('PASS') or not report['verl_megatron_import'].startswith('PASS'):
    raise SystemExit(2)
print('MEGATRON COMPONENT SMOKE PASSED (full RL not tested)', flush=True)
