import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3,4,5,6')
local_rank = int(os.environ['LOCAL_RANK'])
physical_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank]
os.environ['CUDA_VISIBLE_DEVICES'] = physical_gpu
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
import fcntl
import tempfile
from datetime import timedelta
import torch
import torch.distributed as dist
rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group('gloo', timeout=timedelta(seconds=600))
# Serialize CUDA context initialization on this older driver.
with open(os.path.join(tempfile.gettempdir(), f'verl-smoke-cuda-{os.getuid()}.lock'), 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.cuda.set_device(0)
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    print(f'CUDA INIT PASS rank={rank} physical_gpu={physical_gpu}', flush=True)
dist.barrier()
print(f'GLOO BARRIER PASS rank={rank}', flush=True)
nccl_group = dist.new_group(backend='nccl', timeout=timedelta(seconds=120))
print(f'NCCL GROUP PASS rank={rank}', flush=True)
x = torch.tensor([rank + 1.0], device='cuda')
dist.all_reduce(x, group=nccl_group)
n = dist.get_world_size()
assert x.item() == n * (n + 1) / 2
print(f'NCCL PASS rank={rank} sum={x.item()}', flush=True)
dist.destroy_process_group()
