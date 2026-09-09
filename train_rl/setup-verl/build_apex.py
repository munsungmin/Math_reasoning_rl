"""Build Apex's CUDA extensions without probing live GPUs; target SM75 explicitly."""
import os
import runpy
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.1')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')
os.environ.setdefault('MAX_JOBS', '4')
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
import torch
# This process only builds a wheel; it never executes GPU operations.
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
os.chdir(Path(__file__).resolve().parents[1] / 'third_party/apex')
sys.argv = ['setup.py', '--cpp_ext', '--cuda_ext', 'bdist_wheel']
runpy.run_path('setup.py', run_name='__main__')
