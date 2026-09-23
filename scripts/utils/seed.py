"""Independent observation/inference RNG; torch is loaded only inside workers."""

import random
from contextlib import contextmanager


@contextmanager
def isolated_rng(seed=None):
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Every supported worker owns one device. Do not initialize other ranks' GPUs.
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % 2**32)
                # Do not initialize CUDA in CPU configuration/evaluation jobs.
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
