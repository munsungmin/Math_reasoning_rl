"""Extracted native verl worker; legacy run semantics are preserved."""

import signal
import sys
from pathlib import Path


def run_worker(config_path):
    """Only this explicitly launched child imports CUDA-facing trainer modules."""
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import main as verl_main

    def stop_worker(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_worker)
    from scripts.utils.manifest import record_environment

    record_environment(Path(config_path).parent)
    try:
        verl_main(OmegaConf.load(config_path))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    run_worker(sys.argv[1])
