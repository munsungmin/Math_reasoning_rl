"""Compatibility entry point; canonical implementation is scripts.train.rl.runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train.rl import runner as _implementation
from scripts.train.rl.runner import *
if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        _implementation.run_worker(sys.argv[2])
    else:
        _implementation.main()
else:
    sys.modules[__name__] = _implementation
