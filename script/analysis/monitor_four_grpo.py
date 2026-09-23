"""Compatibility entry; implementation lives in scripts.analysis.monitor_four_grpo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis import monitor_four_grpo as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
