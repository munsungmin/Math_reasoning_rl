"""Compatibility entry; implementation lives in scripts.analysis.plot_four_grpo_phases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis import plot_four_grpo_phases as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
