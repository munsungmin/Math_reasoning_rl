"""Compatibility entry; implementation lives in scripts.preprocessing.prepare_math500_grpo_eval."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing import prepare_math500_grpo_eval as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
