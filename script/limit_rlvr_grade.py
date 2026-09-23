"""Compatibility entry point for the versioned math grader."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.graders.worker import *
if __name__ == "__main__":
    main()
