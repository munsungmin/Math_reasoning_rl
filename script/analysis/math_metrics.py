"""Compatibility import for observed-sample pass@k."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.reasoning.metric.functional.probability import pass_at_k
