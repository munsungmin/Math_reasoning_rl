import unittest
import itertools
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "script/analysis")
)
from math_metrics import pass_at_k


class PassKTest(unittest.TestCase):
    def test_exact_small_sample_probabilities(self):
        for c in range(5):
            outcomes = [1] * c + [0] * (4 - c)
            for k in [1, 2, 4]:
                samples = list(itertools.combinations(outcomes, k))
                actual = sum(any(s) for s in samples) / len(samples)
                self.assertAlmostEqual(pass_at_k(4, c, k), actual)

    def test_no_extrapolation_or_invalid_counts(self):
        for args in [(4, 1, 64), (4, 5, 1), (4, -1, 1), (0, 0, 1)]:
            with self.assertRaises(ValueError):
                pass_at_k(*args)


if __name__ == "__main__":
    unittest.main()
