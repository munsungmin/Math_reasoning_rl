"""CPU regression cases for actual MATH reward failures and false positives."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from limit_rlvr_grade import grade


class MathRewardTest(unittest.TestCase):
    def test_equivalence_and_extraction(self):
        cases = [
            ("Therefore, $$", "1", False),
            (r"Therefore, \(\)", "1", False),
            (
                r"\boxed{3 + 2\sqrt{2}, 3 - 2\sqrt{2}}",
                r"3 \pm 2 \sqrt{2}",
                True,
            ),
            (r"\boxed{3-2\sqrt{2},3+2\sqrt{2}}", r"3\pm2\sqrt{2}", True),
            (r"\boxed{3+2\sqrt{2}}", r"3\pm2\sqrt{2}", False),
            (r"\boxed{3+2\sqrt{2},3+2\sqrt{2}}", r"3\pm2\sqrt{2}", False),
            (r"\boxed{(2,1)}", "(1,2)", False),
            (r"\boxed{0.5}", r"\frac{1}{2}", True),
            (r"\boxed{100}", "1", False),
            ("Answer: 10 meters", "10", True),
            ("The answer is _120°_.", "120", True),
            ("Work in progress: 10", "10", False),
            (r"\boxed{10", "10", False),
            ("<answer></answer>", "10", False),
            (r"\boxed{2240/78125}", r"\frac{448}{15625}", True),
            ("Therefore, 3 inches is 1/8 of 2 feet.", r"\frac{1}{8}", True),
            (
                r"Therefore, the expression is: \begin{align*}x(x(1+x)+2*x)-3*(x*x-x+2)&=x^3+3x-6\end{align*}",
                "x^3+3x-6",
                True,
            ),
            (r"Therefore, the answer is $x^3$.", "x^3+3x-6", False),
        ]
        for completion, answer, want in cases:
            with self.subTest(completion=completion):
                self.assertEqual(
                    grade(dict(completion=completion, answer=answer))[
                        "correct"
                    ],
                    want,
                )


if __name__ == "__main__":
    unittest.main()
